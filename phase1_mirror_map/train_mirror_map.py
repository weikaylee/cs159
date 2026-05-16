"""
Phase 1: Train the NAMM mirror maps (g_phi, f_psi).

Usage:
    python train_mirror_map.py \
        --data_root /scratch/oywang/cs159 \
        --output_dir /scratch/oywang/cs159/checkpoints/phase1 \
        --epochs 100 \
        --batch_size 16

To stream metrics to Weights & Biases (online):
    python train_mirror_map.py ... --wandb --wandb_project cs159

On SLURM, submit via the accompanying train_mirror_map.slurm script.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import wandb

from dataset import build_dataloaders
from icnn import ICNN, ICNNGradient
from inverse_map import InverseMap
from losses import ConstraintLoss, namm_loss, reconstruction_metrics


HISTORY_COLUMNS = [
    'phase', 'epoch', 'step', 'elapsed_s',
    'loss', 'l_cycle', 'l_constr', 'l_reg',
    'mae', 'sam', 'psnr', 'ssim',
]


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Phase 1: Train NAMM mirror maps')
    p.add_argument('--data_root',    required=True,
                   help='Path to SEN12MS-CR data root')
    p.add_argument('--output_dir',   default=None,
                   help='Where to save checkpoints and logs')
    p.add_argument('--run_name',     type=str, default=None,
                   help='Run name (used for default output_dir)')
    p.add_argument('--roi',          default='ROIs1158_spring',
                   help='Which ROI to train on')

    # Model
    p.add_argument('--n_channels',   type=int, default=13)
    p.add_argument('--ngf',          type=int, default=64,
                   help='Base feature maps for f_psi')
    p.add_argument('--n_res_blocks', type=int, default=6)
    p.add_argument('--icnn_filters', type=int, default=64)
    p.add_argument('--icnn_layers',  type=int, default=6)
    p.add_argument('--strong_convexity', type=float, default=0.3)

    # Training
    p.add_argument('--epochs',       type=int, default=100)
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--lr',           type=float, default=2e-4)
    p.add_argument('--max_sigma',    type=float, default=0.1,
                   help='Max noise level for inverse map robustness')
    p.add_argument('--style_weight', type=float, default=100.0)
    p.add_argument('--cycle_weight', type=float, default=1.0)
    p.add_argument('--constr_weight',type=float, default=1.0)
    p.add_argument('--reg_weight',   type=float, default=0.001)
    p.add_argument('--ema_rate',     type=float, default=0.999)

    # Misc
    p.add_argument('--patch_size',   type=int, default=None,
                   help='Random-crop spatial size (default: full 256x256)')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--log_every',    type=int, default=50,
                   help='Log every N batches')
    p.add_argument('--resume',       type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--fp16',         action='store_true',
                   help='Use mixed-precision training')

    # Wandb
    p.add_argument('--wandb',          action='store_true',
                   help='Stream metrics to Weights & Biases (online).')
    p.add_argument('--wandb_project',  type=str, default='cs159')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--wandb_entity',   type=str, default=None)
    return p.parse_args()


# ── EMA helper ────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, rate: float = 0.999):
        self.rate = rate
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.rate * self.shadow[k] + (1 - self.rate) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, g_phi, f_psi, opt_g, opt_f,
                    ema_g, ema_f, best_val_loss):
    torch.save({
        'epoch':         epoch,
        'g_phi':         g_phi.state_dict(),
        'f_psi':         f_psi.state_dict(),
        'opt_g':         opt_g.state_dict(),
        'opt_f':         opt_f.state_dict(),
        'ema_g':         ema_g.shadow,
        'ema_f':         ema_f.shadow,
        'best_val_loss': best_val_loss,
    }, path)
    print(f'  Saved checkpoint → {path}')


def load_checkpoint(path, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f):
    ckpt = torch.load(path, map_location='cpu')
    g_phi.load_state_dict(ckpt['g_phi'])
    f_psi.load_state_dict(ckpt['f_psi'])
    opt_g.load_state_dict(ckpt['opt_g'])
    opt_f.load_state_dict(ckpt['opt_f'])
    ema_g.shadow = ckpt['ema_g']
    ema_f.shadow = ckpt['ema_f']
    print(f'  Resumed from epoch {ckpt["epoch"]} ({path})')
    return ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))


# ── History CSV helpers ───────────────────────────────────────────────────────

def _append_history_row(path: str, row: dict) -> None:
    """Append a row to history.csv, writing the header on first call."""
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in HISTORY_COLUMNS})


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(g_phi, f_psi, constraint_loss, val_loader, device, args):
    g_phi.eval(); f_psi.eval()
    keys = ('loss', 'l_cycle', 'l_constr', 'l_reg',
            'mae', 'sam', 'psnr', 'ssim')
    totals = {k: 0.0 for k in keys}
    n = 0
    for x in val_loader:
        x = x.to(device)
        losses = namm_loss(
            g_phi, f_psi, constraint_loss, x,
            max_sigma=args.max_sigma,
            cycle_weight=args.cycle_weight,
            constraint_weight=args.constr_weight,
            reg_weight=args.reg_weight,
            strong_convexity=args.strong_convexity,
            device=device,
        )
        metrics = reconstruction_metrics(losses['x_recon'], x)
        bsz = x.size(0)
        for k in ('loss', 'l_cycle', 'l_constr', 'l_reg'):
            totals[k] += losses[k].item() * bsz
        for k in ('mae', 'sam', 'psnr', 'ssim'):
            totals[k] += metrics[k].item() * bsz
        n += bsz
    g_phi.train(); f_psi.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args = get_args()
    if args.output_dir is None:
        run_name = args.run_name or args.wandb_run_name or 'train_mirror_map'
        args.output_dir = os.path.join(os.getcwd(), 'runs', run_name)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        roi=args.roi,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    icnn = ICNN(
        n_in_channels=args.n_channels,
        n_layers=args.icnn_layers,
        n_filters=args.icnn_filters,
        strong_convexity=args.strong_convexity,
    ).to(device)
    g_phi = ICNNGradient(icnn).to(device)

    f_psi = InverseMap(
        n_channels=args.n_channels,
        ngf=args.ngf,
        n_res_blocks=args.n_res_blocks,
        residual=True,
    ).to(device)

    constraint_loss = ConstraintLoss(
        style_weight=args.style_weight,
        n_input_channels=args.n_channels,
    ).to(device)

    # ── Optimisers ───────────────────────────────────────────────────────────
    # Separate optimisers for g_phi and f_psi, as in the original NAMM code
    opt_g = torch.optim.Adam(g_phi.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_f = torch.optim.Adam(f_psi.parameters(), lr=args.lr, betas=(0.5, 0.999))

    ema_g = EMA(g_phi, rate=args.ema_rate)
    ema_f = EMA(f_psi, rate=args.ema_rate)

    scaler = GradScaler(enabled=args.fp16)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f)

    log_path = os.path.join(args.output_dir, 'history.csv')

    for epoch in range(start_epoch, args.epochs):
        g_phi.train(); f_psi.train()
        epoch_start = time.time()

        train_keys = ('loss', 'l_cycle', 'l_constr', 'l_reg',
                      'mae', 'sam', 'psnr', 'ssim')
        train_totals = {k: 0.0 for k in train_keys}
        train_n = 0

        for step, x in enumerate(train_loader):
            x = x.to(device, non_blocking=True)

            opt_g.zero_grad(set_to_none=True)
            opt_f.zero_grad(set_to_none=True)

            with autocast(enabled=args.fp16):
                losses = namm_loss(
                    g_phi, f_psi, constraint_loss, x,
                    max_sigma=args.max_sigma,
                    cycle_weight=args.cycle_weight,
                    constraint_weight=args.constr_weight,
                    reg_weight=args.reg_weight,
                    strong_convexity=args.strong_convexity,
                    device=device,
                )

            scaler.scale(losses['loss']).backward()
            scaler.unscale_(opt_g)
            scaler.unscale_(opt_f)
            torch.nn.utils.clip_grad_norm_(g_phi.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(f_psi.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.step(opt_f)
            scaler.update()

            ema_g.update(g_phi)
            ema_f.update(f_psi)

            with torch.no_grad():
                batch_metrics = reconstruction_metrics(losses['x_recon'], x)

            bsz = x.size(0)
            train_totals['loss'] += losses['loss'].item() * bsz
            train_totals['l_cycle'] += losses['l_cycle'].item() * bsz
            train_totals['l_constr'] += losses['l_constr'].item() * bsz
            train_totals['l_reg'] += losses['l_reg'].item() * bsz
            train_totals['mae'] += batch_metrics['mae'].item() * bsz
            train_totals['sam'] += batch_metrics['sam'].item() * bsz
            train_totals['psnr'] += batch_metrics['psnr'].item() * bsz
            train_totals['ssim'] += batch_metrics['ssim'].item() * bsz
            train_n += bsz

            if step % args.log_every == 0:
                elapsed = time.time() - epoch_start
                msg = (f'Ep {epoch+1}/{args.epochs} | step {step} | '
                       f'loss={losses["loss"].item():.4f} | '
                       f'cyc={losses["l_cycle"].item():.4f} | '
                       f'constr={losses["l_constr"].item():.4f} | '
                       f'reg={losses["l_reg"].item():.4f} | '
                       f'psnr={batch_metrics["psnr"].item():.2f} | '
                       f'ssim={batch_metrics["ssim"].item():.3f} | '
                       f't={elapsed:.1f}s')
                print(msg)

        # ── Train epoch metrics ─────────────────────────────────────────────
        train_den = max(train_n, 1)
        train_avg = {k: v / train_den for k, v in train_totals.items()}

        _append_history_row(log_path, {
            'phase':     'train',
            'epoch':     epoch + 1,
            'step':      '',
            'elapsed_s': round(time.time() - epoch_start, 2),
            **train_avg,
        })

        if args.wandb:
            train_step = (epoch + 1) * len(train_loader) - 1
            wandb.log({
                'train/loss':    train_avg['loss'],
                'train/l_cycle': train_avg['l_cycle'],
                'train/l_constr': train_avg['l_constr'],
                'train/l_reg':   train_avg['l_reg'],
                'train/mae':     train_avg['mae'],
                'train/sam':     train_avg['sam'],
                'train/psnr':    train_avg['psnr'],
                'train/ssim':    train_avg['ssim'],
                'train/lr':      opt_g.param_groups[0]['lr'],
                'train/epoch':   epoch + 1,
            }, step=train_step)

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = validate(g_phi, f_psi, constraint_loss,
                               val_loader, device, args)
        val_loss = val_metrics['loss']
        print(f'  Val loss: {val_loss:.4f}  (best: {best_val_loss:.4f})')

        _append_history_row(log_path, {
            'phase':     'val',
            'epoch':     epoch + 1,
            'step':      '',
            'elapsed_s': round(time.time() - epoch_start, 2),
            **val_metrics,
        })

        if args.wandb:
            val_step = (epoch + 1) * len(train_loader) - 1
            wandb.log(
                {f'val/{k}': v for k, v in val_metrics.items()}
                | {'val/best_loss': min(best_val_loss, val_loss)},
                step=val_step,
            )

        # ── Checkpoint: best.pt + last.pt ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(args.output_dir, 'best.pt'),
                epoch + 1, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f,
                best_val_loss)

        if epoch == args.epochs - 1:
            save_checkpoint(
                os.path.join(args.output_dir, 'last.pt'),
                epoch + 1, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f,
                best_val_loss)

    if args.wandb:
        wandb.finish()
    print('Phase 1 training complete.')


if __name__ == '__main__':
    main()
