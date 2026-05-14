"""
Phase 1: Train the NAMM mirror maps (g_phi, f_psi).

Usage:
    python train_mirror_map.py \
        --data_root /scratch/oywang/cs159 \
        --output_dir /scratch/oywang/cs159/checkpoints/phase1 \
        --epochs 100 \
        --batch_size 16

On SLURM, submit via the accompanying train_mirror_map.slurm script.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dataset import build_dataloaders
from icnn import ICNN, ICNNGradient
from inverse_map import InverseMap
from losses import ConstraintLoss, namm_loss


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Phase 1: Train NAMM mirror maps')
    p.add_argument('--data_root',    required=True,
                   help='Path to SEN12MS-CR data root')
    p.add_argument('--output_dir',   required=True,
                   help='Where to save checkpoints and logs')
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
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--log_every',    type=int, default=50,
                   help='Log every N batches')
    p.add_argument('--save_every',   type=int, default=10,
                   help='Save checkpoint every N epochs')
    p.add_argument('--resume',       type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--fp16',         action='store_true',
                   help='Use mixed-precision training')
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


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(g_phi, f_psi, constraint_loss, val_loader, device, args):
    g_phi.eval(); f_psi.eval()
    total_loss = 0.0
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
        total_loss += losses['loss'].item() * x.size(0)
        n += x.size(0)
    g_phi.train(); f_psi.train()
    return total_loss / max(n, 1)


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        roi=args.roi,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    # ── Training loop ────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, 'train_log.txt')
    with open(log_path, 'a') as log_f:
        log_f.write(f'epoch,step,loss,l_cycle,l_constr,l_reg,elapsed\n')

    for epoch in range(start_epoch, args.epochs):
        g_phi.train(); f_psi.train()
        epoch_start = time.time()

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

            if step % args.log_every == 0:
                elapsed = time.time() - epoch_start
                msg = (f'Ep {epoch+1}/{args.epochs} | '
                       f'step {step} | '
                       f'loss={losses["loss"].item():.4f} | '
                       f'cyc={losses["l_cycle"].item():.4f} | '
                       f'constr={losses["l_constr"].item():.4f} | '
                       f'reg={losses["l_reg"].item():.4f} | '
                       f't={elapsed:.1f}s')
                print(msg)
                with open(log_path, 'a') as log_f:
                    log_f.write(
                        f'{epoch+1},{step},'
                        f'{losses["loss"].item():.6f},'
                        f'{losses["l_cycle"].item():.6f},'
                        f'{losses["l_constr"].item():.6f},'
                        f'{losses["l_reg"].item():.6f},'
                        f'{elapsed:.1f}\n')

        # ── Validation ───────────────────────────────────────────────────────
        val_loss = validate(g_phi, f_psi, constraint_loss,
                            val_loader, device, args)
        print(f'  Val loss: {val_loss:.4f}  (best: {best_val_loss:.4f})')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(args.output_dir, 'best.pt'),
                epoch + 1, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f,
                best_val_loss)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f'ckpt_ep{epoch+1:04d}.pt'),
                epoch + 1, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f,
                best_val_loss)

    # ── Save final ───────────────────────────────────────────────────────────
    save_checkpoint(
        os.path.join(args.output_dir, 'final.pt'),
        args.epochs, g_phi, f_psi, opt_g, opt_f, ema_g, ema_f,
        best_val_loss)
    print('Phase 1 training complete.')


if __name__ == '__main__':
    main()
