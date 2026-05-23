"""
Phase 1 loss-ablation sweep.

Trains the NAMM mirror maps once per loss configuration by subprocess-launching
train_mirror_map.py, then collates each run's best validation metrics into
ablation_summary.csv. Keeping train_mirror_map.py as the single training
entry-point means there is no duplicated training loop to drift.

Two config families differ in how the constraint distance l_constr is built:

    vgg       L_recon + dsw*L_dis + stw*L_style                   (VGG terms)
    spectral  L_recon + sw*L_sam + mw*L_moments                   (spectral terms)

The vgg family sweeps style_weight x dis_weight over --vgg_style_values
(default 10, 100, 1000) x --vgg_dis_values (default 0.1, 1, 10), anchored on
the paper's (style=100, dis=1) which sits at the grid centre. The spectral
family sweeps sam_weight x moment_weight over --sweep_values (default 0.1,
1, 10). Default sweep => 9 + 9 = 18 configs.

Usage:
    # inspect the configs (no training)
    python run_loss_ablation.py --list
    python run_loss_ablation.py --data_root <DATA> --dry_run

    # run on local test data
    python phase1_mirror_map/run_loss_ablation.py \
      --data_root ./data \
      --output_root runs/run_loss_ablation \
      --roi all \
      --epochs 1 --batch_size 8 --patch_size 64 --num_workers 0 \
      --wandb --wandb_project tests \

    # run the whole sweep sequentially
    python run_loss_ablation.py --data_root <DATA> --output_root <OUT> --fp16 --wandb

    # run a single config (SLURM array task)
    python run_loss_ablation.py --data_root <DATA> --output_root <OUT> \
        --config_index $SLURM_ARRAY_TASK_ID --fp16 --wandb

    # rebuild ablation_summary.csv from finished runs
    python run_loss_ablation.py --output_root <OUT> --collate_only

The --config_index ordering is stable for fixed sweep values: indices 0-8
are the vgg style x dis grid, 9-17 the spectral sam x moment grid. Pass the
same --sweep_values / --vgg_style_values / --vgg_dis_values when collating
that you passed when training.
"""

import argparse
import csv
import os
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = 'train_mirror_map.py'

SUMMARY_COLUMNS = [
    'config', 'family', 'dis_weight', 'style_weight',
    'sam_weight', 'moment_weight', 'best_epoch',
    'val_loss', 'val_mae', 'val_sam', 'val_psnr', 'val_ssim', 'status',
]


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Phase 1 loss-ablation sweep over train_mirror_map.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Sweep / data
    p.add_argument('--data_root', default=None,
                   help='SEN12MS-CR data root (required unless --list/--collate_only)')
    p.add_argument('--output_root', default=os.path.join(os.getcwd(),
                                                         'runs', 'loss_ablation'),
                   help='Parent dir; each config writes to <output_root>/<config>')
    p.add_argument('--roi', default='all',
                   help="ROI passed to train_mirror_map.py ('all' = every season)")
    p.add_argument('--sweep_values', default='0.1,1,10',
                   help='Comma-separated sam_weight/moment_weight grid values '
                        '(spectral family)')
    p.add_argument('--vgg_style_values', default='10,100,1000',
                   help='Comma-separated style_weight grid values (vgg family)')
    p.add_argument('--vgg_dis_values', default='0.1,1,10',
                   help='Comma-separated dis_weight grid values (vgg family)')
    p.add_argument('--epochs', type=int, default=25)

    # Training passthroughs
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_sigma', type=float, default=0.1)
    p.add_argument('--patch_size', type=int, default=None,
                   help='Random-crop size; the official NAMM resolution is 64')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--fp16', action='store_true', help='Mixed-precision training')

    # Wandb passthroughs (one run per config)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='cs159')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--wandb_prefix', default='ablation-',
                   help='Prefix for per-config wandb run names')

    # Control flow
    p.add_argument('--list', action='store_true',
                   help='Print the config list and exit')
    p.add_argument('--dry_run', action='store_true',
                   help='Print the resolved commands without training')
    p.add_argument('--config_index', type=int, default=None,
                   help='Run only the i-th config (for SLURM array jobs)')
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated config names to run (subset of the sweep)')
    p.add_argument('--collate_only', action='store_true',
                   help='Skip training; rebuild ablation_summary.csv from runs')
    return p.parse_args()


# ── Config construction ───────────────────────────────────────────────────────

def parse_sweep_values(spec: str) -> list[float]:
    values = [float(v) for v in spec.split(',') if v.strip()]
    if not values:
        raise ValueError(f'--sweep_values parsed empty from {spec!r}')
    return values


def build_configs(sweep_values: list[float],
                  vgg_style_values: list[float],
                  vgg_dis_values: list[float]) -> list[dict]:
    """Build the ablation config list. Order is stable for fixed sweep values."""
    configs = []
    # vgg family: sweep style_weight x dis_weight.
    for stw in vgg_style_values:
        for dsw in vgg_dis_values:
            configs.append({
                'name': f'vgg_st{stw:g}_ds{dsw:g}',
                'family': 'vgg',
                'dis_weight': dsw, 'style_weight': stw,
                'sam_weight': 0.0, 'moment_weight': 0.0,
            })
    # spectral family: sweep sam_weight x moment_weight (VGG terms off).
    for sw in sweep_values:
        for mw in sweep_values:
            configs.append({
                'name': f'spectral_sw{sw:g}_mw{mw:g}',
                'family': 'spectral',
                'dis_weight': 0.0, 'style_weight': 0.0,
                'sam_weight': sw, 'moment_weight': mw,
            })
    return configs


def build_command(cfg: dict, args) -> list[str]:
    """Resolved `python train_mirror_map.py ...` command for one config."""
    out_dir = os.path.join(args.output_root, cfg['name'])
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--data_root', args.data_root,
        '--output_dir', out_dir,
        '--roi', args.roi,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--num_workers', str(args.num_workers),
        '--max_sigma', str(args.max_sigma),
        '--seed', str(args.seed),
        '--log_every', str(args.log_every),
        '--dis_weight', f"{cfg['dis_weight']:g}",
        '--style_weight', f"{cfg['style_weight']:g}",
        '--sam_weight', f"{cfg['sam_weight']:g}",
        '--moment_weight', f"{cfg['moment_weight']:g}",
    ]
    if args.patch_size is not None:
        cmd += ['--patch_size', str(args.patch_size)]
    if args.fp16:
        cmd.append('--fp16')
    if args.wandb:
        cmd += ['--wandb',
                '--wandb_project', args.wandb_project,
                '--wandb_run_name', f"{args.wandb_prefix}{cfg['name']}"]
        if args.wandb_entity:
            cmd += ['--wandb_entity', args.wandb_entity]
    return cmd


# ── Running ───────────────────────────────────────────────────────────────────

def run_config(cfg: dict, args) -> int:
    """Train one config; tee combined output to <config>/train.log. Returns rc."""
    out_dir = os.path.join(args.output_root, cfg['name'])
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_command(cfg, args)
    log_path = os.path.join(out_dir, 'train.log')

    print('\n' + '=' * 78)
    print(f"[{cfg['name']}] {' '.join(cmd)}")
    print('=' * 78, flush=True)

    with open(log_path, 'w') as logf:
        logf.write(' '.join(cmd) + '\n\n')
        logf.flush()
        # cwd=THIS_DIR so train_mirror_map.py's relative imports resolve.
        proc = subprocess.Popen(
            cmd, cwd=THIS_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()

    print(f"[{cfg['name']}] {'ok' if rc == 0 else f'FAILED (exit {rc})'}",
          flush=True)
    return rc


# ── Collation ─────────────────────────────────────────────────────────────────

def best_val_row(history_path: str) -> dict | None:
    """Return the val row with the lowest loss (matches best.pt selection)."""
    if not os.path.isfile(history_path):
        return None
    best = None
    with open(history_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('phase') != 'val':
                continue
            try:
                loss = float(row['loss'])
            except (KeyError, ValueError, TypeError):
                continue
            if best is None or loss < float(best['loss']):
                best = row
    return best


def collate(configs: list[dict], output_root: str) -> None:
    """Write ablation_summary.csv and print a ranked table."""
    rows = []
    for cfg in configs:
        row = {
            'config': cfg['name'], 'family': cfg['family'],
            'dis_weight': f"{cfg['dis_weight']:g}",
            'style_weight': f"{cfg['style_weight']:g}",
            'sam_weight': f"{cfg['sam_weight']:g}",
            'moment_weight': f"{cfg['moment_weight']:g}",
        }
        best = best_val_row(os.path.join(output_root, cfg['name'], 'history.csv'))
        if best is None:
            row.update({k: '' for k in ('best_epoch', 'val_loss', 'val_mae',
                                        'val_sam', 'val_psnr', 'val_ssim')})
            row['status'] = 'missing'
        else:
            row['best_epoch'] = best.get('epoch', '')
            row['val_loss'] = best['loss']
            row['val_mae'] = best.get('mae', '')
            row['val_sam'] = best.get('sam', '')
            row['val_psnr'] = best.get('psnr', '')
            row['val_ssim'] = best.get('ssim', '')
            row['status'] = 'ok'
        rows.append(row)

    rows.sort(key=lambda r: (0, float(r['val_loss'])) if r['status'] == 'ok'
              else (1, 0.0))

    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, 'ablation_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {summary_path}')

    print('\nAblation summary (sorted by val_loss):')
    header = (f"{'rank':>4}  {'config':<22} {'family':<9} "
              f"{'sam':>5} {'mom':>5} {'val_loss':>10} "
              f"{'val_psnr':>9} {'val_sam':>9} {'val_ssim':>9}  status")
    print(header)
    print('-' * len(header))
    for i, r in enumerate(rows, 1):
        def fmt(key: str) -> str:
            v = r[key]
            try:
                return f'{float(v):.4f}'
            except (ValueError, TypeError):
                return '-'
        print(f"{i:>4}  {r['config']:<22} {r['family']:<9} "
              f"{r['sam_weight']:>5} {r['moment_weight']:>5} "
              f"{fmt('val_loss'):>10} {fmt('val_psnr'):>9} "
              f"{fmt('val_sam'):>9} {fmt('val_ssim'):>9}  {r['status']}")


def print_config_table(configs: list[dict]) -> None:
    header = (f"{'idx':>3}  {'config':<22} {'family':<9} "
              f"{'dis':>5} {'style':>6} {'sam':>5} {'moment':>6}")
    print(header)
    print('-' * len(header))
    for i, c in enumerate(configs):
        print(f"{i:>3}  {c['name']:<22} {c['family']:<9} "
              f"{c['dis_weight']:>5g} {c['style_weight']:>6g} "
              f"{c['sam_weight']:>5g} {c['moment_weight']:>6g}")
    print(f"\n{len(configs)} configs total.")


# ── Main ──────────────────────────────────────────────────────────────────────

def select_configs(configs: list[dict], args) -> list[dict]:
    """Resolve --config_index / --only down to the configs to actually run."""
    if args.config_index is not None and args.only is not None:
        sys.exit('error: pass at most one of --config_index / --only')
    if args.config_index is not None:
        if not 0 <= args.config_index < len(configs):
            sys.exit(f'error: --config_index {args.config_index} out of range '
                     f'[0, {len(configs)})')
        return [configs[args.config_index]]
    if args.only is not None:
        wanted = [n.strip() for n in args.only.split(',') if n.strip()]
        by_name = {c['name']: c for c in configs}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            sys.exit(f'error: unknown config name(s): {", ".join(unknown)}\n'
                     f'valid names: {", ".join(by_name)}')
        return [by_name[n] for n in wanted]
    return configs


def main() -> None:
    args = get_args()
    configs = build_configs(
        parse_sweep_values(args.sweep_values),
        parse_sweep_values(args.vgg_style_values),
        parse_sweep_values(args.vgg_dis_values),
    )
    args.output_root = os.path.abspath(args.output_root)

    if args.list:
        print_config_table(configs)
        return

    if args.collate_only:
        collate(configs, args.output_root)
        return

    if args.data_root is None:
        sys.exit('error: --data_root is required (unless --list/--collate_only)')
    args.data_root = os.path.abspath(args.data_root)

    selected = select_configs(configs, args)

    if args.dry_run:
        for cfg in selected:
            print(f"[{cfg['name']}] {' '.join(build_command(cfg, args))}")
        return

    failures = []
    for cfg in selected:
        if run_config(cfg, args) != 0:
            failures.append(cfg['name'])

    # Single-config (array-task) mode: don't collate — that would race with
    # sibling tasks on ablation_summary.csv. Run --collate_only after the array.
    if args.config_index is None:
        collate(configs, args.output_root)

    if failures:
        print(f'\n{len(failures)} config(s) failed: {", ".join(failures)}',
              file=sys.stderr)
        sys.exit(1)
    print('\nLoss-ablation sweep complete.')


if __name__ == '__main__':
    main()
