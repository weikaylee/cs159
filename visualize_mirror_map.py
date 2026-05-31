"""
Visualize Phase 1 mirror maps from a saved checkpoint.

Usage:
    python visualize_mirror_map.py \
        --ckpt /scratch/oywang/cs159/checkpoints/phase1/spectral_sw0.1_mw1/best.pt \
        --data_root data/ROIs1158_spring_s2 \
        --out vis_mirror_map.png

Produces three figures saved to a single output PNG:
  1. Spatial grid: x | g_phi(x) | f_psi(g_phi(x)) | residual g_phi(x)-x
  2. Per-band spectral profiles: mean reflectance per band for all three
  3. Residual histogram: distribution of g_phi(x)-x across all pixels/bands
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).parent / 'phase1_mirror_map'))
from icnn import ICNN, ICNNGradient
from inverse_map import InverseMap


BAND_NAMES = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']


def get_args():
    p = argparse.ArgumentParser(description='Visualize Phase 1 mirror maps')
    p.add_argument('--ckpt',        required=True,
                   help='Path to best.pt / last.pt checkpoint')
    p.add_argument('--data_root',   default='data/ROIs1158_spring_s2',
                   help='Directory tree containing cloud-free .tif patches')
    p.add_argument('--n_patches',   type=int, default=4,
                   help='Number of patches to visualize')
    p.add_argument('--out',         default='vis_mirror_map.png',
                   help='Output image path')
    p.add_argument('--n_channels',  type=int, default=13)
    p.add_argument('--ngf',         type=int, default=64)
    p.add_argument('--n_res_blocks',type=int, default=6)
    p.add_argument('--icnn_filters',type=int, default=64)
    p.add_argument('--icnn_layers', type=int, default=6)
    p.add_argument('--strong_convexity', type=float, default=0.3)
    return p.parse_args()


def load_models(args, device):
    icnn = ICNN(
        n_in_channels=args.n_channels,
        n_filters=args.icnn_filters,
        n_layers=args.icnn_layers,
        strong_convexity=args.strong_convexity,
    )
    g_phi = ICNNGradient(icnn).to(device)
    f_psi = InverseMap(
        n_channels=args.n_channels,
        ngf=args.ngf,
        n_res_blocks=args.n_res_blocks,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    g_phi.load_state_dict(ckpt['g_phi'])
    f_psi.load_state_dict(ckpt['f_psi'])
    g_phi.eval(); f_psi.eval()
    print(f'Loaded checkpoint  epoch={ckpt["epoch"]}  '
          f'best_val_loss={ckpt["best_val_loss"]:.5f}')
    return g_phi, f_psi


def load_patches(data_root, n_patches):
    tifs = sorted(Path(data_root).rglob('*.tif'))[:n_patches]
    if not tifs:
        raise FileNotFoundError(
            f'No .tif files found under {data_root}. '
            'Run `python download_local_data.py` first.')
    print(f'Using {len(tifs)} patches: {[t.name for t in tifs]}')
    patches = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float32) / 10000.0
        patches.append(arr)
    return tifs, np.stack(patches)   # (N, 13, H, W)


def to_rgb(arr13, p=(2, 98)):
    """(13, H, W) float -> (H, W, 3) display RGB using bands B4/B3/B2."""
    rgb = arr13[[3, 2, 1]].copy()
    for c in range(3):
        lo, hi = np.percentile(rgb[c], p)
        rgb[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def plot_spatial_grid(x_np, y_np, xr_np, tif_names, fig_offset, n_figs):
    N = len(tif_names)
    fig, axes = plt.subplots(N, 4, figsize=(16, 4 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        r'$x$ (original)',
        r'$g_\phi(x)$ (mirror space)',
        r'$f_\psi(g_\phi(x))$ (reconstruction)',
        r'$g_\phi(x) - x$ (residual)',
    ]

    for row, (xi, yi, xri, name) in enumerate(zip(x_np, y_np, xr_np, tif_names)):
        d = (yi - xi).mean(axis=0)
        scale = max(abs(d.min()), abs(d.max()), 1e-8)
        d_norm = d / scale

        for col, arr in enumerate([xi, yi, xri]):
            ax = axes[row, col]
            ax.imshow(to_rgb(arr))
            ax.axis('off')
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(name.stem, fontsize=9)

        ax = axes[row, 3]
        im = ax.imshow(d_norm, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[3], fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f'scale={scale:.4f}')

    fig.suptitle('Phase 1 mirror map — spatial visualisation', fontsize=13)
    fig.tight_layout()
    return fig


def plot_spectral_profiles(x_np, y_np, xr_np, tif_names):
    N = len(tif_names)
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 4), sharey=False)
    if N == 1:
        axes = [axes]

    ticks = range(13)
    for ax, xi, yi, xri, name in zip(axes, x_np, y_np, xr_np, tif_names):
        ax.plot(ticks, xi.mean(axis=(1, 2)),  'b-o',  ms=4, label=r'$x$')
        ax.plot(ticks, yi.mean(axis=(1, 2)),  'r--s', ms=4, label=r'$g_\phi(x)$')
        ax.plot(ticks, xri.mean(axis=(1, 2)), 'g:^',  ms=4, label=r'$f_\psi(g_\phi(x))$')
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(BAND_NAMES, rotation=45, fontsize=8)
        ax.set_xlabel('Band')
        ax.set_ylabel('Mean reflectance')
        ax.set_title(name.stem, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-band spectral profiles: original vs mirror vs reconstruction',
                 fontsize=12)
    fig.tight_layout()
    return fig


def plot_residual_histogram(x_np, y_np, xr_np):
    residuals = (y_np - x_np).ravel()
    recon_err = np.abs(xr_np - x_np).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=200, density=True, color='steelblue', alpha=0.8)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel(r'$g_\phi(x) - x$ (reflectance units)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of mirror-map residuals across all patches and bands')
    ax.text(0.98, 0.95,
            f'mean={residuals.mean():.4f}\nstd={residuals.std():.4f}\n'
            f'|max|={np.abs(residuals).max():.4f}\n'
            f'round-trip MAE={recon_err:.5f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    g_phi, f_psi = load_models(args, device)
    tifs, patches = load_patches(args.data_root, args.n_patches)

    x_batch = torch.tensor(patches).to(device)
    with torch.no_grad():
        y_batch  = g_phi(x_batch)
        xr_batch = f_psi(y_batch)

    x_np  = x_batch.detach().cpu().numpy()
    y_np  = y_batch.detach().cpu().numpy()
    xr_np = xr_batch.detach().cpu().numpy()

    fig1 = plot_spatial_grid(x_np, y_np, xr_np, tifs, 0, 3)
    fig2 = plot_spectral_profiles(x_np, y_np, xr_np, tifs)
    fig3 = plot_residual_histogram(x_np, y_np, xr_np)

    # Save all three figures as a multi-page PDF or side-by-side PNG
    out = Path(args.out)
    if out.suffix.lower() == '.pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out) as pdf:
            pdf.savefig(fig1, bbox_inches='tight')
            pdf.savefig(fig2, bbox_inches='tight')
            pdf.savefig(fig3, bbox_inches='tight')
    else:
        # Stack vertically into one PNG
        stem = out.with_suffix('')
        p1 = stem.with_name(stem.name + '_spatial.png')
        p2 = stem.with_name(stem.name + '_spectral.png')
        p3 = stem.with_name(stem.name + '_hist.png')
        fig1.savefig(p1, dpi=150, bbox_inches='tight')
        fig2.savefig(p2, dpi=150, bbox_inches='tight')
        fig3.savefig(p3, dpi=150, bbox_inches='tight')
        print(f'Saved:\n  {p1}\n  {p2}\n  {p3}')
        return

    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
