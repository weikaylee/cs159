#!/usr/bin/env python3
"""
Visualize mirror-space EDM cloud removal results.

Loads predicted TIFs from run_mirror_diffusion.py, finds the matching cloudy
and cloud-free GT images from the split pkl, and saves a side-by-side RGB
comparison grid: [Cloudy | Predicted | Ground Truth].

Sentinel-2 band indices (0-based, 13-band L1C order):
    RGB natural colour  : B4(3), B3(2), B2(1)
    SWIR false colour   : B12(12), B8(7), B4(3)  — highlights cloud/snow/burn

Usage
-----
    python visualize_mirror_diffusion.py \\
        --output_dir /path/to/mirror_diffusion_eval \\
        --data_root  /path/to/data \\
        --n_samples  8 \\
        --select     spread          # best + worst + median
        --save       viz.png
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    import tifffile as _tifffile
    _HAS_RASTERIO = False

# Sentinel-2 L1C 13-band order (0-indexed)
BANDS_RGB  = (3, 2, 1)   # B4, B3, B2 — natural colour
BANDS_SWIR = (12, 7, 3)  # B12, B8, B4 — SWIR false colour


# ─── I/O ─────────────────────────────────────────────────────────────────────

def read_tif(path: str) -> np.ndarray:
    """Return (C, H, W) float32."""
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
    else:
        data = _tifffile.imread(path).astype(np.float32)
        if data.ndim == 2:
            return data[np.newaxis]
        if data.ndim == 3 and data.shape[2] < data.shape[0]:
            return data.transpose(2, 0, 1)
        return data


# ─── Colour mapping ──────────────────────────────────────────────────────────

def to_rgb(
    arr: np.ndarray,
    bands: tuple = BANDS_RGB,
    lo: float = 2.0,
    hi: float = 98.0,
) -> np.ndarray:
    """Convert (C, H, W) float32 → (H, W, 3) uint8 via percentile stretch."""
    rgb = arr[list(bands)].transpose(1, 2, 0).astype(np.float32)  # (H,W,3)
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[:, :, c]
        p_lo = np.percentile(ch, lo)
        p_hi = np.percentile(ch, hi)
        if p_hi > p_lo:
            out[:, :, c] = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# ─── Sample selection ────────────────────────────────────────────────────────

def load_metrics(metrics_csv: str) -> list:
    """Return list of dicts (excluding the MEAN summary row)."""
    rows = []
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["sample"] == "MEAN":
                continue
            rows.append({
                "sample": row["sample"],
                "mae":    float(row["mae"]),
                "sam":    float(row["sam"]),
                "psnr":   float(row["psnr"]),
                "ssim":   float(row["ssim"]),
            })
    return rows


def select_samples(rows: list, n: int, mode: str) -> list:
    """
    mode:
        spread  — best + worst + evenly-spaced median (by PSNR)
        best    — highest PSNR
        worst   — lowest PSNR
        random  — random subset
        all     — every sample (ignores n)
    """
    if mode == "all":
        return rows

    sorted_by_psnr = sorted(rows, key=lambda r: r["psnr"], reverse=True)

    if mode == "best":
        return sorted_by_psnr[:n]
    if mode == "worst":
        return sorted_by_psnr[-n:]
    if mode == "random":
        rng = np.random.default_rng(0)
        idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        return [rows[i] for i in sorted(idx)]

    # spread: cover the full PSNR range
    total = len(sorted_by_psnr)
    if total <= n:
        return sorted_by_psnr
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [sorted_by_psnr[i] for i in indices]


# ─── Build name → paths mapping from pkl ─────────────────────────────────────

def build_name_map(pkl_path: str) -> dict:
    """Return {stem: {"S2": rel, "S2_cloudy": rel}} from the split pkl."""
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    out = {}
    for s in samples:
        name = Path(s["S2"]).stem
        out[name] = s
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize mirror-space EDM cloud removal results."
    )
    p.add_argument("--output_dir", required=True,
                   help="Directory produced by run_mirror_diffusion.py "
                        "(contains predicted TIFs + metrics.csv)")
    p.add_argument("--data_root", required=True,
                   help="Data root for original cloudy + cloud-free TIFs")
    p.add_argument("--pkl", default=None,
                   help="Split pkl file (default: data_root/all_test_paths.pkl)")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Number of samples to show")
    p.add_argument("--select", default="spread",
                   choices=["spread", "best", "worst", "random", "all"],
                   help="How to choose which samples to display")
    p.add_argument("--bands", default="rgb", choices=["rgb", "swir"],
                   help="Colour composite: rgb (natural) or swir (false colour)")
    p.add_argument("--save", default=None,
                   help="Output PNG path (default: output_dir/viz_{bands}.png)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    metrics_csv = os.path.join(args.output_dir, "metrics.csv")
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(
            f"metrics.csv not found in {args.output_dir}. "
            "Run run_mirror_diffusion.py first."
        )

    pkl_path = args.pkl or os.path.join(args.data_root, "all_test_paths.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"pkl not found: {pkl_path}\n"
            "Pass --pkl to point at all_val_paths.pkl or all_train_paths.pkl."
        )

    bands     = BANDS_RGB if args.bands == "rgb" else BANDS_SWIR
    band_label = "Natural RGB (B4/B3/B2)" if args.bands == "rgb" \
                 else "SWIR False Colour (B12/B8/B4)"

    save_path = args.save or os.path.join(
        args.output_dir, f"viz_{args.bands}.png"
    )

    print("Loading metrics ...")
    rows = load_metrics(metrics_csv)
    if not rows:
        raise ValueError("metrics.csv is empty (no sample rows).")

    mean_psnr = np.mean([r["psnr"] for r in rows])
    mean_ssim = np.mean([r["ssim"] for r in rows])
    mean_sam  = np.mean([r["sam"]  for r in rows])
    print(
        f"  {len(rows)} samples | "
        f"mean PSNR={mean_psnr:.2f} dB  SSIM={mean_ssim:.3f}  SAM={mean_sam:.4f} rad"
    )

    selected = select_samples(rows, args.n_samples, args.select)
    print(f"  Showing {len(selected)} samples ({args.select})")

    print("Building name→path map from pkl ...")
    name_map = build_name_map(pkl_path)

    # ── Layout: rows = samples, cols = [Cloudy | Predicted | GT] ─────────────
    n_rows = len(selected)
    n_cols = 3
    fig_w  = n_cols * 3.2
    fig_h  = n_rows * 3.4 + 0.6  # extra for suptitle

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig, hspace=0.05, wspace=0.04,
        left=0.02, right=0.98, top=0.96, bottom=0.01,
    )

    col_titles = ["Cloudy Input", "Predicted (ours)", "Ground Truth"]

    for col, title in enumerate(col_titles):
        fig.text(
            (col + 0.5) / n_cols, 0.975, title,
            ha="center", va="top", fontsize=11, fontweight="bold",
        )

    for row_idx, meta in enumerate(selected):
        name = meta["sample"]

        # ── Locate TIFs ──────────────────────────────────────────────────────
        pred_path = os.path.join(args.output_dir, f"{name}.tif")
        if not os.path.isfile(pred_path):
            print(f"  WARNING: prediction not found: {pred_path} — skipping")
            continue

        sample_paths = name_map.get(name)
        if sample_paths is None:
            print(f"  WARNING: {name} not in pkl — skipping")
            continue

        gt_path     = os.path.join(args.data_root, sample_paths["S2"])
        cloudy_path = os.path.join(args.data_root, sample_paths["S2_cloudy"])

        # ── Load & convert ───────────────────────────────────────────────────
        pred   = read_tif(pred_path)
        gt     = read_tif(gt_path)   / 10_000.0
        cloudy = read_tif(cloudy_path) / 10_000.0

        imgs = [
            to_rgb(cloudy, bands),
            to_rgb(pred,   bands),
            to_rgb(gt,     bands),
        ]

        # ── Plot ─────────────────────────────────────────────────────────────
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(name, fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=4)

            if col == 1:
                ax.set_title(
                    f"PSNR {meta['psnr']:.1f} dB  SSIM {meta['ssim']:.3f}  "
                    f"SAM {meta['sam']:.4f}",
                    fontsize=6.5, pad=2,
                )

    fig.suptitle(
        f"Mirror-space EDM — cloud removal  |  {band_label}\n"
        f"mean: PSNR={mean_psnr:.2f} dB  SSIM={mean_ssim:.3f}  "
        f"SAM={mean_sam:.4f} rad  ({args.select}, N={len(selected)})",
        fontsize=9, y=0.998,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
