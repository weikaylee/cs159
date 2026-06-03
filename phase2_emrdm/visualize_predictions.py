#!/usr/bin/env python3
"""Visualise cloud-removal predictions from run_mirror_diffusion.py.

For each output TIF in output_dir/, this script finds the matching cloudy
input and ground-truth cloud-free image from the pkl split file, then saves
a side-by-side PNG: [Cloudy | Predicted | Ground Truth].

Usage
-----
    python phase2_emrdm/visualize_predictions.py \\
        --output_dir  /resnick/.../output/mirror_diffusion \\
        --data_root   /resnick/.../data \\
        --split       test \\
        --n_samples   12 \\
        --out_png     predictions_grid.png

Sentinel-2 band indices (0-indexed, standard GEE order)
--------------------------------------------------------
    0=B1  1=B2(Blue)  2=B3(Green)  3=B4(Red)  4=B5 ... 12=B12
    True colour RGB:  R=3, G=2, B=1
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ─── Band config ──────────────────────────────────────────────────────────────
# Sentinel-2 true colour: B4 (Red), B3 (Green), B2 (Blue)
RGB_BANDS = (3, 2, 1)   # 0-indexed


def read_tif(path: str) -> np.ndarray:
    """Return (C, H, W) float32 array."""
    with rasterio.open(path) as src:
        return src.read().astype("float32")


def to_rgb(img: np.ndarray, bands=RGB_BANDS, gamma: float = 1.0,
           percentile: float = 2.0) -> np.ndarray:
    """Convert (C, H, W) float32 in [0,1] to (H, W, 3) uint8 for display.

    Applies per-channel percentile stretch then optional gamma correction.
    """
    rgb = np.stack([img[b] for b in bands], axis=-1)  # (H, W, 3)
    lo  = np.percentile(rgb, percentile)
    hi  = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        rgb = rgb ** (1.0 / gamma)
    return (rgb * 255).astype("uint8")


def load_split_index(data_root: str, split: str) -> dict:
    """Return {stem_name: sample_dict} from the pkl split file."""
    pkl = os.path.join(data_root, f"all_{split}_paths.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(
            f"all_{split}_paths.pkl not found under {data_root}.\n"
            "Run eval_emrdm.py to generate split pkl files."
        )
    with open(pkl, "rb") as f:
        samples = pickle.load(f)
    return {
        os.path.splitext(os.path.basename(s["S2_cloudy"]))[0]: s
        for s in samples
    }


def main():
    p = argparse.ArgumentParser(
        description="Visualise mirror-space EDM predictions vs ground truth."
    )
    p.add_argument("--output_dir",  required=True,
                   help="Directory containing predicted TIFs from run_mirror_diffusion.py")
    p.add_argument("--data_root",   required=True,
                   help="SEN12MS-CR data root (for loading cloudy + GT images)")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--n_samples",   type=int, default=12,
                   help="How many samples to include in the grid")
    p.add_argument("--out_png",     default="predictions_grid.png",
                   help="Output PNG path (saved in output_dir if not absolute)")
    p.add_argument("--gamma",       type=float, default=1.0,
                   help="Gamma correction for display (try 1.5 for dark scenes)")
    p.add_argument("--percentile",  type=float, default=2.0,
                   help="Percentile for per-image contrast stretch (2 = 2nd–98th)")
    p.add_argument("--cols",        type=int, default=3,
                   help="Columns per row (3 = one triplet per row)")
    args = p.parse_args()

    out_png = args.out_png
    if not os.path.isabs(out_png):
        out_png = os.path.join(args.output_dir, out_png)

    # ── Find predicted TIFs ───────────────────────────────────────────────────
    pred_tifs = sorted(
        p for p in Path(args.output_dir).glob("*.tif")
        if "mirror" not in p.stem   # skip mirror/ subdir if --save_mirror used
    )
    if not pred_tifs:
        sys.exit(f"No predicted TIFs found in {args.output_dir}")

    pred_tifs = pred_tifs[: args.n_samples]
    print(f"Found {len(pred_tifs)} predicted TIFs — showing {len(pred_tifs)}")

    # ── Build index into split pkl ────────────────────────────────────────────
    index = load_split_index(args.data_root, args.split)

    # ── Build figure: each row = one triplet (Cloudy | Predicted | GT) ───────
    n = len(pred_tifs)
    n_rows = n  # one row per sample
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4 * n_rows),
        squeeze=False,
    )

    col_labels = ["Cloudy S2 (input)", "Predicted cloud-free", "Ground truth S2"]

    for row, pred_path in enumerate(pred_tifs):
        stem = pred_path.stem   # e.g. ROIs1158_spring_s2_cloudy_0001_p001

        # Try to match stem against the pkl index
        sample = index.get(stem)
        if sample is None:
            # Fallback: match on the numeric suffix (scene_patch part)
            for key, s in index.items():
                if stem in key or key in stem:
                    sample = s
                    break

        pred_img = read_tif(str(pred_path))
        rgb_pred = to_rgb(pred_img, gamma=args.gamma, percentile=args.percentile)

        # Cloudy input
        ax_cloudy = axes[row][0]
        if sample is not None:
            cloudy_path = os.path.join(args.data_root, sample["S2_cloudy"])
            if os.path.isfile(cloudy_path):
                cloudy_img = read_tif(cloudy_path) / 10_000.0
                rgb_cloudy = to_rgb(cloudy_img, gamma=args.gamma, percentile=args.percentile)
                ax_cloudy.imshow(rgb_cloudy)
            else:
                ax_cloudy.text(0.5, 0.5, "file not found", ha="center", va="center",
                               transform=ax_cloudy.transAxes)
        else:
            ax_cloudy.text(0.5, 0.5, "no match in pkl", ha="center", va="center",
                           transform=ax_cloudy.transAxes)
        ax_cloudy.axis("off")

        # Predicted
        ax_pred = axes[row][1]
        ax_pred.imshow(rgb_pred)
        ax_pred.axis("off")

        # Ground truth
        ax_gt = axes[row][2]
        if sample is not None:
            gt_path = os.path.join(args.data_root, sample["S2"])
            if os.path.isfile(gt_path):
                gt_img  = read_tif(gt_path) / 10_000.0
                rgb_gt  = to_rgb(gt_img, gamma=args.gamma, percentile=args.percentile)
                ax_gt.imshow(rgb_gt)
            else:
                ax_gt.text(0.5, 0.5, "file not found", ha="center", va="center",
                           transform=ax_gt.transAxes)
        else:
            ax_gt.text(0.5, 0.5, "no match in pkl", ha="center", va="center",
                       transform=ax_gt.transAxes)
        ax_gt.axis("off")

        # Row label: sample name (trimmed for readability)
        axes[row][0].set_ylabel(
            stem[:40], fontsize=7, rotation=0, labelpad=120, va="center"
        )

    # Column headers on first row
    for col, label in enumerate(col_labels):
        axes[0][col].set_title(label, fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Mirror-space EDM predictions  |  split={args.split}  "
        f"RGB=B{RGB_BANDS[0]+1}/B{RGB_BANDS[1]+1}/B{RGB_BANDS[2]+1}",
        fontsize=12, y=1.002,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_png}")

    # ── Also load and print metrics summary if available ──────────────────────
    metrics_csv = os.path.join(args.output_dir, "metrics.csv")
    if os.path.isfile(metrics_csv):
        import csv
        with open(metrics_csv) as f:
            rows = list(csv.DictReader(f))
        mean_row = next((r for r in rows if r["sample"] == "MEAN"), None)
        if mean_row:
            print(
                f"\nTest-set metrics (from metrics.csv):\n"
                f"  MAE  = {float(mean_row['mae']):.4f}\n"
                f"  SAM  = {float(mean_row['sam']):.4f}\n"
                f"  PSNR = {float(mean_row['psnr']):.2f} dB\n"
                f"  SSIM = {float(mean_row['ssim']):.4f}"
            )


if __name__ == "__main__":
    main()
