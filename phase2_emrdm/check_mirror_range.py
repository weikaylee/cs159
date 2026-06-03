#!/usr/bin/env python3
"""Scan all mirror_s2 TIFs and report value ranges.

Reports:
  - Any file containing NaN or Inf (these cause training NaN directly)
  - Global min/max across the whole dataset (after the /10000 normalisation)
  - The 10 most-extreme patches (lowest min, highest max)

Usage:
    python phase2_emrdm/check_mirror_range.py --data_root /resnick/groups/perona/oywang/cs159/data
"""

import argparse
import glob
import sys

import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--mirror_subdir", default="mirror_s2")
    args = parser.parse_args()

    pattern = f"{args.data_root}/{args.mirror_subdir}/**/*.tif"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"No TIFs found under {pattern}")

    print(f"Scanning {len(files)} mirror TIFs in {args.mirror_subdir}/\n")

    global_min = float("inf")
    global_max = float("-inf")
    n_nan = 0
    n_inf = 0
    records = []  # (fmin, fmax, rel_path)
    # Welford-style accumulators for global mean and std (finite values only)
    total_sum    = 0.0
    total_sq_sum = 0.0
    total_count  = 0

    for i, path in enumerate(files, 1):
        with rasterio.open(path) as src:
            raw = src.read().astype("float32")

        has_nan = bool(np.isnan(raw).any())
        has_inf = bool(np.isinf(raw).any())
        if has_nan:
            n_nan += 1
            rel = path.split(args.mirror_subdir + "/")[-1]
            print(f"  NaN  [{i:5d}] {rel}")
        if has_inf:
            n_inf += 1
            rel = path.split(args.mirror_subdir + "/")[-1]
            print(f"  Inf  [{i:5d}] {rel}")

        y = raw / 10_000.0
        finite = y[np.isfinite(y)]
        total_sum    += float(finite.sum())
        total_sq_sum += float((finite ** 2).sum())
        total_count  += finite.size

        fmin = float(np.nanmin(y))
        fmax = float(np.nanmax(y))
        global_min = min(global_min, fmin)
        global_max = max(global_max, fmax)
        records.append((fmin, fmax, path))

        if i % 2000 == 0:
            print(f"  ... {i}/{len(files)}  running global min={global_min:.4f}  max={global_max:.4f}")

    global_mean = total_sum / max(total_count, 1)
    global_std  = np.sqrt(max(total_sq_sum / max(total_count, 1) - global_mean ** 2, 0.0))

    print(f"\n{'='*60}")
    print(f"Files scanned    : {len(files)}")
    print(f"Files with NaN   : {n_nan}  ← direct cause of training NaN if > 0")
    print(f"Files with Inf   : {n_inf}")
    print(f"Global min       : {global_min:.6f}")
    print(f"Global max       : {global_max:.6f}")
    print(f"Data range       : {global_max - global_min:.6f}")
    print(f"Global mean      : {global_mean:.6f}")
    print(f"Global std       : {global_std:.6f}  ← set --sigma_data to this value")

    print(f"\n10 patches with lowest min (most extreme negative):")
    for fmin, fmax, path in sorted(records, key=lambda r: r[0])[:10]:
        rel = path.split(args.mirror_subdir + "/")[-1]
        print(f"  min={fmin:.4f}  max={fmax:.4f}  {rel}")

    print(f"\n10 patches with highest max:")
    for fmin, fmax, path in sorted(records, key=lambda r: -r[1])[:10]:
        rel = path.split(args.mirror_subdir + "/")[-1]
        print(f"  min={fmin:.4f}  max={fmax:.4f}  {rel}")

    print(f"\n{'='*60}")
    if n_nan > 0 or n_inf > 0:
        print("ACTION NEEDED: NaN/Inf in mirror TIFs will cause training NaN.")
        print("Re-run prepare_mirror_dataset.py (--overwrite) to regenerate these patches.")
    else:
        print(f"No NaN/Inf. Data is in [{global_min:.3f}, {global_max:.3f}]  "
              f"mean={global_mean:.3f}  std={global_std:.3f}")
        print(f"→ Set --sigma_data {global_std:.3f} in train_mirror_diffusion.sh")


if __name__ == "__main__":
    main()
