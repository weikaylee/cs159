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
        fmin = float(np.nanmin(y))
        fmax = float(np.nanmax(y))
        global_min = min(global_min, fmin)
        global_max = max(global_max, fmax)
        records.append((fmin, fmax, path))

        if i % 2000 == 0:
            print(f"  ... {i}/{len(files)}  running global min={global_min:.4f}  max={global_max:.4f}")

    print(f"\n{'='*60}")
    print(f"Files scanned    : {len(files)}")
    print(f"Files with NaN   : {n_nan}  ← direct cause of training NaN if > 0")
    print(f"Files with Inf   : {n_inf}")
    print(f"Global min       : {global_min:.6f}")
    print(f"Global max       : {global_max:.6f}")
    print(f"Data range       : {global_max - global_min:.6f}")

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
    elif global_min < -1.0 or global_max > 2.0:
        print("CAUTION: Mirror values span a wide range. The EMRDM noise schedule")
        print("(sigma_max=100) was tuned for [0,1] data. With this range the noise")
        print("schedule may be mismatched — consider normalising g_phi outputs.")
    else:
        print(f"No NaN/Inf. Values are in [{global_min:.3f}, {global_max:.3f}].")
        print("The EMRDM can train in this range but the EDM sigma schedule assumes")
        print("data ≈ [0,1]. Consider normalising mirror outputs to [0,1] before")
        print("re-running prepare_mirror_dataset.py.")


if __name__ == "__main__":
    main()
