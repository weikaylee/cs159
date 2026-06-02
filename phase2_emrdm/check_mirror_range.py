#!/usr/bin/env python3
"""Scan all mirror_s2 TIFs and report value ranges.

Prints every file whose per-band max > 1.0 or min < 0.0 after the /10000
normalisation applied by SEN12MSCRInterface, plus an overall summary.

Usage (run from repo root or phase2_emrdm/):
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

    print(f"Scanning {len(files)} mirror TIFs in {args.mirror_subdir}/")

    global_min = float("inf")
    global_max = float("-inf")
    n_bad = 0

    for i, path in enumerate(files, 1):
        with rasterio.open(path) as src:
            y = src.read().astype("float32") / 10_000.0

        fmin, fmax = float(y.min()), float(y.max())
        global_min = min(global_min, fmin)
        global_max = max(global_max, fmax)

        if fmin < 0.0 or fmax > 1.0:
            n_bad += 1
            rel = path.split(args.mirror_subdir + "/")[-1]
            print(f"  [{i:5d}/{len(files)}] OUT OF RANGE  min={fmin:.4f}  max={fmax:.4f}  {rel}")

        if i % 1000 == 0:
            print(f"  ... checked {i}/{len(files)}, bad so far: {n_bad}")

    print(f"\n{'='*60}")
    print(f"Files scanned : {len(files)}")
    print(f"Files out of [0,1]: {n_bad}")
    print(f"Global min : {global_min:.6f}")
    print(f"Global max : {global_max:.6f}")
    if n_bad == 0:
        print("All mirror TIFs are within [0, 1] — clipping is harmless.")
    else:
        print("Some TIFs exceed [0, 1]. SEN12MSCRInterface will clip these,")
        print("which loses information. Consider re-running prepare_mirror_dataset.py")
        print("without rescaling, or normalising g_phi outputs to [0, 1] before saving.")


if __name__ == "__main__":
    main()
