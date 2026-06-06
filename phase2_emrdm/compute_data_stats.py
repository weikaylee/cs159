#!/usr/bin/env python3
"""
Compute mean and std of clean S2 reflectance over the training split.

Reads all_train_paths.pkl from data_root and streams through patches to
compute a Welford online mean/variance (numerically stable, single pass).
Prints per-channel and global statistics — use global_std as --sigma_data.

Usage
-----
    python compute_data_stats.py --data_root /path/to/data
    python compute_data_stats.py --data_root /path/to/data --max_samples 2000
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import rasterio

S2_SCALE = 10_000.0


def welford_update(count, mean, M2, x):
    """Update Welford running stats with a new batch of values (flattened array)."""
    for val in x.ravel():
        count += 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        M2 += delta * delta2
    return count, mean, M2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of patches (default: all train patches)")
    args = p.parse_args()

    pkl = os.path.join(args.data_root, "all_train_paths.pkl")
    if not os.path.isfile(pkl):
        sys.exit(f"ERROR: {pkl} not found — run eval_emrdm.py first.")

    with open(pkl, "rb") as f:
        samples = pickle.load(f)

    if args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), min(args.max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in idx]

    print(f"Computing stats over {len(samples)} patches ...")

    n_channels = 13
    # Per-channel Welford accumulators
    counts = np.zeros(n_channels, dtype=np.float64)
    means  = np.zeros(n_channels, dtype=np.float64)
    M2s    = np.zeros(n_channels, dtype=np.float64)

    for i, s in enumerate(samples):
        path = os.path.join(args.data_root, s["S2"])
        with rasterio.open(path) as src:
            img = src.read().astype(np.float64)   # (13, H, W)
        img = np.clip(img / S2_SCALE, 0.0, 1.0)

        for c in range(n_channels):
            counts[c], means[c], M2s[c] = welford_update(
                counts[c], means[c], M2s[c], img[c]
            )

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(samples)} ...")

    stds = np.sqrt(M2s / counts)

    print("\n── Per-channel statistics ──────────────────────")
    print(f"{'Band':>6}  {'mean':>8}  {'std':>8}")
    band_names = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
    for c in range(n_channels):
        print(f"{band_names[c]:>6}  {means[c]:8.4f}  {stds[c]:8.4f}")

    global_mean = float(np.mean(means))
    # Combine per-channel variances: E[Var] + Var[E] over channels
    global_std  = float(np.sqrt(np.mean(stds**2 + (means - global_mean)**2)))

    print(f"\n── Global (all channels) ───────────────────────")
    print(f"  mean      = {global_mean:.4f}")
    print(f"  std       = {global_std:.4f}")
    print(f"\n  → Use --sigma_data {global_std:.3f} in train_raw_diffusion.py")


if __name__ == "__main__":
    main()
