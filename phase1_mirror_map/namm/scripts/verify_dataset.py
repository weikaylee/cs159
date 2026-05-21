"""Sanity-check script for the SEN12MS-CR dataset loader.

Instantiates SEN12MSCRDataset, iterates one full epoch, and prints:
  - triplet counts per split
  - tensor shapes and dtypes
  - per-band min / max / mean for the first batch
  - a report of any missing or mismatched files detected during indexing

Usage (from the namm/ directory):
    python scripts/verify_dataset.py --data_root /data/

Optional flags:
    --seasons spring summer          limit to specific seasons
    --num_channels 4                 number of S2 bands
    --patch_size 64                  random-crop size
    --batch_size 4                   batch size for the sanity pass
    --num_workers 2
"""

import argparse
import sys
import os
import time

# Allow importing from the parent namm/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from datasets.sen12mscr import SEN12MSCRDataset, get_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_stats(tensor_np: np.ndarray, name: str) -> None:
    """Print per-band min / max / mean for a (B, C, H, W) numpy array."""
    print(f"\n  {name}  shape={tensor_np.shape}  dtype={tensor_np.dtype}")
    B, C, H, W = tensor_np.shape
    for c in range(C):
        ch = tensor_np[:, c]
        print(
            f"    band {c:2d}:  min={ch.min():.4f}  "
            f"max={ch.max():.4f}  mean={ch.mean():.4f}"
        )


def _check_value_range(tensor_np: np.ndarray, name: str) -> bool:
    ok = True
    if tensor_np.min() < -1e-3 or tensor_np.max() > 1.0 + 1e-3:
        print(
            f"  WARNING: {name} values outside [0, 1]  "
            f"(min={tensor_np.min():.4f}  max={tensor_np.max():.4f})"
        )
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the SEN12MS-CR dataset loader."
    )
    parser.add_argument(
        "--data_root", default="/data/",
        help="Root directory of the SEN12MS-CR dataset."
    )
    parser.add_argument(
        "--seasons", nargs="*", default=None,
        help="Seasons to include (e.g. spring summer). Default = all found."
    )
    parser.add_argument(
        "--num_channels", type=int, default=13,
        help="Number of S2 bands to load (leading channels)."
    )
    parser.add_argument(
        "--patch_size", type=int, default=64,
        help="Random-crop patch size. Use 256 for full patches."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for the sanity pass."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker count."
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"SEN12MSCR dataset verification")
    print(f"  data_root   : {args.data_root}")
    print(f"  seasons     : {args.seasons or '(all found)'}")
    print(f"  num_channels: {args.num_channels}")
    print(f"  patch_size  : {args.patch_size}")
    print("=" * 60)

    # ── Full dataset (no split) ─────────────────────────────────────────
    print("\n[1/4] Indexing all triplets ...")
    try:
        full_ds = SEN12MSCRDataset(
            args.data_root,
            seasons=args.seasons,
            num_channels=args.num_channels,
            patch_size=None,   # no crop for raw count
        )
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print(f"  Total matched triplets: {len(full_ds)}")
    if len(full_ds) == 0:
        print("  No triplets found — check data_root and directory layout.")
        sys.exit(1)

    # ── Split counts ────────────────────────────────────────────────────
    print("\n[2/4] Checking split sizes ...")
    for split in ("train", "val", "test"):
        loader = get_dataloader(
            args.data_root,
            split=split,
            batch_size=args.batch_size,
            num_workers=0,    # avoid spawn for quick check
            shuffle=False,
            seasons=args.seasons,
            num_channels=args.num_channels,
            patch_size=args.patch_size,
        )
        n = len(loader.dataset)
        print(f"  {split:5s}: {n} samples  ({n // args.batch_size} full batches)")

    # ── First-batch shape / dtype / value range ─────────────────────────
    print("\n[3/4] Inspecting first batch ...")
    train_loader = get_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seasons=args.seasons,
        num_channels=args.num_channels,
        patch_size=args.patch_size,
    )
    batch = next(iter(train_loader))

    all_ok = True
    for key in ("cloudy", "clean", "sar"):
        arr = batch[key].numpy()
        _band_stats(arr, key)
        ok = _check_value_range(arr, key)
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All values in [0, 1] — OK")
    else:
        print("\n  Value range issues detected — review normalisation constants.")

    # ── Full-epoch iteration timing ─────────────────────────────────────
    print("\n[4/4] Iterating one full training epoch ...")
    t0 = time.perf_counter()
    n_batches = 0
    for _ in train_loader:
        n_batches += 1
    elapsed = time.perf_counter() - t0

    print(
        f"  {n_batches} batches in {elapsed:.1f}s  "
        f"({elapsed / max(n_batches, 1) * 1000:.1f} ms/batch)"
    )

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
