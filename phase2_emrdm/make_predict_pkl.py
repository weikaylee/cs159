#!/usr/bin/env python3
"""
Build all_predict_paths.pkl from the raw (non-mirror) train/val/test pkl files.

Usage:
    python make_predict_pkl.py [--data_root PATH]

The output pkl uses raw ROIs1158_spring_s{1,2,2_cloudy} paths so that when
run_emrdm_predict.py runs inference, the _target.png ground-truth column
shows raw cloud-free S2 images (not mirror-space).
"""

import argparse
import os
import pickle

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", default=DATA_ROOT,
        help="Directory containing all_{train,val,test}_paths.pkl (default: cs159/data)",
    )
    args = parser.parse_args()

    triplets = []
    for split in ("train", "val", "test"):
        src = os.path.join(args.data_root, f"all_{split}_paths.pkl")
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found — skipping")
            continue
        with open(src, "rb") as f:
            data = pickle.load(f)
        triplets.extend(data)
        print(f"  {split}: {len(data)} triplets loaded from {src}")

    if not triplets:
        raise FileNotFoundError(f"No source pkl files found in {args.data_root}")

    out = os.path.join(args.data_root, "all_predict_paths.pkl")
    with open(out, "wb") as f:
        pickle.dump(triplets, f)
    print(f"\n  {len(triplets)} total triplets -> {out}")
    print("  Sample:", triplets[0])


if __name__ == "__main__":
    main()
