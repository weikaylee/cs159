#!/usr/bin/env python3
"""Prepare EMRDM mirror-space training data from a pretrained NAMM g_phi.

This script reads clean Sentinel-2 patches from SEN12MS-CR, applies the
pretrained NAMM forward map g_phi to each clean image, and writes the
mirror-space outputs to disk. It also writes split pickles of triplets that
point the EMRDM target field to the generated mirror images.

The resulting files are compatible with a custom EMRDM config that sets
`paths_file` for the SEN12MSCRInterface.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

# Ensure phase1_mirror_map is importable when running from phase2_emrdm/.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT.parent / "phase1_mirror_map"))

from icnn import ICNN, ICNNGradient  # type: ignore
from eval_emrdm import build_triplets  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mirror-space EMRDM training targets from a pretrained NAMM g_phi."
    )
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing SEN12MS-CR scene folders.")
    parser.add_argument("--namm_ckpt", required=True,
                        help="Path to a trained NAMM checkpoint containing g_phi state.")
    parser.add_argument("--mirror_subdir", default="mirror_s2",
                        help="Subdirectory under data_root where generated mirror targets are written.")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for applying g_phi.")
    parser.add_argument("--n_channels", type=int, default=13,
                        help="Number of Sentinel-2 channels.")
    parser.add_argument("--icnn_layers", type=int, default=6,
                        help="Number of ICNN hidden layers used during NAMM training.")
    parser.add_argument("--icnn_filters", type=int, default=64,
                        help="Number of ICNN filters used during NAMM training.")
    parser.add_argument("--strong_convexity", type=float, default=0.3,
                        help="Strong convexity coefficient used during NAMM training.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, limit the number of clean patches to process (useful for debugging).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test splits.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing mirror outputs and pickle files.")
    return parser.parse_args()


def load_g_phi(args: argparse.Namespace) -> ICNNGradient:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    icnn = ICNN(
        n_in_channels=args.n_channels,
        n_layers=args.icnn_layers,
        n_filters=args.icnn_filters,
        strong_convexity=args.strong_convexity,
    ).to(device)
    g_phi = ICNNGradient(icnn).to(device)

    ckpt = torch.load(args.namm_ckpt, map_location="cpu")
    if "g_phi" not in ckpt:
        raise KeyError(f"Checkpoint {args.namm_ckpt} does not contain 'g_phi' state.")
    g_phi.load_state_dict(ckpt["g_phi"])
    g_phi.eval()
    return g_phi


def mirror_target_path(data_root: str, mirror_subdir: str, clean_rel: str) -> str:
    return os.path.join(mirror_subdir, clean_rel)


def write_mirror_image(
    clean_path: str,
    mirror_path: str,
    g_phi: ICNNGradient,
    device: torch.device,
) -> None:
    os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
    with rasterio.open(clean_path) as src:
        profile = src.profile.copy()
        img = src.read().astype(np.float32)

    img = np.clip(img / 10000.0, 0.0, 1.0)
    x = torch.from_numpy(img).to(device)

    with torch.no_grad():
        y = g_phi(x.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)

    profile.update(dtype="float32", count=y.shape[0], compress="lzw")
    with rasterio.open(mirror_path, "w", **profile) as dst:
        dst.write(y)


def write_split_pickles(
    data_root: str,
    triplets: list[dict],
    mirror_subdir: str,
    seed: int,
    val_fraction: float = 0.10,
    test_fraction: float = 0.15,
) -> None:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(triplets)).tolist()
    n = len(order)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    splits = {
        "train": order[: n - n_val - n_test],
        "val":   order[n - n_val - n_test : n - n_test],
        "test":  order[n - n_test :],
    }

    for split, indices in splits.items():
        out = []
        for i in indices:
            sample = triplets[i]
            out.append({
                "S1": sample["S1"],
                "S2": mirror_target_path(data_root, mirror_subdir, sample["S2"]),
                "S2_cloudy": sample["S2_cloudy"],
            })
        pickle_path = os.path.join(data_root, f"all_{split}_paths_mirror.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Wrote {len(out)} samples → {pickle_path}")


def main() -> None:
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    mirror_root = os.path.join(data_root, args.mirror_subdir)
    os.makedirs(mirror_root, exist_ok=True)

    g_phi = load_g_phi(args)
    device = next(g_phi.parameters()).device

    print(f"Loading triplets from {data_root}")
    triplets = build_triplets(data_root)
    if args.max_samples is not None:
        triplets = triplets[: args.max_samples]
        print(f"Limiting to {len(triplets)} triplets for debug run.")

    if len(triplets) == 0:
        raise RuntimeError(f"No SEN12MS-CR triplets found under {data_root}")

    print(f"Generating mirror targets for {len(triplets)} clean S2 patches")
    start = time.time()
    for idx, sample in enumerate(triplets, 1):
        clean_rel = sample["S2"]
        clean_path = os.path.join(data_root, clean_rel)
        mirror_rel = mirror_target_path(data_root, args.mirror_subdir, clean_rel)
        mirror_path = os.path.join(data_root, mirror_rel)

        if not args.overwrite and os.path.exists(mirror_path):
            print(f"[{idx}/{len(triplets)}] skipping existing mirror: {mirror_rel}")
            continue

        write_mirror_image(clean_path, mirror_path, g_phi, device)
        print(f"[{idx}/{len(triplets)}] wrote mirror target: {mirror_rel}")

    elapsed = time.time() - start
    print(f"Mirror dataset prepared in {elapsed:.1f}s")

    print("Writing mirror split pickle files")
    write_split_pickles(data_root, triplets, args.mirror_subdir, args.seed)
    print("Done. Use the generated `all_{split}_paths_mirror.pkl` files with a mirror-space EMRDM config.")


if __name__ == "__main__":
    main()
