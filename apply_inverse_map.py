#!/usr/bin/env python3
"""
Apply Phase 1 inverse map (f_psi) to EMRDM prediction TIFs.

Usage:
    python apply_inverse_map.py \
        --input_dir  /resnick/groups/perona/oywang/cs159/output/emrdm_predict \
        --output_dir /resnick/groups/perona/oywang/cs159/output/emrdm_predict_inv \
        --ckpt       /resnick/groups/perona/oywang/cs159/checkpoints/phase1/best.pt
"""

import argparse
import os
import sys

import numpy as np
import rasterio
import torch

# Make phase1_mirror_map importable regardless of cwd
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_mirror_map"))

from inverse_map import InverseMap  # noqa: E402


def verify_counts(input_dir):
    """Confirm every .tif has matching .png, _mu.png, _target.png."""
    tifs = sorted(f for f in os.listdir(input_dir) if f.endswith(".tif"))
    missing = []
    for t in tifs:
        stem = os.path.splitext(t)[0]
        for suffix in (".png", "_mu.png", "_target.png"):
            if not os.path.exists(os.path.join(input_dir, stem + suffix)):
                missing.append(stem + suffix)
    print(f"TIFs: {len(tifs)}")
    print(f"Expected PNGs: {len(tifs) * 3}")
    actual_pngs = sum(
        1 for f in os.listdir(input_dir)
        if f.endswith(".png") and f != "results_comparison.png"
    )
    print(f"Actual PNGs:   {actual_pngs}")
    if missing:
        print(f"WARNING: {len(missing)} missing PNG files (first 5: {missing[:5]})")
    else:
        print("All TIFs have matching PNG triplets.")
    return tifs


def load_f_psi(ckpt_path, device, ngf=64, n_res_blocks=6, n_channels=13):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    f_psi = InverseMap(
        n_channels=n_channels,
        ngf=ngf,
        n_res_blocks=n_res_blocks,
        residual=True,
    )
    # Support both raw state_dict and checkpoint dict with 'f_psi' key
    if "f_psi" in ckpt:
        state = ckpt["f_psi"]
        epoch = ckpt.get("epoch", "?")
    else:
        state = ckpt
        epoch = "?"
    f_psi.load_state_dict(state)
    f_psi.eval().to(device)
    print(f"Loaded f_psi from epoch {epoch}: {ckpt_path}")
    return f_psi


def apply(input_dir, output_dir, ckpt_path, device, ngf, n_res_blocks):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Verifying input counts ===")
    tifs = verify_counts(input_dir)
    if not tifs:
        print("No TIF files found — nothing to do.")
        return

    print("\n=== Loading f_psi ===")
    f_psi = load_f_psi(ckpt_path, device, ngf=ngf, n_res_blocks=n_res_blocks)

    print(f"\n=== Applying f_psi to {len(tifs)} patches ===")
    for i, fname in enumerate(tifs):
        src_path = os.path.join(input_dir, fname)
        with rasterio.open(src_path) as src:
            arr = src.read().astype(np.float32)  # (13, H, W)
            profile = src.profile

        x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 13, H, W)
        with torch.no_grad():
            out = f_psi(x)  # (1, 13, H, W)
        out_arr = out.squeeze(0).cpu().numpy()  # (13, H, W)

        dst_path = os.path.join(output_dir, fname)
        profile.update(dtype=rasterio.float32, count=out_arr.shape[0])
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(out_arr)

        if (i + 1) % 500 == 0 or (i + 1) == len(tifs):
            print(f"  {i + 1}/{len(tifs)}")

    print(f"\nDone. Outputs → {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing EMRDM .tif predictions")
    p.add_argument("--output_dir", required=True,
                   help="Where to write f_psi-processed .tif files")
    p.add_argument("--ckpt",       required=True,
                   help="Phase 1 checkpoint (best.pt or last.pt)")
    p.add_argument("--ngf",          type=int, default=64)
    p.add_argument("--n_res_blocks", type=int, default=6)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    apply(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt,
        device=args.device,
        ngf=args.ngf,
        n_res_blocks=args.n_res_blocks,
    )


if __name__ == "__main__":
    main()
