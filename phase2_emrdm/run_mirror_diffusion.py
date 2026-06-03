#!/usr/bin/env python3
"""
Inference for the mirror-space EDM (Phase 2).

g_phi is NOT used at inference — the EDM denoiser is conditioned directly on
SAR + cloudy S2 (s1s2), so cloudy images never need to pass through the mirror
map.  Only f_psi (Phase 1 inverse map) is needed to project the denoised
mirror-space sample back to reflectance space.

Pipeline per batch
------------------
1. Load (s2_clean, s1s2) triplets; s1s2 = SAR (2ch) || cloudy S2 (13ch).
2. Start from x_T ~ N(0, sigma_max^2 * I) in mirror space.
3. Run Heun (or Euler) EDM sampler conditioned on s1s2.
4. Apply frozen f_psi to recover reflectance-space S2; clip to [0, 1].
5. Save as 13-band float32 GeoTIFF; compute MAE/SAM/PSNR/SSIM vs GT.

Usage
-----
    cd phase2_emrdm
    python run_mirror_diffusion.py \\
        --data_root /resnick/groups/perona/oywang/cs159/data \\
        --namm_ckpt /resnick/.../runs/stage3_top/spectral_sw0.1_mw1/best.pt \\
        --edm_ckpt  /resnick/.../runs/mirror_edm/best.pt \\
        --output_dir /resnick/.../output/mirror_diffusion \\
        --split test --sampler heun --steps 40 --fp16
"""

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase1_mirror_map"))

from inverse_map import InverseMap          # type: ignore
from losses import reconstruction_metrics   # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mirror_diffusion import (        # type: ignore
    EDMDenoiser, EDMUNet, SEN12MSCRTripletDataset,
)
try:
    from eval_emrdm import build_triplets   # type: ignore
except ImportError:
    build_triplets = None


# ─── Sigma schedule ──────────────────────────────────────────────────────────

def make_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
    rho: float = 7.0,
) -> torch.Tensor:
    """Karras et al. (2022) eq. 5 monotone schedule with 0 appended."""
    t = torch.linspace(0.0, 1.0, n_steps)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + t * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0 sentinel


# ─── Samplers ────────────────────────────────────────────────────────────────

@torch.no_grad()
def heun_sample(
    denoiser: EDMDenoiser,
    shape: tuple,
    s1s2: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
    fp16: bool = False,
) -> torch.Tensor:
    """2nd-order deterministic Heun sampler (Karras et al. 2022, Algorithm 1)."""
    B = shape[0]
    x = torch.randn(shape, device=device, dtype=torch.float32) * sigmas[0]

    for i in range(len(sigmas) - 1):
        s_i    = sigmas[i].item()
        s_next = sigmas[i + 1].item()
        sig_b  = torch.full((B,), s_i, device=device)

        with torch.cuda.amp.autocast(enabled=fp16):
            D_i = denoiser(x, sig_b, s1s2).float()
        d_i    = (x - D_i) / s_i
        x_next = x + (s_next - s_i) * d_i

        if s_next > 0.0:
            sig_b_next = torch.full((B,), s_next, device=device)
            with torch.cuda.amp.autocast(enabled=fp16):
                D_next = denoiser(x_next, sig_b_next, s1s2).float()
            d_next = (x_next - D_next) / s_next
            x_next = x + (s_next - s_i) * (d_i + d_next) * 0.5

        x = x_next

    return x


@torch.no_grad()
def euler_sample(
    denoiser: EDMDenoiser,
    shape: tuple,
    s1s2: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
    fp16: bool = False,
) -> torch.Tensor:
    """1st-order deterministic Euler sampler."""
    B = shape[0]
    x = torch.randn(shape, device=device, dtype=torch.float32) * sigmas[0]

    for i in range(len(sigmas) - 1):
        s_i   = sigmas[i].item()
        sig_b = torch.full((B,), s_i, device=device)

        with torch.cuda.amp.autocast(enabled=fp16):
            D_i = denoiser(x, sig_b, s1s2).float()
        d_i = (x - D_i) / s_i
        x   = x + (sigmas[i + 1].item() - s_i) * d_i

    return x


# ─── Model loaders ───────────────────────────────────────────────────────────

def load_f_psi(args, device: torch.device) -> InverseMap:
    f_psi = InverseMap(
        n_channels=args.n_channels,
        ngf=args.ngf,
        n_res_blocks=args.n_res_blocks,
        residual=True,
    ).to(device)
    ckpt = torch.load(args.namm_ckpt, map_location="cpu")
    if "f_psi" not in ckpt:
        raise KeyError(
            f"Checkpoint {args.namm_ckpt!r} has no 'f_psi' key. "
            "Expected a checkpoint saved by train_mirror_map.py."
        )
    f_psi.load_state_dict(ckpt["f_psi"])
    f_psi.eval()
    for p in f_psi.parameters():
        p.requires_grad_(False)
    print(f"  f_psi loaded from {args.namm_ckpt}")
    return f_psi


def load_denoiser(args, device: torch.device) -> EDMDenoiser:
    network  = EDMUNet(
        in_ch   = args.n_channels + 15,
        out_ch  = args.n_channels,
        base_ch = args.base_ch,
        depth   = args.depth,
        emb_dim = args.emb_dim,
    ).to(device)
    denoiser = EDMDenoiser(network, sigma_data=args.sigma_data).to(device)
    ckpt = torch.load(args.edm_ckpt, map_location="cpu")
    if "denoiser" not in ckpt:
        raise KeyError(
            f"Checkpoint {args.edm_ckpt!r} has no 'denoiser' key. "
            "Expected a checkpoint saved by train_mirror_diffusion.py."
        )
    denoiser.load_state_dict(ckpt["denoiser"])

    # Prefer EMA weights for inference — they are smoother and give better quality.
    if "ema" in ckpt:
        from train_mirror_diffusion import EMA  # type: ignore
        ema = EMA(denoiser, decay=0.999)
        ema.load_state_dict(ckpt["ema"])
        ema.apply(denoiser)  # swap in EMA weights permanently for inference
        print(f"  EDMDenoiser loaded with EMA weights from {args.edm_ckpt}")
    else:
        print(f"  EDMDenoiser loaded (no EMA key — using raw weights) from {args.edm_ckpt}")

    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)
    return denoiser


# ─── Data loading ────────────────────────────────────────────────────────────

def load_samples(data_root: str, split: str) -> list:
    """Load sample list for split from pkl, or scan data_root as fallback."""
    pkl = os.path.join(data_root, f"all_{split}_paths.pkl")
    if os.path.isfile(pkl):
        with open(pkl, "rb") as f:
            samples = pickle.load(f)
        print(f"  {split}: {len(samples)} triplets  ({pkl})")
        return samples

    if build_triplets is None:
        raise FileNotFoundError(
            f"all_{split}_paths.pkl not found under {data_root}. "
            "Run eval_emrdm.py to generate pkl files, or ensure "
            "eval_emrdm.py is importable for the fallback scan."
        )
    print(f"  {split}_paths.pkl not found — scanning data_root ...")
    all_triplets = build_triplets(data_root)
    rng   = np.random.RandomState(42)
    idx   = rng.permutation(len(all_triplets)).tolist()
    n_test = max(1, int(len(idx) * 0.15))
    n_val  = max(1, int(len(idx) * 0.10))
    splits_map = {
        "train": idx[: len(idx) - n_val - n_test],
        "val":   idx[len(idx) - n_val - n_test : len(idx) - n_test],
        "test":  idx[len(idx) - n_test :],
    }
    samples = [all_triplets[i] for i in splits_map.get(split, splits_map["test"])]
    print(f"  {split}: {len(samples)} triplets (from scan)")
    return samples


# ─── TIF writing ─────────────────────────────────────────────────────────────

def write_tif(out_path: str, data: np.ndarray, ref_path: str = None) -> None:
    """Write (C, H, W) float32 array as GeoTIFF, copying profile from ref_path."""
    C, H, W = data.shape
    if ref_path is not None and os.path.isfile(ref_path):
        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()
        profile.update(count=C, dtype="float32")
    else:
        profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": W, "height": H, "count": C,
        }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


# ─── Inference loop ──────────────────────────────────────────────────────────

METRICS_COLS = ["sample", "mae", "sam", "psnr", "ssim"]


def run_inference(
    denoiser, f_psi, samples, loader,
    sigmas, sampler_fn, device, output_dir, data_root, fp16, save_mirror,
):
    os.makedirs(output_dir, exist_ok=True)
    if save_mirror:
        os.makedirs(os.path.join(output_dir, "mirror"), exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=METRICS_COLS).writeheader()

    running = {k: [] for k in ("mae", "sam", "psnr", "ssim")}
    global_idx = 0
    t0 = time.time()

    for batch_idx, (s2_clean, s1s2) in enumerate(loader):
        s2_clean = s2_clean.to(device)
        s1s2     = s1s2.to(device)
        B, C, H, W = s2_clean.shape

        mirror_pred = sampler_fn(denoiser, (B, C, H, W), s1s2, sigmas, device, fp16)

        with torch.no_grad():
            s2_pred = f_psi(mirror_pred).clamp(0.0, 1.0)

        batch_samples = samples[global_idx : global_idx + B]

        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_COLS)
            for j in range(B):
                m = reconstruction_metrics(s2_pred[j : j + 1], s2_clean[j : j + 1])
                gt_rel = batch_samples[j]["S2"]
                name   = os.path.splitext(os.path.basename(gt_rel))[0]

                write_tif(
                    os.path.join(output_dir, f"{name}.tif"),
                    s2_pred[j].cpu().numpy(),
                    ref_path=os.path.join(data_root, gt_rel),
                )
                if save_mirror:
                    write_tif(
                        os.path.join(output_dir, "mirror", f"{name}_mirror.tif"),
                        mirror_pred[j].cpu().numpy(),
                    )

                row = {k: m[k].item() for k in ("mae", "sam", "psnr", "ssim")}
                writer.writerow({"sample": name, **row})
                for k in running:
                    running[k].append(row[k])

        global_idx += B
        if (batch_idx + 1) % 10 == 0:
            print(
                f"  [{global_idx}/{len(samples)}]"
                f"  mae={np.mean(running['mae']):.4f}"
                f"  psnr={np.mean(running['psnr']):.2f}"
                f"  ssim={np.mean(running['ssim']):.3f}"
                f"  t={time.time()-t0:.0f}s"
            )

    summary = {k: float(np.mean(v)) for k, v in running.items()}
    print(
        f"\nSummary ({global_idx} samples)"
        f"  mae={summary['mae']:.4f}"
        f"  sam={summary['sam']:.4f}"
        f"  psnr={summary['psnr']:.2f}"
        f"  ssim={summary['ssim']:.3f}"
        f"  elapsed={time.time()-t0:.0f}s"
    )
    with open(metrics_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=METRICS_COLS).writerow(
            {"sample": "MEAN", **summary}
        )
    print(f"  Metrics → {metrics_path}")
    return summary


# ─── Argument parser ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mirror-space EDM inference for cloud removal (Phase 2)."
    )
    # Paths
    p.add_argument("--data_root",  required=True)
    p.add_argument("--namm_ckpt",  required=True,
                   help="Phase 1 best.pt (must contain both g_phi and f_psi keys)")
    p.add_argument("--edm_ckpt",   required=True,
                   help="Phase 2 checkpoint saved by train_mirror_diffusion.py")
    p.add_argument("--output_dir", required=True)
    # Phase 1 InverseMap arch — must match train_mirror_map.py args exactly
    p.add_argument("--n_channels",  type=int, default=13)
    p.add_argument("--ngf",         type=int, default=64)
    p.add_argument("--n_res_blocks",type=int, default=6)
    # Phase 2 denoiser arch — must match train_mirror_diffusion.py args exactly
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--depth",      type=int,   default=4)
    p.add_argument("--emb_dim",    type=int,   default=256)
    p.add_argument("--sigma_data", type=float, default=0.1)
    # EDM sampler
    p.add_argument("--sampler",   choices=["heun", "euler"], default="heun",
                   help="Heun (2nd-order) is recommended; Euler is faster")
    p.add_argument("--steps",     type=int,   default=40,
                   help="Number of denoising steps")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=5.0,
                   help="Must stay within the training sigma distribution: "
                        "exp(P_mean + 2*P_std) = exp(-1.2 + 2.4) ≈ 5. "
                        "The original EMRDM default of 80 is calibrated for "
                        "data std≈0.5; mirror data has std=0.033 so sigma=80 "
                        "is ~2400x the data std and produces pure noise.")
    p.add_argument("--rho",       type=float, default=7.0,
                   help="Schedule curvature (Karras eq. 5)")
    # Data
    p.add_argument("--split",       default="test",
                   help="pkl split to evaluate: train / val / test")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit inference to this many samples (useful for quick checks)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    # Misc
    p.add_argument("--fp16",        action="store_true",
                   help="AMP for the denoiser forward passes")
    p.add_argument("--save_mirror", action="store_true",
                   help="Also write raw mirror-space predictions to output_dir/mirror/")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models ...")
    f_psi    = load_f_psi(args, device)
    denoiser = load_denoiser(args, device)

    sigmas     = make_sigma_schedule(args.sigma_min, args.sigma_max, args.steps, args.rho).to(device)
    sampler_fn = heun_sample if args.sampler == "heun" else euler_sample
    print(
        f"  Sampler: {args.sampler}  steps={args.steps}"
        f"  sigma=[{args.sigma_min}, {args.sigma_max}]  rho={args.rho}"
    )

    print(f"\nLoading {args.split} samples ...")
    samples = load_samples(args.data_root, args.split)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"  Limiting to {len(samples)} samples (--max_samples)")
    dataset = SEN12MSCRTripletDataset(samples, args.data_root)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  {len(samples)} samples  {len(loader)} batches")

    print(f"\nRunning inference → {args.output_dir}")
    run_inference(
        denoiser, f_psi, samples, loader, sigmas, sampler_fn,
        device, args.output_dir, args.data_root, args.fp16, args.save_mirror,
    )


if __name__ == "__main__":
    main()
