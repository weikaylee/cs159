#!/usr/bin/env python3
"""
Train a standard (non-residual) EDM in mirror space, conditioned on S1S2.

Conceptually replaces phase1_mirror_map/namm/train_mdm.py with:
  - PyTorch (vs JAX/Flax)
  - Conditional on S1S2 = SAR + cloudy S2 for cloud removal (vs unconditional)
  - Standard EDM formulation (Karras et al. 2022), not VP/VE SDE score matching
  - g_phi applied on-the-fly per batch — no pre-saved mirror images required
  - Works with Phase 1 checkpoints from train_mirror_map.py / run_loss_ablation.py

Training procedure per step
---------------------------
1. Load clean S2 + SAR + cloudy S2 triplet from disk, normalised to [0, 1].
2. Apply frozen g_phi: y = g_phi(clean_s2)                 mirror-space target
3. Sample sigma ~ log-N(P_mean, P_std^2)
4. Noisy sample: y_t = y + sigma * eps,  eps ~ N(0, I)
5. EDM preconditioning (Karras eq. 7):
      c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
      c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
      c_in    = 1 / sqrt(sigma^2 + sigma_data^2)
      c_noise = 0.25 * log(sigma)
6. Network input: [c_in * y_t  ||  s1s2]  →  F_theta  →  raw_out
   Denoised prediction: D = c_skip * y_t + c_out * raw_out
7. Loss: w(sigma) * mean(||D - y||^2),   w = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

Inference (not implemented here)
---------------------------------
Start from x_T ~ N(0, sigma_max^2 * I) in mirror space.
Run an EDM sampler (e.g. Euler) conditioned on SAR + cloudy S2.
Apply Phase 1 f_psi (inverse map) to recover reflectance-space cloud-free S2.

Usage
-----
    cd phase2_emrdm
    python train_mirror_diffusion.py \\
        --data_root /resnick/groups/perona/oywang/cs159/data \\
        --namm_ckpt /resnick/groups/perona/oywang/cs159/runs/stage3_top/spectral_sw0.1_mw1/best.pt \\
        --output_dir /resnick/groups/perona/oywang/cs159/runs/mirror_edm \\
        --epochs 100 --batch_size 4 --fp16 --wandb
"""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase1_mirror_map"))

from icnn import ICNN, ICNNGradient  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from eval_emrdm import build_triplets  # type: ignore
except ImportError:
    build_triplets = None


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of model weights for inference.

    Keeps a shadow copy of each trainable parameter:
        shadow = decay * shadow + (1 - decay) * param

    Use EMA weights at inference — they are smoother than the raw training
    weights at any given step and typically give better sample quality.
    decay=0.9999 averages over ~10K past steps; use 0.999 for small datasets.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap model weights with EMA weights; return originals for restore."""
        originals = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                originals[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))
        return originals

    @staticmethod
    def restore(model: nn.Module, originals: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(originals[name])

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.float() for k, v in state.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Constants — match SEN12MSCRInterface in sgm/data/sentinel/sentinel.py
# ─────────────────────────────────────────────────────────────────────────────
S2_SCALE = 10_000.0
S1_DB_MIN, S1_DB_MAX = -25.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SEN12MSCRTripletDataset(Dataset):
    """Load SEN12MS-CR triplets for on-the-fly mirror-space training.

    g_phi is applied in the training loop (not here) so DataLoader workers
    only do disk I/O and normalisation.

    Args:
        samples:   list of {"S1": rel, "S2": rel, "S2_cloudy": rel}
        data_root: directory that all relative paths are anchored to

    Returns (per item):
        s2_clean: (13, H, W) float32 in [0, 1]   — input to g_phi
        s1s2:     (15, H, W) float32 in [0, 1]   — SAR (2ch) || cloudy S2 (13ch)
    """

    def __init__(self, samples: list, data_root: str):
        self.samples = samples
        self.root = data_root

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read(rel: str, root: str) -> np.ndarray:
        with rasterio.open(os.path.join(root, rel)) as src:
            return src.read().astype(np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        s2 = self._read(s["S2"], self.root)
        s2c = self._read(s["S2_cloudy"], self.root)
        s1 = self._read(s["S1"], self.root)

        s2  = np.clip(s2  / S2_SCALE, 0.0, 1.0)
        s2c = np.clip(s2c / S2_SCALE, 0.0, 1.0)
        s1  = np.clip((s1 - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN), 0.0, 1.0)

        s1s2 = np.concatenate([s1, s2c], axis=0)  # (15, H, W)
        return torch.from_numpy(s2), torch.from_numpy(s1s2)


def load_splits(data_root: str):
    """Return (train_samples, val_samples) from pkl files or by scanning data_root."""
    splits = {}
    for split in ("train", "val"):
        pkl = os.path.join(data_root, f"all_{split}_paths.pkl")
        if os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                splits[split] = pickle.load(f)
            print(f"  {split}: {len(splits[split])} triplets  ({pkl})")
        else:
            splits[split] = None

    if any(v is None for v in splits.values()):
        if build_triplets is None:
            raise FileNotFoundError(
                "all_train_paths.pkl / all_val_paths.pkl not found under data_root. "
                "Run eval_emrdm.py to generate them, or ensure eval_emrdm.py is importable."
            )
        print("  pkl files not found — scanning data_root with build_triplets ...")
        all_triplets = build_triplets(data_root)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_triplets)).tolist()
        n_val = max(1, int(len(idx) * 0.10))
        splits["train"] = [all_triplets[i] for i in idx[:-n_val]]
        splits["val"]   = [all_triplets[i] for i in idx[-n_val:]]
        print(f"  train: {len(splits['train'])}, val: {len(splits['val'])}")

    return splits["train"], splits["val"]


# ─────────────────────────────────────────────────────────────────────────────
# Mirror-map loader
# ─────────────────────────────────────────────────────────────────────────────
def load_g_phi(args, device: torch.device) -> ICNNGradient:
    """Reconstruct g_phi from Phase 1 checkpoint and freeze its weights."""
    icnn = ICNN(
        n_in_channels=args.n_channels,
        n_layers=args.icnn_layers,
        n_filters=args.icnn_filters,
        strong_convexity=args.strong_convexity,
    ).to(device)
    g_phi = ICNNGradient(icnn).to(device)

    ckpt = torch.load(args.namm_ckpt, map_location="cpu")
    if "g_phi" not in ckpt:
        raise KeyError(
            f"Checkpoint {args.namm_ckpt!r} has no 'g_phi' key. "
            "Expected a checkpoint saved by train_mirror_map.py / run_loss_ablation.py."
        )
    g_phi.load_state_dict(ckpt["g_phi"])
    g_phi.eval()
    for p in g_phi.parameters():
        p.requires_grad_(False)
    print(f"  g_phi loaded and frozen from {args.namm_ckpt}")
    return g_phi


# ─────────────────────────────────────────────────────────────────────────────
# Network: U-Net denoiser
# ─────────────────────────────────────────────────────────────────────────────
def _find_groups(n_ch: int, target: int = 8) -> int:
    """Largest divisor of n_ch that is <= target."""
    g = min(target, n_ch)
    while n_ch % g != 0 and g > 1:
        g -= 1
    return g


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=x.device)
            / half
        )
        emb = x.float()[:, None] * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with AdaGN sigma conditioning (scale + shift from embedding)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1  = nn.GroupNorm(_find_groups(in_ch),  in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(_find_groups(out_ch), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.skip   = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.conv2(F.silu(h)) + self.skip(x)


class EDMUNet(nn.Module):
    """U-Net denoiser for standard EDM.

    Input channels  = noisy mirror (out_ch=13) + S1S2 conditioning (15) = 28.
    Output channels = mirror space (13).
    Sigma is injected through a Fourier embedding → MLP → AdaGN at every block.

    Args:
        in_ch:   total input channels (noisy mirror + S1S2). Default 28.
        out_ch:  output channels (mirror space). Default 13.
        base_ch: width of the first encoder level. Doubles each level.
        depth:   number of encoder/decoder down-sampling stages.
        emb_dim: dimension of the sigma embedding MLP output.
    """

    def __init__(
        self,
        in_ch: int = 28,
        out_ch: int = 13,
        base_ch: int = 64,
        depth: int = 4,
        emb_dim: int = 256,
    ):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]

        self.emb_net = nn.Sequential(
            SinusoidalEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.enc_in = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList(
            [ResBlock(chs[i], chs[i], emb_dim) for i in range(depth - 1)]
        )
        self.down_convs = nn.ModuleList(
            [nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1)
             for i in range(depth - 1)]
        )
        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim)

        self.up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
             for i in reversed(range(depth - 1))]
        )
        self.dec_blocks = nn.ModuleList(
            [ResBlock(chs[i] * 2, chs[i], emb_dim)
             for i in reversed(range(depth - 1))]
        )

        self.out_norm = nn.GroupNorm(_find_groups(chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, in_ch, H, W)  [c_in * y_noisy || s1s2]
        c_noise: (B,)               0.25 * log(sigma), the EDM noise embedding
        returns: (B, out_ch, H, W)  raw network output F_theta
        """
        emb = self.emb_net(c_noise)

        h = self.enc_in(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_convs):
            h = enc(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid2(self.mid1(h, emb), emb)

        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ─────────────────────────────────────────────────────────────────────────────
# EDM preconditioning wrapper
# ─────────────────────────────────────────────────────────────────────────────
class EDMDenoiser(nn.Module):
    """Wraps EDMUNet with Karras et al. (2022) c_skip/c_out/c_in preconditioning.

    sigma_data should be set to approximately the standard deviation of the
    mirror-space target distribution. Run check_mirror_range.py first to find
    the actual data range and set sigma_data accordingly (default 0.1 for
    mirror values in roughly [-0.17, 0.10]).
    """

    def __init__(self, network: EDMUNet, sigma_data: float = 0.1):
        super().__init__()
        self.network = network
        self.sigma_data = sigma_data

    def forward(
        self,
        y_noisy: torch.Tensor,  # (B, 13, H, W) noisy mirror target
        sigma:   torch.Tensor,  # (B,) noise level
        s1s2:    torch.Tensor,  # (B, 15, H, W) SAR + cloudy S2 conditioning
    ) -> torch.Tensor:
        """Return the denoised prediction D(y_noisy, sigma | s1s2)."""
        sd = self.sigma_data
        sg = sigma[:, None, None, None]

        c_skip  = sd ** 2 / (sg ** 2 + sd ** 2)
        c_out   = sg * sd / (sg ** 2 + sd ** 2).sqrt()
        c_in    = 1.0   / (sg ** 2 + sd ** 2).sqrt()
        c_noise = 0.25  * sigma.log()               # (B,)

        net_in  = torch.cat([c_in * y_noisy, s1s2], dim=1)  # (B, 28, H, W)
        raw_out = self.network(net_in, c_noise)               # (B, 13, H, W)
        return c_skip * y_noisy + c_out * raw_out


# ─────────────────────────────────────────────────────────────────────────────
# EDM training loss
# ─────────────────────────────────────────────────────────────────────────────
def edm_loss(
    denoiser: EDMDenoiser,
    y:    torch.Tensor,   # (B, 13, H, W) mirror-space clean target
    s1s2: torch.Tensor,   # (B, 15, H, W) conditioning: SAR (ch0-1) + cloudy S2 (ch2-14)
    P_mean: float = -1.2,
    P_std:  float = 1.2,
    s2_dropout: float = 0.0,
) -> torch.Tensor:
    """Standard EDM training loss (Karras et al. 2022, Algorithm 1).

    s2_dropout: per-sample probability of zeroing the 13 cloudy-S2 channels
    (ch 2-14) while keeping SAR (ch 0-1). Forces the model to learn cloud
    removal from SAR alone on those steps, improving heavy-cloud robustness.
    """
    sd = denoiser.sigma_data
    sigma  = (P_mean + P_std * torch.randn(y.shape[0], device=y.device)).exp()
    y_noisy = y + sigma[:, None, None, None] * torch.randn_like(y)

    # Per-sample S2 dropout: zero cloudy-S2 channels, keep SAR channels.
    if s2_dropout > 0.0:
        s1s2 = s1s2.clone()
        drop_mask = torch.rand(y.shape[0], device=y.device) < s2_dropout  # (B,)
        s1s2[drop_mask, 2:, :, :] = 0.0   # zero ch2-14; ch0-1 (SAR) unchanged

    D = denoiser(y_noisy, sigma, s1s2)

    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2   # (B,)
    # TODO: if loss=nan occurs during training, add this weight cap —
    #   weight = weight.clamp(max=1.0 / sd ** 2)
    # With sd=0.033, a tail sigma~0.001 gives weight~160K which overflows.
    # The cap sets the ceiling at 1/sd^2 ≈ 918 (the value at sigma→0 relative
    # to sigma_data), matching EDM's soft-minSNR intent without distorting
    # the normal training range. Also add a skip-bad-step guard in the loop:
    #   if not loss.isfinite(): optimizer.zero_grad(); scaler.update(); continue
    return (weight[:, None, None, None] * (D - y) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Validation metrics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def val_metrics(
    denoiser: EDMDenoiser,
    y:    torch.Tensor,
    s1s2: torch.Tensor,
    sigma_eval: float = 0.05,
) -> dict:
    """Denoising quality at a fixed low sigma (mirror space)."""
    sigma = torch.full((y.shape[0],), sigma_eval, device=y.device)
    y_noisy = y + sigma[:, None, None, None] * torch.randn_like(y)
    pred = denoiser(y_noisy, sigma, s1s2)
    err  = pred - y
    return {
        "mae": err.abs().mean().item(),
        "mse": (err ** 2).mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(path, epoch, denoiser, ema, optimizer, scheduler, scaler, best_val_loss):
    torch.save({
        "epoch":         epoch,
        "denoiser":      denoiser.state_dict(),
        "ema":           ema.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_ckpt(path, denoiser, ema, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    denoiser.load_state_dict(ckpt["denoiser"])
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from epoch {ckpt['epoch']}  ({path})")
    return ckpt["epoch"], ckpt.get("best_val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Standard EDM in mirror space, conditioned on S1S2."
    )
    # Paths
    p.add_argument("--data_root",   required=True)
    p.add_argument("--namm_ckpt",   required=True,
                   help="Phase 1 best.pt from train_mirror_map.py / run_loss_ablation.py")
    p.add_argument("--output_dir",  required=True)
    # ICNN architecture — must match Phase 1 training args exactly
    p.add_argument("--n_channels",       type=int,   default=13)
    p.add_argument("--icnn_layers",      type=int,   default=6)
    p.add_argument("--icnn_filters",     type=int,   default=64)
    p.add_argument("--strong_convexity", type=float, default=0.3)
    # Denoiser network
    p.add_argument("--base_ch",   type=int, default=64,
                   help="U-Net base channel width (doubles each level)")
    p.add_argument("--depth",     type=int, default=4,
                   help="Number of encoder/decoder down-sampling stages")
    p.add_argument("--emb_dim",   type=int, default=256,
                   help="Sigma embedding MLP output dimension")
    # EDM hyperparameters
    p.add_argument("--sigma_data", type=float, default=0.1,
                   help="Set to ~std of mirror targets. Run check_mirror_range.py first.")
    p.add_argument("--P_mean",     type=float, default=-1.2,
                   help="Log-normal sigma distribution mean")
    p.add_argument("--P_std",      type=float, default=1.2,
                   help="Log-normal sigma distribution std")
    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay — prevents overfitting (set 0 to disable)")
    p.add_argument("--s2_dropout",  type=float, default=0.15,
                   help="Per-sample probability of zeroing cloudy-S2 conditioning "
                        "channels (ch2-14) while keeping SAR (ch0-1). Improves "
                        "heavy-cloud robustness by forcing SAR-only inference.")
    p.add_argument("--ema_decay",   type=float, default=0.999,
                   help="EMA decay for inference weights (0.999 for ~1K-step average, "
                        "0.9999 for ~10K-step average; use 0.999 with small datasets)")
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--fp16",        action="store_true")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--seed",        type=int,   default=42)
    # Logging
    p.add_argument("--wandb",           action="store_true")
    p.add_argument("--wandb_project",   default="cs159")
    p.add_argument("--wandb_run_name",  default=None)
    # Resume
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from (last.pt or best.pt)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Logging ──────────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "mirror-edm",
            config=vars(args),
        )

    log_path = os.path.join(args.output_dir, "history.csv")
    csv_fields = ["phase", "epoch", "step", "elapsed_s", "loss", "mae", "mse"]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Load frozen g_phi ────────────────────────────────────────────────────
    print("Loading g_phi ...")
    g_phi = load_g_phi(args, device)

    # ── Dataset & loaders ────────────────────────────────────────────────────
    print("Loading dataset ...")
    train_samples, val_samples = load_splits(args.data_root)
    train_ds = SEN12MSCRTripletDataset(train_samples, args.data_root)
    val_ds   = SEN12MSCRTripletDataset(val_samples,   args.data_root)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    network  = EDMUNet(
        in_ch   = args.n_channels + 15,   # 13 mirror + 15 S1S2
        out_ch  = args.n_channels,
        base_ch = args.base_ch,
        depth   = args.depth,
        emb_dim = args.emb_dim,
    ).to(device)
    denoiser = EDMDenoiser(network, sigma_data=args.sigma_data).to(device)

    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"  EDMDenoiser: {n_params / 1e6:.1f}M parameters")

    ema = EMA(denoiser, decay=args.ema_decay)
    print(f"  EMA decay: {args.ema_decay}")

    # ── Optimiser & scaler ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )
    scaler    = GradScaler(enabled=args.fp16)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_ckpt(
            args.resume, denoiser, ema, optimizer, scheduler, scaler
        )

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        denoiser.train()
        step_losses = []

        for step, (s2_clean, s1s2) in enumerate(train_loader):
            s2_clean = s2_clean.to(device)
            s1s2     = s1s2.to(device)

            # Apply frozen g_phi on-the-fly. ICNNGradient.forward uses
            # torch.enable_grad() internally so no_grad wrapping is safe here
            # (documented in CLAUDE.md). phi.sum() in the gradient computation
            # gives correct per-sample gradients for independent batch samples.
            with torch.no_grad():
                mirror_target = g_phi(s2_clean)   # (B, 13, H, W)

            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                loss = edm_loss(denoiser, mirror_target, s1s2,
                               args.P_mean, args.P_std, args.s2_dropout)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(denoiser)

            step_losses.append(loss.item())
            global_step += 1

            if (step + 1) % args.log_every == 0:
                elapsed   = time.time() - t0
                mean_loss = float(np.mean(step_losses[-args.log_every:]))
                print(
                    f"  [ep {epoch:03d} step {step + 1:05d}]"
                    f"  loss={mean_loss:.4f}  elapsed={elapsed:.0f}s"
                )
                with open(log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(dict(
                        phase="train", epoch=epoch, step=global_step,
                        elapsed_s=f"{elapsed:.1f}", loss=f"{mean_loss:.6f}",
                        mae="", mse="",
                    ))
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": mean_loss, "step": global_step})

        # ── Validation (run under EMA weights) ───────────────────────────────
        denoiser.eval()
        orig_weights = ema.apply(denoiser)
        v_losses, v_maes, v_mses = [], [], []
        with torch.no_grad():
            for s2_clean, s1s2 in val_loader:
                s2_clean = s2_clean.to(device)
                s1s2     = s1s2.to(device)
                mirror_target = g_phi(s2_clean)

                v_losses.append(
                    edm_loss(denoiser, mirror_target, s1s2, args.P_mean, args.P_std).item()
                )
                m = val_metrics(denoiser, mirror_target, s1s2)
                v_maes.append(m["mae"])
                v_mses.append(m["mse"])

        val_loss = float(np.mean(v_losses))
        val_mae  = float(np.mean(v_maes))
        val_mse  = float(np.mean(v_mses))
        elapsed  = time.time() - t0

        print(
            f"Epoch {epoch:03d} | val_loss={val_loss:.4f}"
            f"  mae={val_mae:.4f}  mse={val_mse:.6f}  elapsed={elapsed:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(dict(
                phase="val", epoch=epoch, step="",
                elapsed_s=f"{elapsed:.1f}", loss=f"{val_loss:.6f}",
                mae=f"{val_mae:.6f}", mse=f"{val_mse:.6f}",
            ))
        if args.wandb:
            import wandb
            wandb.log({"val/loss": val_loss, "val/mae": val_mae,
                       "val/mse": val_mse, "epoch": epoch})

        ema.restore(denoiser, orig_weights)
        denoiser.train()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if args.wandb:
            import wandb
            wandb.log({"train/lr": current_lr, "epoch": epoch})

        # ── Checkpoints ──────────────────────────────────────────────────────
        save_ckpt(
            os.path.join(args.output_dir, "last.pt"),
            epoch + 1, denoiser, ema, optimizer, scheduler, scaler, best_val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                os.path.join(args.output_dir, "best.pt"),
                epoch + 1, denoiser, ema, optimizer, scheduler, scaler, best_val_loss,
            )
            print(f"  → new best val_loss={best_val_loss:.4f}, saved best.pt")

    if args.wandb:
        import wandb
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
