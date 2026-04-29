"""
Validation of PDE-Constrained Diffusion Model on SEN12MS-CR
=============================================================
Dataset triplets:  (S1: 2-ch SAR)  +  (S2_cloudy: 13-ch)  +  (S2_cloudfree: 13-ch)
Patch size:        256 × 256 px
Task:              Cloud removal / field reconstruction — the model is given the
                   S2_cloudy + S1 fields (mapped into the model's 6-channel PDE
                   space) and evaluated against the cloud-free reference.

Directory layout expected (standard HPN-CR / DSen2-CR convention):
    <root>/
      test/
        s1/            ROIs<id>_<season>_<roi>_p<patch>.tif   (2 bands)
        s2_cloudy/     ROIs<id>_<season>_<roi>_p<patch>.tif   (13 bands)
        s2_cloudfree/  ROIs<id>_<season>_<roi>_p<patch>.tif   (13 bands)
      [val/  same layout — used if split='val']

Metrics reported (per patch, then aggregated):
  • PSNR    — Peak Signal-to-Noise Ratio            (dB)
  • SSIM    — Structural Similarity Index
  • MAE     — Mean Absolute Error
  • SAM     — Spectral Angle Mapper                 (degrees)
  • NS-Res  — Navier-Stokes momentum residual norm  (physics constraint)
  • EM-Res  — Maxwell divergence residual norm      (physics constraint)
  • Div     — Divergence penalty  ‖∇·u‖             (incompressibility)
  • CloudCov— Estimated cloud-coverage of input patch (0–1)

Results are written to:
  • val_results.csv   — per-patch metrics
  • val_summary.json  — aggregated mean ± std
  • val_visuals/      — qualitative comparison images (RGB + residual maps)
"""

import os
import csv
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# rasterio for GeoTIFF reading
try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not found — install with:  pip install rasterio")

# optional: matplotlib for saving visualisations
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── import model components from the training module ──────────────────────────
# Adjust the import path if the training file lives elsewhere.
try:
    from pde_diffusion_model import (
        PDEDiffusionModel,
        DiffusionConfig,
        _ddx, _ddy, _laplacian,
        ns_residual_loss,
        maxwell_residual_loss,
        divergence_penalty,
    )
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("[WARN] pde_diffusion_model.py not found on sys.path. "
          "Physics-residual metrics will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants — SEN12MS-CR band semantics
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel-2 band order in the .tif files
S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
S2_RGB   = [3, 2, 1]          # B4 (R), B3 (G), B2 (B) — 0-indexed
S1_BANDS = ["VV", "VH"]

# Approximate per-band normalisation statistics (reflectance ×10000 → [0,1])
S2_MEAN = np.array([1353, 1117,  1041,  947,  1199, 2003, 2374, 2301,
                    2599,  255,    16,  1819, 1349], dtype=np.float32)
S2_STD  = np.array([ 262,  375,   447,  500,   502,  607,  761,  786,
                     825,  102,    20,   984,  820], dtype=np.float32)
S1_MEAN = np.array([-12.5, -20.3], dtype=np.float32)   # dB
S1_STD  = np.array([  5.0,   5.5], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SEN12MSCRDataset(Dataset):
    """
    Loads SEN12MS-CR triplets from the standard folder layout.

    Returns a dict:
        s1          : (2, 256, 256)  float32  normalised SAR
        s2_cloudy   : (13, 256, 256) float32  normalised cloudy optical
        s2_cloudfree: (13, 256, 256) float32  normalised cloud-free optical
        cloud_frac  : scalar float   estimated cloud coverage [0,1]
        filename    : str            stem of the .tif file
    """

    def __init__(self, root: str, split: str = "test",
                 patch_size: int = 256,
                 max_samples: Optional[int] = None,
                 normalise: bool = True):
        self.root       = Path(root) / split
        self.patch_size = patch_size
        self.normalise  = normalise

        s1_dir   = self.root / "s1"
        s2c_dir  = self.root / "s2_cloudy"
        s2f_dir  = self.root / "s2_cloudfree"

        for d in [s1_dir, s2c_dir, s2f_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Expected directory not found: {d}")

        # Build triplet list keyed by filename stem
        stems = sorted(p.stem for p in s1_dir.glob("*.tif"))
        if max_samples is not None:
            stems = stems[:max_samples]

        self.triplets: List[Tuple[Path, Path, Path]] = []
        for stem in stems:
            t = (s1_dir  / f"{stem}.tif",
                 s2c_dir / f"{stem}.tif",
                 s2f_dir / f"{stem}.tif")
            if all(p.exists() for p in t):
                self.triplets.append(t)
            else:
                print(f"[WARN] incomplete triplet skipped: {stem}")

        print(f"SEN12MS-CR [{split}]: {len(self.triplets)} valid triplets found.")

    def __len__(self) -> int:
        return len(self.triplets)

    @staticmethod
    def _read_tif(path: Path) -> np.ndarray:
        """Read a GeoTIFF and return (C, H, W) float32 array."""
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required to read .tif files.")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)   # (C, H, W)
        return arr

    @staticmethod
    def _estimate_cloud_fraction(s2_cloudy: np.ndarray) -> float:
        """
        Simple cloud mask heuristic: pixels with high reflectance in
        B3 (green, idx=2) and B4 (red, idx=3) relative to B8 (NIR, idx=7)
        are flagged as likely cloudy.
        """
        b3  = s2_cloudy[2]
        b4  = s2_cloudy[3]
        b8  = s2_cloudy[7]
        bright = (b3 > 2000) & (b4 > 2000) & (b3 > b8 * 0.8)
        return float(bright.mean())

    def __getitem__(self, idx: int) -> dict:
        s1_path, s2c_path, s2f_path = self.triplets[idx]

        s1   = self._read_tif(s1_path)            # (2,  H, W)  raw dB
        s2c  = self._read_tif(s2c_path)           # (13, H, W)  raw refl*10000
        s2f  = self._read_tif(s2f_path)           # (13, H, W)

        cloud_frac = self._estimate_cloud_fraction(s2c)

        # ── Normalise ──────────────────────────────────────────────────────
        if self.normalise:
            s1  = (s1  - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
            s2c = (s2c - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
            s2f = (s2f - S2_MEAN[:, None, None]) / S2_STD[:, None, None]

        # ── Pad/crop to patch_size if needed ──────────────────────────────
        s1  = self._ensure_size(s1,  self.patch_size)
        s2c = self._ensure_size(s2c, self.patch_size)
        s2f = self._ensure_size(s2f, self.patch_size)

        return {
            "s1":           torch.from_numpy(s1),
            "s2_cloudy":    torch.from_numpy(s2c),
            "s2_cloudfree": torch.from_numpy(s2f),
            "cloud_frac":   torch.tensor(cloud_frac, dtype=torch.float32),
            "filename":     s1_path.stem,
        }

    @staticmethod
    def _ensure_size(arr: np.ndarray, size: int) -> np.ndarray:
        """Centre-crop to (C, size, size)."""
        C, H, W = arr.shape
        if H == size and W == size:
            return arr
        h0 = max(0, (H - size) // 2)
        w0 = max(0, (W - size) // 2)
        arr = arr[:, h0:h0 + size, w0:w0 + size]
        # Pad if still smaller
        ph = max(0, size - arr.shape[1])
        pw = max(0, size - arr.shape[2])
        if ph or pw:
            arr = np.pad(arr, ((0,0),(0,ph),(0,pw)), mode="reflect")
        return arr[:, :size, :size]


# ─────────────────────────────────────────────────────────────────────────────
#  Input adapter: SEN12MS-CR → PDE model 6-channel space
# ─────────────────────────────────────────────────────────────────────────────

def sen12mscr_to_pde_field(s1: torch.Tensor,
                            s2_cloudy: torch.Tensor) -> torch.Tensor:
    """
    Map the multi-sensor input to the model's [u, v, p, Ex, Ey, Bz] layout.

    The analogy:
      u, v   ← SAR VV and VH channels  (Sentinel-1, shape 2×H×W)
               SAR backscatter encodes surface roughness/velocity-like fields.
      p      ← S2 Band 8 (NIR)  — proxy for scene "pressure" (vegetation density)
      Ex, Ey ← S2 Band 3 (Green), Band 4 (Red)  — EM field proxy
      Bz     ← S2 Band 11 (SWIR1)  — magnetic-analogue (thermal emission)

    All bands are already normalised to ~N(0,1) by the dataset loader.
    The mapping is intentionally physically motivated: NIR/SWIR channels vary
    slowly in space (like pressure) while visible bands oscillate faster (EM).
    """
    B = s1.shape[0]
    u  = s1[:, 0:1]          # VV
    v  = s1[:, 1:2]          # VH
    p  = s2_cloudy[:, 7:8]   # B8 NIR
    Ex = s2_cloudy[:, 2:3]   # B3 Green
    Ey = s2_cloudy[:, 3:4]   # B4 Red
    Bz = s2_cloudy[:, 10:11] # B11 SWIR1
    return torch.cat([u, v, p, Ex, Ey, Bz], dim=1)   # (B, 6, H, W)


def pde_field_to_s2_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Extract a rough RGB preview from the 6-channel PDE field.
    Channels 3 (Ex→G) and 4 (Ey→R) and channel 2 (p→B proxy) are used.
    """
    R = x[:, 4]   # Ey → Red
    G = x[:, 3]   # Ex → Green
    B = x[:, 2]   # p  → Blue
    return torch.stack([R, G, B], dim=1)    # (B, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> float:
    """PSNR (dB). Inputs: (C, H, W) or (B, C, H, W)."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(data_range**2 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, data_range: float = 1.0) -> float:
    """
    Simplified SSIM averaged over channels.
    Uses a uniform 11×11 window for speed; a Gaussian window
    can be swapped in for publication-grade results.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Ensure (B, C, H, W)
    if pred.ndim == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, C, H, W = pred.shape
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=pred.device) / (window_size ** 2)

    ssim_vals = []
    for c in range(C):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        mu_p  = F.conv2d(p, kernel, padding=pad)
        mu_t  = F.conv2d(t, kernel, padding=pad)
        sigma_pp = F.conv2d(p * p, kernel, padding=pad) - mu_p ** 2
        sigma_tt = F.conv2d(t * t, kernel, padding=pad) - mu_t ** 2
        sigma_pt = F.conv2d(p * t, kernel, padding=pad) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2))
        ssim_vals.append(ssim_map.mean().item())

    return float(np.mean(ssim_vals))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_sam(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper (degrees).
    Treats the channel dimension as the spectral dimension.
    pred/target: (C, H, W)
    """
    # Flatten spatial dims
    p = pred.flatten(1)    # (C, H*W)
    t = target.flatten(1)  # (C, H*W)
    dot   = (p * t).sum(0)
    norm  = p.norm(dim=0).clamp(min=eps) * t.norm(dim=0).clamp(min=eps)
    angle = torch.acos((dot / norm).clamp(-1.0, 1.0))
    return float(angle.mean().item() * 180.0 / math.pi)


def compute_physics_metrics(x_hat: torch.Tensor,
                            nu: float, eps_em: float, mu: float
                            ) -> Dict[str, float]:
    """
    Evaluate PDE residuals on predicted field (B, 6, H, W).
    Returns dict with ns_res, em_res, div values.
    """
    if not HAS_MODEL:
        return {"ns_res": float("nan"), "em_res": float("nan"),
                "div": float("nan")}
    with torch.no_grad():
        ns  = ns_residual_loss(x_hat, nu).item()
        em  = maxwell_residual_loss(x_hat, eps_em, mu).item()
        div = divergence_penalty(x_hat).item()
    return {"ns_res": ns, "em_res": em, "div": div}


# ─────────────────────────────────────────────────────────────────────────────
#  Inference: run diffusion reverse chain or one-step estimate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_batch(model: "PDEDiffusionModel",
                x_input: torch.Tensor,
                mode: str = "ddim",
                ddim_steps: int = 50) -> torch.Tensor:
    """
    Run the reverse diffusion process on x_input.

    mode = "full"  : full T-step DDPM reverse chain
    mode = "ddim"  : accelerated DDIM sampling (ddim_steps steps)
    mode = "direct": one-step direct prediction (fastest — for benchmarking)
    """
    device = next(model.parameters()).device
    x = x_input.to(device)
    T = model.cfg.T
    sched = model.schedule

    if mode == "direct":
        # Single-step: predict noise at t=T//2 and reconstruct x̂₀
        t = torch.full((x.shape[0],), T // 2, device=device, dtype=torch.long)
        eps_hat  = model.score_net(x, t)
        sqrt_a   = sched._extract(sched.sqrt_alphas_cumprod, t, x)
        sqrt_om  = sched._extract(sched.sqrt_one_minus_alphas_cumprod, t, x)
        x0_hat   = (x - sqrt_om * eps_hat) / sqrt_a.clamp(min=1e-5)
        return x0_hat.clamp(-4.0, 4.0)

    elif mode == "ddim":
        # DDIM deterministic reverse: uniformly-spaced subset of timesteps
        timesteps = list(reversed(
            np.linspace(0, T - 1, ddim_steps, dtype=int).tolist()
        ))
        x_t = x.clone()
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            eps_hat = model.score_net(x_t, t_batch)

            alpha_t  = sched.alphas_cumprod[t_val]
            if i + 1 < len(timesteps):
                alpha_prev = sched.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # Predict x₀, then step to xₜ₋₁ deterministically
            x0_pred = (x_t - (1 - alpha_t).sqrt() * eps_hat) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-4.0, 4.0)
            dir_xt  = (1 - alpha_prev).sqrt() * eps_hat
            x_t     = alpha_prev.sqrt() * x0_pred + dir_xt

        return x_t.clamp(-4.0, 4.0)

    else:  # "full" DDPM
        return model.sample(x.shape, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_rgb(tensor: torch.Tensor, s2_bands: bool = True) -> np.ndarray:
    """
    Convert a (C, H, W) normalised tensor to an (H, W, 3) uint8 RGB image.
    For S2 data use bands [B4, B3, B2] = indices [3, 2, 1].
    """
    if s2_bands and tensor.shape[0] >= 4:
        rgb = tensor[[3, 2, 1]].cpu().numpy()
    else:
        # Treat first 3 channels as RGB
        rgb = tensor[:3].cpu().numpy()
    # Rescale to [0, 1] using 2nd–98th percentile for visual contrast
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = (rgb - lo) / max(hi - lo, 1e-6)
    rgb = rgb.transpose(1, 2, 0).clip(0, 1)
    return (rgb * 255).astype(np.uint8)


def save_visual(filename: str,
                s2_cloudy: torch.Tensor,
                s2_cloudfree: torch.Tensor,
                prediction_6ch: torch.Tensor,
                out_dir: Path,
                ns_res: float, em_res: float) -> None:
    """Save a 4-panel comparison: cloudy | clear | prediction | residual."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(filename, fontsize=10)

    axes[0].imshow(_to_rgb(s2_cloudy))
    axes[0].set_title("S2 cloudy (input)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(_to_rgb(s2_cloudfree))
    axes[1].set_title("S2 cloud-free (GT)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(_to_rgb(prediction_6ch, s2_bands=False))
    axes[2].set_title("Model prediction", fontsize=9)
    axes[2].axis("off")

    # Residual map: absolute difference on NIR channel
    diff = (pde_field_to_s2_rgb(prediction_6ch.unsqueeze(0)).squeeze(0)
            - s2_cloudfree[[3, 2, 1]]).abs().mean(0).cpu().numpy()
    diff_norm = (diff / diff.max().clip(1e-6)).clip(0, 1)
    im = axes[3].imshow(diff_norm, cmap="hot")
    axes[3].set_title(f"Abs diff  NS:{ns_res:.3f}  EM:{em_res:.3f}", fontsize=9)
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    out_path = out_dir / f"{filename}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def cloud_bucket(frac: float) -> str:
    """Assign a cloud-coverage bucket label."""
    if   frac < 0.10: return "clear"
    elif frac < 0.35: return "light"
    elif frac < 0.65: return "medium"
    elif frac < 0.90: return "heavy"
    else:             return "overcast"


def summarise(records: List[dict]) -> dict:
    """Compute mean ± std over all numeric fields in records."""
    if not records:
        return {}
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float))]
    out  = {}
    for k in keys:
        vals = [r[k] for r in records if not math.isnan(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "min":  float(np.min(vals)),  "max": float(np.max(vals))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main validation loop
# ─────────────────────────────────────────────────────────────────────────────

def validate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SEN12MSCRDataset(
        root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
        max_samples=args.max_samples,
        normalise=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = None
    if args.checkpoint and HAS_MODEL:
        cfg = DiffusionConfig(
            H=args.patch_size, W=args.patch_size, C=6,
            T=1000, schedule="cosine",
            nu=args.nu, eps=args.eps_em, mu=args.mu_em,
            base_channels=args.base_channels,
        )
        model = PDEDiffusionModel(cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Support both raw state_dict and {model_state_dict: ...} checkpoints
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("[INFO] No checkpoint provided — computing dataset statistics only.")

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "val_visuals"
    if args.save_visuals:
        vis_dir.mkdir(exist_ok=True)

    csv_path  = out_dir / "val_results.csv"
    json_path = out_dir / "val_summary.json"

    fieldnames = ["filename", "cloud_frac", "cloud_bucket",
                  "psnr", "ssim", "mae", "sam",
                  "ns_res", "em_res", "div"]

    records: List[dict] = []
    bucket_records: Dict[str, List[dict]] = {
        b: [] for b in ["clear", "light", "medium", "heavy", "overcast"]
    }

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        n_processed = 0
        for batch_idx, batch in enumerate(loader):
            s1           = batch["s1"].to(device)           # (B, 2, H, W)
            s2_cloudy    = batch["s2_cloudy"].to(device)    # (B,13, H, W)
            s2_cloudfree = batch["s2_cloudfree"].to(device) # (B,13, H, W)
            cloud_fracs  = batch["cloud_frac"].to(device)   # (B,)
            filenames    = batch["filename"]                 # list[str]

            B = s1.shape[0]

            # ── Map inputs to PDE field space ─────────────────────────────
            x_input = sen12mscr_to_pde_field(s1, s2_cloudy)  # (B,6,H,W)

            # ── Forward pass ─────────────────────────────────────────────
            if model is not None:
                x_pred = infer_batch(model, x_input,
                                     mode=args.infer_mode,
                                     ddim_steps=args.ddim_steps)
            else:
                x_pred = x_input   # baseline: identity (no denoising)

            # Ground-truth in PDE space (use cloud-free S2 + S1)
            x_target = sen12mscr_to_pde_field(s1, s2_cloudfree)  # (B,6,H,W)

            # ── Per-sample metrics ────────────────────────────────────────
            for i in range(B):
                fname  = filenames[i]
                cf     = cloud_fracs[i].item()
                bucket = cloud_bucket(cf)

                pred_i   = x_pred[i]     # (6, H, W)
                target_i = x_target[i]   # (6, H, W)

                psnr = compute_psnr(pred_i, target_i)
                ssim = compute_ssim(pred_i, target_i)
                mae  = compute_mae(pred_i, target_i)
                sam  = compute_sam(pred_i, target_i)

                phys = compute_physics_metrics(
                    x_pred[i:i+1],
                    nu=args.nu, eps_em=args.eps_em, mu=args.mu_em
                )

                row = {
                    "filename":     fname,
                    "cloud_frac":   round(cf, 4),
                    "cloud_bucket": bucket,
                    "psnr":         round(psnr, 4),
                    "ssim":         round(ssim, 4),
                    "mae":          round(mae,  6),
                    "sam":          round(sam,  4),
                    "ns_res":       round(phys["ns_res"], 6),
                    "em_res":       round(phys["em_res"], 6),
                    "div":          round(phys["div"],    8),
                }
                writer.writerow(row)
                records.append(row)
                bucket_records[bucket].append(row)

                # ── Optional visualisation ────────────────────────────────
                if args.save_visuals and (n_processed % args.vis_every == 0):
                    save_visual(
                        filename=fname,
                        s2_cloudy=s2_cloudy[i].cpu(),
                        s2_cloudfree=s2_cloudfree[i].cpu(),
                        prediction_6ch=pred_i.cpu(),
                        out_dir=vis_dir,
                        ns_res=phys["ns_res"],
                        em_res=phys["em_res"],
                    )
                n_processed += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                last = records[-1]
                print(f"  [{n_processed:5d}/{len(dataset)}]  "
                      f"psnr={last['psnr']:.2f}dB  "
                      f"ssim={last['ssim']:.4f}  "
                      f"mae={last['mae']:.5f}  "
                      f"cloud={last['cloud_frac']:.2f}  "
                      f"ns={last['ns_res']:.4f}  "
                      f"em={last['em_res']:.4f}")

    # ── Aggregate and save summary ────────────────────────────────────────────
    summary = {
        "overall":    summarise(records),
        "n_samples":  len(records),
        "split":      args.split,
        "infer_mode": args.infer_mode,
        "checkpoint": args.checkpoint or "none",
    }
    for bname, brecs in bucket_records.items():
        if brecs:
            summary[f"bucket_{bname}"] = {
                "n": len(brecs),
                **summarise(brecs)
            }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  Validation complete  ·  {len(records)} samples  ·  split={args.split}")
    print("═" * 65)
    ov = summary["overall"]
    for metric in ["psnr", "ssim", "mae", "sam", "ns_res", "em_res", "div"]:
        if metric in ov:
            m = ov[metric]
            print(f"  {metric:<10s}  mean={m['mean']:>10.5f}   std={m['std']:>10.5f}"
                  f"   min={m['min']:>10.5f}   max={m['max']:>10.5f}")
    print("─" * 65)
    print(f"  Per-patch CSV  →  {csv_path}")
    print(f"  Summary JSON   →  {json_path}")
    if args.save_visuals:
        print(f"  Visuals        →  {vis_dir}/")
    print("═" * 65 + "\n")

    # ── Cloud-bucket breakdown ────────────────────────────────────────────────
    print("  Per cloud-coverage bucket (PSNR / SSIM):")
    for bname in ["clear", "light", "medium", "heavy", "overcast"]:
        bkey = f"bucket_{bname}"
        if bkey in summary:
            bp = summary[bkey]
            n  = bp["n"]
            pm = bp.get("psnr", {}).get("mean", float("nan"))
            sm = bp.get("ssim", {}).get("mean", float("nan"))
            print(f"    {bname:<10s}  n={n:5d}  psnr={pm:.2f}dB  ssim={sm:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone dataset statistics (no model required)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_statistics(args: argparse.Namespace) -> None:
    """Scan the dataset and report cloud-coverage distribution."""
    print("\nComputing dataset cloud-coverage statistics …")
    dataset = SEN12MSCRDataset(args.data_root, split=args.split,
                               normalise=False, max_samples=args.max_samples)
    fracs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        fracs.append(sample["cloud_frac"].item())
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(dataset)}")

    fracs = np.array(fracs)
    buckets = {"clear": (fracs < 0.10).sum(),
               "light":   ((fracs >= 0.10) & (fracs < 0.35)).sum(),
               "medium":  ((fracs >= 0.35) & (fracs < 0.65)).sum(),
               "heavy":   ((fracs >= 0.65) & (fracs < 0.90)).sum(),
               "overcast": (fracs >= 0.90).sum()}
    print(f"\n  Cloud-coverage statistics  (n={len(fracs)})")
    print(f"  mean={fracs.mean():.3f}  std={fracs.std():.3f}  "
          f"median={np.median(fracs):.3f}")
    for b, c in buckets.items():
        print(f"    {b:<10s}: {c:5d}  ({c/len(fracs)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate PDE Diffusion Model on SEN12MS-CR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   required=True,
                   help="Root directory of the SEN12MS-CR dataset "
                        "(contains train/ val/ test/ subdirectories).")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate.")
    p.add_argument("--patch_size",  type=int, default=256,
                   help="Spatial size to crop/pad patches to.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the number of samples (useful for quick runs).")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",    default=None,
                   help="Path to model checkpoint (.pt / .pth). "
                        "If omitted, reports dataset statistics only.")
    p.add_argument("--base_channels", type=int, default=64,
                   help="base_channels of the score network (must match training).")
    p.add_argument("--infer_mode",    default="ddim",
                   choices=["direct", "ddim", "full"],
                   help="Inference mode: direct=single-step, "
                        "ddim=accelerated, full=DDPM.")
    p.add_argument("--ddim_steps",    type=int, default=50,
                   help="Number of steps for DDIM sampling.")

    # ── PDE parameters ───────────────────────────────────────────────────────
    p.add_argument("--nu",     type=float, default=1e-3, help="NS viscosity ν.")
    p.add_argument("--eps_em", type=float, default=1.0,  help="EM permittivity ε.")
    p.add_argument("--mu_em",  type=float, default=1.0,  help="EM permeability μ.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",  default="./val_outputs",
                   help="Directory to write CSV / JSON / visuals.")
    p.add_argument("--save_visuals", action="store_true",
                   help="Save qualitative comparison images.")
    p.add_argument("--vis_every",   type=int, default=50,
                   help="Save a visual every N samples.")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                            else "cpu")

    # ── Modes ─────────────────────────────────────────────────────────────────
    p.add_argument("--stats_only",  action="store_true",
                   help="Only compute cloud-coverage statistics, skip inference.")

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Programmatic API (import from notebooks / other scripts)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    data_root:    str,
    checkpoint:   Optional[str] = None,
    split:        str  = "test",
    infer_mode:   str  = "ddim",
    ddim_steps:   int  = 50,
    patch_size:   int  = 256,
    batch_size:   int  = 4,
    num_workers:  int  = 4,
    max_samples:  Optional[int] = None,
    output_dir:   str  = "./val_outputs",
    save_visuals: bool = False,
    vis_every:    int  = 50,
    device:       Optional[str] = None,
    nu:           float = 1e-3,
    eps_em:       float = 1.0,
    mu_em:        float = 1.0,
    base_channels: int  = 64,
) -> dict:
    """
    Programmatic entry-point for use in notebooks or pipelines.

    Returns the parsed summary dict.

    Example
    -------
    >>> from validate_sen12mscr import run_validation
    >>> summary = run_validation(
    ...     data_root   = "/data/SEN12MS-CR",
    ...     checkpoint  = "checkpoints/pde_diffusion_step100k.pt",
    ...     infer_mode  = "ddim",
    ...     ddim_steps  = 50,
    ...     max_samples = 500,
    ... )
    >>> print(summary["overall"]["psnr"])
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ns = argparse.Namespace(
        data_root=data_root, split=split, patch_size=patch_size,
        max_samples=max_samples, batch_size=batch_size, num_workers=num_workers,
        checkpoint=checkpoint, base_channels=base_channels,
        infer_mode=infer_mode, ddim_steps=ddim_steps,
        nu=nu, eps_em=eps_em, mu_em=mu_em,
        output_dir=output_dir, save_visuals=save_visuals, vis_every=vis_every,
        device=device, stats_only=False,
    )
    validate(ns)
    json_path = Path(output_dir) / "val_summary.json"
    with open(json_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    print(f"\nPDE Diffusion Model — SEN12MS-CR Validation")
    print(f"  data_root   : {args.data_root}")
    print(f"  split       : {args.split}")
    print(f"  checkpoint  : {args.checkpoint or '(none)'}")
    print(f"  infer_mode  : {args.infer_mode}")
    print(f"  device      : {args.device}")
    print()

    if args.stats_only or args.checkpoint is None:
        compute_dataset_statistics(args)
    else:
        validate(args)
