"""
Constraint distance function l_constr and NAMM training losses.

l_constr = L_recon + L_dis + style_weight * L_style
  - L_recon: pixel-wise MSE between generated and reference images.
  - L_dis:   MMD-style distribution loss via VGG feature maps in RKHS.
  - L_style: Gram-matrix style loss over VGG feature maps.

The combined loss is differentiable end-to-end, allowing gradients to flow
through f_psi during NAMM joint training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── VGG feature extractor ──────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """Extract intermediate VGG-16 feature maps for style and distribution loss.

    Because Sentinel-2 has 13 bands, we project to 3 channels before passing
    through VGG.  The projection is a learned 1x1 conv initialised to average
    the bands most similar to RGB (B4, B3, B2 → indices 3, 2, 1).
    """

    # VGG-16 layers up to relu1_2, relu2_2, relu3_3, relu4_3
    LAYER_INDICES = [4, 9, 16, 23]

    def __init__(self, n_input_channels: int = 13):
        super().__init__()
        # Band projection: 13 → 3
        self.band_proj = nn.Conv2d(n_input_channels, 3, 1, bias=False)
        # Initialise to select the RGB-like bands (B4=3, B3=2, B2=1 in 0-index)
        with torch.no_grad():
            w = torch.zeros(3, n_input_channels, 1, 1)
            w[0, 3] = 1.0   # Red   ← B4
            w[1, 2] = 1.0   # Green ← B3
            w[2, 1] = 1.0   # Blue  ← B2
            self.band_proj.weight.copy_(w)

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_INDICES:
            self.slices.append(nn.Sequential(*features[prev:idx + 1]))
            prev = idx + 1

        # Freeze VGG weights
        for p in self.parameters():
            if p is not self.band_proj.weight:
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of feature maps at each VGG slice."""
        h = self.band_proj(x)
        # Normalise to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406],
                             device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        h = (h - mean) / std
        feats = []
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


# ── Individual loss terms ───────────────────────────────────────────────────────

def l_recon(x_hat: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Pixel-wise MSE reconstruction loss (Eq. 1 in proposal)."""
    return F.mse_loss(x_hat, x_ref)


def l_dis(feats_gen: list[torch.Tensor],
          feats_ref: list[torch.Tensor]) -> torch.Tensor:
    """Distribution loss: squared MMD on mean feature maps (Eq. 2).

    Computes ||mean(phi(x_gen)) - mean(phi(x_ref))||^2 over the batch,
    summed across VGG layers.  phi is implicitly the RKHS feature map.
    """
    loss = 0.0
    for fg, fr in zip(feats_gen, feats_ref):
        # Mean over batch, then over spatial dims → (C,)
        mu_gen = fg.mean(dim=(0, 2, 3))
        mu_ref = fr.mean(dim=(0, 2, 3))
        loss = loss + (mu_gen - mu_ref).pow(2).sum()
    return loss


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Gram matrix G = F @ F^T / (C * H * W)^2, shape (B, C, C)."""
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W) ** 2


def l_style(feats_gen: list[torch.Tensor],
            feats_ref: list[torch.Tensor]) -> torch.Tensor:
    """Gram-matrix style loss (Eq. 3)."""
    loss = 0.0
    for fg, fr in zip(feats_gen, feats_ref):
        Gg = gram_matrix(fg)
        Gr = gram_matrix(fr)
        loss = loss + (Gg - Gr).pow(2).sum(dim=(1, 2)).mean()
    return loss


# ── Combined constraint distance function ──────────────────────────────────────

class ConstraintLoss(nn.Module):
    """l_constr = L_recon + L_dis + style_weight * L_style.

    Used both to define the NAMM constraint and to evaluate spectral consistency.

    Args:
        style_weight:     Weight for the style loss (100 as in Yu et al.).
        n_input_channels: Number of image channels (13 for Sentinel-2).
    """

    def __init__(self, style_weight: float = 100.0,
                 n_input_channels: int = 13):
        super().__init__()
        self.style_weight = style_weight
        self.vgg = VGGFeatureExtractor(n_input_channels)

    def forward(self, x_hat: torch.Tensor,
                x_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hat: (B, C, H, W) images restored to constrained space by f_psi.
            x_ref: (B, C, H, W) cloud-free reference images.
        Returns:
            Scalar constraint distance.
        """
        recon = l_recon(x_hat, x_ref)

        feats_hat = self.vgg(x_hat)
        feats_ref = self.vgg(x_ref)

        dis   = l_dis(feats_hat, feats_ref)
        style = l_style(feats_hat, feats_ref)

        return recon + dis + self.style_weight * style


# ── Reconstruction-quality metrics ─────────────────────────────────────────────

def spectral_angle_mapper(x_hat: torch.Tensor,
                          x_ref: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    """Mean spectral angle (radians) between x_hat and x_ref across pixels.

    Args:
        x_hat, x_ref: (B, C, H, W) reflectance tensors.
    Returns:
        Scalar tensor: mean of arccos of per-pixel cosine similarity
        along the channel axis. Lower = more spectrally consistent.
    """
    dot = (x_hat * x_ref).sum(dim=1)
    nx = x_hat.norm(dim=1).clamp(min=eps)
    ny = x_ref.norm(dim=1).clamp(min=eps)
    cos = (dot / (nx * ny)).clamp(-1.0, 1.0)
    return torch.acos(cos).mean()


def reconstruction_metrics(x_hat: torch.Tensor,
                           x_ref: torch.Tensor,
                           data_range: float = 1.0) -> dict:
    """MAE / SAM (rad) / PSNR (dB) / SSIM for x_hat vs x_ref.

    x_hat is clamped to [0, data_range] before PSNR/SSIM so the
    InverseMap residual connection's tail values don't blow up the
    metric.
    """
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    x_clipped = x_hat.clamp(0.0, data_range)
    return {
        'mae':  F.l1_loss(x_clipped, x_ref),
        'sam':  spectral_angle_mapper(x_clipped, x_ref),
        'psnr': peak_signal_noise_ratio(x_clipped, x_ref, data_range=data_range),
        'ssim': structural_similarity_index_measure(x_clipped, x_ref,
                                                   data_range=data_range),
    }


# ── Full NAMM objective ─────────────────────────────────────────────────────────

def namm_loss(g_phi: nn.Module,
              f_psi: nn.Module,
              constraint_loss: ConstraintLoss,
              x: torch.Tensor,
              max_sigma: float = 0.1,
              cycle_weight: float = 1.0,
              constraint_weight: float = 1.0,
              reg_weight: float = 0.001,
              strong_convexity: float = 0.3,
              device: torch.device = None) -> dict:
    """Full NAMM training loss for one batch.

    L = cycle_weight * L_cycle
      + constraint_weight * L_constr
      + reg_weight * L_reg

    Args:
        g_phi:             Forward mirror map (ICNNGradient).
        f_psi:             Inverse mirror map (InverseMap).
        constraint_loss:   ConstraintLoss module.
        x:                 (B, C, H, W) cloud-free Sentinel-2 batch.
        max_sigma:         Max noise level for inverse map robustness training.
        cycle_weight:      Weight for cycle-consistency loss.
        constraint_weight: Weight for constraint distance loss.
        reg_weight:        Weight for ICNN sparsity regularisation.
        strong_convexity:  Strong-convexity coefficient alpha.
        device:            Torch device.

    Returns:
        Dict with keys: loss, l_cycle, l_constr, l_reg, x_recon
        where x_recon = f_psi(g_phi(x)) is the cycle reconstruction
        (detached) — used by reconstruction_metrics() in the training
        loop to compute MAE / SAM / PSNR / SSIM without a second
        forward pass.
    """
    B = x.shape[0]
    if device is None:
        device = x.device

    # ── Forward pass: x → mirror space ───────────────────────────────────────
    y = g_phi(x)                         # (B, C, H, W) in mirror space

    # ── Cycle: x → y → x_hat → y_hat ────────────────────────────────────────
    x_fwdbwd = f_psi(y)                  # should recover x
    y_bwdfwd = g_phi(x_fwdbwd)          # should recover y

    l_cycle_fwd = F.l1_loss(x_fwdbwd, x)
    l_cycle_bwd = F.l1_loss(y_bwdfwd, y)
    l_cyc = 0.5 * l_cycle_fwd + 0.5 * l_cycle_bwd

    # ── Constraint loss: noisy mirror → inverse → constrained ────────────────
    # Sample noise levels uniformly in [0, max_sigma]
    sigmas = torch.rand(B, device=device) * max_sigma          # (B,)
    noise  = torch.randn_like(y)
    y_noisy = y + sigmas.view(B, 1, 1, 1) * noise             # perturbed mirror

    x_hat = f_psi(y_noisy)                                     # restored image
    l_c   = constraint_loss(x_hat, x)

    # ── Regularisation: sparse ICNN (encourage sparse gradients of g_phi) ────
    # grad_phi = (y - alpha * x) / (1 - alpha)  where alpha = strong_convexity
    grad = (y.detach() - strong_convexity * x.detach()) / (1.0 - strong_convexity)
    l_r  = grad.abs().mean()

    # ── Total loss ────────────────────────────────────────────────────────────
    total = (cycle_weight * l_cyc
             + constraint_weight * l_c
             + reg_weight * l_r)

    return {
        'loss':     total,
        'l_cycle':  l_cyc.detach(),
        'l_constr': l_c.detach(),
        'l_reg':    l_r.detach(),
        'x_recon':  x_fwdbwd.detach(),
    }
