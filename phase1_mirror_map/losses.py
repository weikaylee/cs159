"""
Constraint distance function l_constr and NAMM training losses.

l_constr = L_recon + dis_weight * L_dis + style_weight * L_style
                    + sam_weight * L_sam + moment_weight * L_moments
    - L_recon:   pixel-wise MSE between generated and reference images.
    - L_dis:     Gaussian-kernel MMD distribution loss on VGG-19 content.
    - L_style:   Gram-matrix style loss over VGG-19 style layers.
    - L_sam:     Spectral Angle Mapper loss (per-pixel spectral angle).
    - L_moments: per-band mean + std matching across batch/spatial dims.

The combined loss is differentiable end-to-end, allowing gradients to flow
through f_psi during NAMM joint training.
"""

import torch
from typing import cast
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── VGG feature extractor ──────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """Extract VGG-19 feature maps for style and content losses.

    Style layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 with weights
    [1.0, 0.8, 0.5, 0.3, 0.1]. Content/distribution layer: conv4_2.

    Because Sentinel-2 has 13 bands, we select the RGB channels directly
    before passing through VGG: B04 (Red), B03 (Green), B02 (Blue).
    """

    STYLE_LAYER_INDICES = [1, 6, 11, 20, 29]   # relu1_1 ... relu5_1
    CONTENT_LAYER_INDEX = 22                  # relu4_2
    STYLE_WEIGHTS = [1.0, 0.8, 0.5, 0.3, 0.1]

    def __init__(self, n_input_channels: int = 13):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = vgg.features

        # Freeze VGG weights
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def style_weights(self) -> list[float]:
        return self.STYLE_WEIGHTS

    def forward(self, x: torch.Tensor) -> dict:
        """Return dict with style feature list and content feature map."""
        # Sentinel-2 RGB bands (1-indexed): B02=Blue, B03=Green, B04=Red
        # Convert to 0-index: [1, 2, 3], ordered as RGB: [Red, Green, Blue]
        h = x[:, [3, 2, 1], ...]
        # Normalise to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406],
                             device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        h = (h - mean) / std

        style_feats = []
        content_feat = None
        max_idx = max(self.STYLE_LAYER_INDICES + [self.CONTENT_LAYER_INDEX])
        features = list(self.features.children())
        for idx, layer in enumerate(features):
            h = layer(h)
            if idx in self.STYLE_LAYER_INDICES:
                style_feats.append(h)
            if idx == self.CONTENT_LAYER_INDEX:
                content_feat = h
            if idx >= max_idx:
                break

        if content_feat is None:
            raise RuntimeError("VGG content feature not captured (conv4_2).")

        return {
            'style': style_feats,
            'content': content_feat,
        }


# ── Individual loss terms ───────────────────────────────────────────────────────

def l_recon(x_hat: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Pixel-wise MSE reconstruction loss (Eq. 1 in proposal)."""
    return F.mse_loss(x_hat, x_ref)


def _gaussian_mmd_squared(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """Squared MMD between two sample sets using a multi-bandwidth Gaussian kernel.

    Implements MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)] via the kernel trick,
    which is the RKHS norm of the difference of mean embeddings.

    Args:
        source: (N_s, D) samples from the generated distribution.
        target: (N_t, D) samples from the reference distribution.
        kernel_mul: Multiplier between successive kernel bandwidths.
        kernel_num: Number of Gaussian kernels to sum (multi-bandwidth trick).

    Returns:
        Scalar tensor: squared MMD.
    """
    n_s, n_t = source.size(0), target.size(0)
    total = torch.cat([source, target], dim=0)             # (N_s + N_t, D)

    # Pairwise squared L2 distances
    sq_dist = torch.cdist(total, total, p=2.0) ** 2        # (N, N)

    # Median heuristic for the base bandwidth (detached so it's a hyperparam)
    with torch.no_grad():
        n_total = sq_dist.size(0)
        bandwidth = sq_dist.sum() / (n_total * n_total - n_total + 1e-8)
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    # Sum of Gaussian kernels at multiple bandwidths
    kernels = torch.zeros_like(sq_dist)
    for i in range(kernel_num):
        kernels = kernels + torch.exp(
            -sq_dist / (bandwidth * (kernel_mul ** i) + 1e-8)
        )

    K_ss = kernels[:n_s, :n_s]
    K_tt = kernels[n_s:, n_s:]
    K_st = kernels[:n_s, n_s:]
    return K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()


def l_dis(
    feats_gen: list[torch.Tensor] | torch.Tensor,
    feats_ref: list[torch.Tensor] | torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    max_samples: int = 1024,
) -> torch.Tensor:
    """Distribution loss: squared MMD in a Gaussian-kernel RKHS (paper Eq. 4).

    L_dis = || (1/N) sum_i phi(x_i) - (1/N) sum_j phi(x_hat_j) ||_H^2

    where phi is the implicit feature map of a Gaussian kernel. Computed via
    the kernel trick — we never materialise phi(x). Each spatial location in
    each feature map is treated as a sample in C-dim space.

    For VGG-19, pass either:
      - a single content feature map (conv4_2) as a Tensor, or
      - a list of feature maps (e.g. 5 style layers) — losses are summed.

    Args:
        feats_gen: (B, C, H, W) generated feature map(s).
        feats_ref: (B, C, H, W) reference feature map(s).
        kernel_mul, kernel_num: Multi-bandwidth Gaussian kernel parameters.
        max_samples: Cap on samples per layer to keep the kernel matrix tractable.

    Returns:
        Scalar tensor: summed squared MMD across layers.
    """
    if isinstance(feats_gen, torch.Tensor):
        feats_gen_list = [feats_gen]
        feats_ref_list = [cast(torch.Tensor, feats_ref)]
    else:
        feats_gen_list = cast(list[torch.Tensor], feats_gen)
        feats_ref_list = cast(list[torch.Tensor], feats_ref)

    loss = torch.zeros((), device=feats_gen_list[0].device)
    for fg, fr in zip(feats_gen_list, feats_ref_list):
        b, c, h, w = fg.shape
        # Each spatial location is a sample in R^C: (B*H*W, C)
        src = fg.permute(0, 2, 3, 1).reshape(-1, c)
        tgt = fr.permute(0, 2, 3, 1).reshape(-1, c)

        # Subsample for memory — kernel matrix is O(N^2)
        if src.size(0) > max_samples:
            idx = torch.randperm(src.size(0), device=src.device)[:max_samples]
            src = src[idx]
        if tgt.size(0) > max_samples:
            idx = torch.randperm(tgt.size(0), device=tgt.device)[:max_samples]
            tgt = tgt[idx]

        loss = loss + _gaussian_mmd_squared(
            src, tgt, kernel_mul=kernel_mul, kernel_num=kernel_num
        )
    return loss


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Unnormalised Gram matrix G = F F^T, shape (B, C, C).
    
    Captures channel-wise correlations in the feature map. Normalisation
    is applied in the style loss (see l_style) following Gatys et al. /
    SatelliteMaker's Eq. 6.
    """
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    return torch.bmm(f, f.transpose(1, 2))


# ── Design note: a scale-invariant alternative to the Gatys L_style ──────────
#
# The Gatys formula below normalises by 4·C²·H²·W² but the Gram entries
# themselves scale as O(M·a²) (M = H·W, a = activation magnitude), so per
# layer the loss scales as O(a⁴). On ImageNet-normalised Sentinel-2 RGB with
# VGG-19 ReLU activations of O(10–100), L_style lands at ~1e7 per layer; at
# the paper-recommended `style_weight=100` it contributes ~1e9 to the
# constraint loss while cycle (~24), recon (~0.5), and the spectral terms
# (~0.5) sit 6–7 orders of magnitude below. The imbalance is worse at init:
# untrained `f_psi` outputs land in [0, ~40] (the `InverseMap` residual
# passes the mirror-space y through, ReLU clips only negatives), so
# ‖G_gen‖ ≫ ‖G_ref‖ and any one-sided relative form like
# ‖ΔG‖² / ‖G_ref‖² also diverges.
#
# A symmetric cosine-distance variant (unit-Frobenius-normalise both Grams,
# then squared diff) bounds the per-layer loss in [0, 4] regardless of
# activation scale. It discards Gram-magnitude information that the Gatys
# formula encodes, but that magnitude signal is already captured elsewhere
# in the constraint distance — `l_recon` (pixel-MSE) and `l_moments`
# (per-band mean / std matching). The project keeps the Gatys formulation
# below so style stays faithful to SatelliteMaker's Eq. 6; uncomment the
# block below to switch back to the bounded variant.
#
# def l_style(
#     feats_gen: list[torch.Tensor],
#     feats_ref: list[torch.Tensor],
#     weights: list[float] | None = None,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """Scale-invariant style loss: cosine distance between Gram matrices.
#
#     L_style = sum_l w_l · ‖G_gen/‖G_gen‖ − G_ref/‖G_ref‖‖_F²,  in [0, 4].
#     """
#     if weights is None:
#         weights = [1.0] * len(feats_gen)
#     if len(weights) != len(feats_gen):
#         raise ValueError("Style weights must match number of feature maps.")
#     loss = torch.zeros((), device=feats_gen[0].device)
#     for fg, fr, w in zip(feats_gen, feats_ref, weights):
#         Gg = gram_matrix(fg)
#         Gr = gram_matrix(fr)
#         ng = Gg.flatten(1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
#         nr = Gr.flatten(1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
#         diff_sq = (Gg / ng - Gr / nr).pow(2).sum(dim=(1, 2))
#         loss = loss + w * diff_sq.mean()
#     return loss


def l_style(
    feats_gen: list[torch.Tensor],
    feats_ref: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Gram-matrix style loss over style layers (paper Eq. 6).

    L_style = sum_l  w_l * (1 / (4 C_l^2 H_l^2 W_l^2)) * || G_gen^l - G_ref^l ||_F^2

    """
    if weights is None:
        weights = [1.0] * len(feats_gen)
    if len(weights) != len(feats_gen):
        raise ValueError("Style weights must match number of feature maps.")

    loss = torch.zeros((), device=feats_gen[0].device)
    for fg, fr, w in zip(feats_gen, feats_ref, weights):
        _, C, H, W = fg.shape
        Gg = gram_matrix(fg)
        Gr = gram_matrix(fr)
        # Squared Frobenius norm of the Gram difference, per sample
        diff_sq = (Gg - Gr).pow(2).sum(dim=(1, 2))           # (B,)
        # Paper's normalisation: 1 / (4 C^2 H^2 W^2)
        denom = 4.0 * (C ** 2) * (H ** 2) * (W ** 2)
        loss = loss + w * diff_sq.mean() / denom
    return loss


def l_moments(x_hat: torch.Tensor,
              x_ref: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Per-band mean + std matching across batch and spatial dims."""
    mean_hat = x_hat.mean(dim=(0, 2, 3))
    mean_ref = x_ref.mean(dim=(0, 2, 3))
    std_hat = x_hat.std(dim=(0, 2, 3), unbiased=False).clamp(min=eps)
    std_ref = x_ref.std(dim=(0, 2, 3), unbiased=False).clamp(min=eps)
    return F.mse_loss(mean_hat, mean_ref) + F.mse_loss(std_hat, std_ref)


# ── Combined constraint distance function ──────────────────────────────────────

class ConstraintLoss(nn.Module):
    """l_constr = L_recon + dis_weight * L_dis + style_weight * L_style +
    sam_weight * L_sam + moment_weight * L_moments.

    Used both to define the NAMM constraint and to evaluate spectral consistency.

    Args:
        style_weight:     Weight for the style loss (100 as in Yu et al.).
        n_input_channels: Number of image channels (13 for Sentinel-2).
        sam_weight:       Weight for SAM spectral-angle loss.
        moment_weight:    Weight for per-band mean/std matching.
        dis_weight:       Weight for the MMD distribution loss. Set to 0
                          (together with style_weight=0) to disable the
                          VGG-based perceptual terms entirely.
    """

    def __init__(self, style_weight: float = 100.0,
                 n_input_channels: int = 13,
                 sam_weight: float = 0,
                 moment_weight: float = 0,
                 dis_weight: float = 1.0):
        super().__init__()
        self.style_weight = style_weight
        self.sam_weight = sam_weight
        self.moment_weight = moment_weight
        self.dis_weight = dis_weight
        self.vgg = VGGFeatureExtractor(n_input_channels)

    def forward_terms(self, x_hat: torch.Tensor,
                      x_ref: torch.Tensor) -> dict:
        """
        Args:
            x_hat: (B, C, H, W) images restored to constrained space by f_psi.
            x_ref: (B, C, H, W) cloud-free reference images.
        Returns:
            Dict with the total loss, the sub-losses, and the VGG
            feature maps used to compute them.
        """
        recon = l_recon(x_hat, x_ref)

        # L_dis and L_style both depend on the VGG feature maps. When both are
        # disabled, skip the two VGG forward passes — the most expensive part
        # of this loss — since their output would be multiplied by zero.
        if self.dis_weight == 0 and self.style_weight == 0:
            feats_hat = None
            feats_ref = None
            dis = torch.zeros((), device=x_hat.device)
            style = torch.zeros((), device=x_hat.device)
        else:
            feats_hat = self.vgg(x_hat)
            feats_ref = self.vgg(x_ref)
            dis = l_dis(feats_hat['content'], feats_ref['content'])
            style = l_style(feats_hat['style'], feats_ref['style'],
                    weights=self.vgg.style_weights)

        sam = l_sam(x_hat, x_ref)
        moments = l_moments(x_hat, x_ref)
        loss = (recon
            + self.dis_weight * dis
            + self.style_weight * style
            + self.sam_weight * sam
            + self.moment_weight * moments)

        return {
            'loss': loss,
            'recon': recon,
            'dis': dis,
            'style': style,
            'sam': sam,
            'moments': moments,
            'feats_hat': feats_hat,
            'feats_ref': feats_ref,
        }

    def forward(self, x_hat: torch.Tensor,
                x_ref: torch.Tensor) -> torch.Tensor:
        """Return the scalar constraint distance."""
        return self.forward_terms(x_hat, x_ref)['loss']


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


def l_sam(x_hat: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Mapper loss (mean angle over all pixels)."""
    return spectral_angle_mapper(x_hat, x_ref)


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

def namm_loss(
    g_phi: nn.Module,
    f_psi: nn.Module,
    constraint_loss: ConstraintLoss,
    x: torch.Tensor,
    max_sigma: float = 0.1,
    cycle_weight: float = 1.0,
    constraint_weight: float = 1.0,
    reg_weight: float = 0.001,
    device: torch.device | None = None,
) -> dict:
    """Full NAMM training loss for one batch (Eqs. 1, 3, 4, 5 of Feng et al. 2025).

    L_total = cycle_weight * L_cycle + constraint_weight * L_constr + reg_weight * R(g_phi)

    where
        L_cycle  = E_x ||x - f_psi(g_phi(x))||_1
                 + E_{x, sigma~U, z~N} ||y_noisy - g_phi(f_psi(y_noisy))||_1
        L_constr = E_{x, sigma~U, z~N} [ ell_constr(f_psi(y_noisy)) ]
        R        = E_x ||x - g_phi(x)||_1
    """
    B = x.shape[0]
    if device is None:
        device = x.device

    # ── Forward pass: x → mirror space ───────────────────────────────────────
    y = g_phi(x)                                    # (B, C, H, W)

    # ── Sample noisy mirror points (shared by cycle-backward and constraint) ─
    sigmas = torch.rand(B, device=device) * max_sigma       # (B,)
    z = torch.randn_like(y)
    y_noisy = y + sigmas.view(B, 1, 1, 1) * z              # (B, C, H, W)

    # ── Cycle loss (Eq. 3) ───────────────────────────────────────────────────
    # Forward direction: x -> g_phi -> f_psi should recover x (clean)
    x_recon = f_psi(y)
    l_cycle_fwd = F.l1_loss(x_recon, x)

    # Backward direction: noisy y -> f_psi -> g_phi should recover noisy y
    x_from_noisy = f_psi(y_noisy)
    y_recon = g_phi(x_from_noisy)
    l_cycle_bwd = F.l1_loss(y_recon, y_noisy)

    l_cyc = l_cycle_fwd + l_cycle_bwd

    # ── Constraint loss (Eq. 4) ──────────────────────────────────────────────
    # Reuse x_from_noisy from above to avoid a second f_psi forward pass
    constr_terms = constraint_loss.forward_terms(x_from_noisy, x)
    l_c = constr_terms["loss"]

    # ── Regularisation (Eq. 5): g_phi should be close to identity ────────────
    l_r = F.l1_loss(y, x)

    # ── Total loss ───────────────────────────────────────────────────────────
    total = (
        cycle_weight * l_cyc
        + constraint_weight * l_c
        + reg_weight * l_r
    )

    return {
        "loss": total,
        "l_cycle": l_cyc.detach(),
        "l_constr": l_c.detach(),
        "l_reg": l_r.detach(),
        "x_recon": x_recon.detach(),
    }
