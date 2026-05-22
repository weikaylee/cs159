# sen_loss_jax.py

from typing import List, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp


# =============================================================================
# VGG FEATURE EXTRACTOR
# =============================================================================

class VGGBlock(nn.Module):
    features: int
    n_convs: int

    @nn.compact
    def __call__(self, x):

        for _ in range(self.n_convs):
            x = nn.Conv(
                features=self.features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                use_bias=True,
            )(x)

            x = nn.relu(x)

        return x


class FlaxVGGFeatureExtractor(nn.Module):
    """
    Lightweight VGG-style feature extractor for Sentinel-2 imagery.

    Input:
        (B, H, W, C)

    Output:
        List of feature tensors at multiple scales.
    """

    n_input_channels: int = 13

    @nn.compact
    def __call__(self, x):

        # ---------------------------------------------------------------------
        # Sentinel-2 band projection: 13 -> 3
        #
        # Mimics:
        #   R <- B4
        #   G <- B3
        #   B <- B2
        #
        # Input assumed:
        #   [B1, B2, B3, B4, ...]
        # ---------------------------------------------------------------------

        x = nn.Conv(
            features=3,
            kernel_size=(1, 1),
            use_bias=False,
            name='band_projection'
        )(x)

        # ---------------------------------------------------------------------
        # ImageNet normalization
        # ---------------------------------------------------------------------

        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])

        mean = mean.reshape((1, 1, 1, 3))
        std = std.reshape((1, 1, 1, 3))

        x = (x - mean) / std

        feats = []

        # ---------------------------------------------------------------------
        # VGG Block 1
        # ---------------------------------------------------------------------

        x = VGGBlock(features=64, n_convs=2)(x)
        feats.append(x)

        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='SAME'
        )

        # ---------------------------------------------------------------------
        # VGG Block 2
        # ---------------------------------------------------------------------

        x = VGGBlock(features=128, n_convs=2)(x)
        feats.append(x)

        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='SAME'
        )

        # ---------------------------------------------------------------------
        # VGG Block 3
        # ---------------------------------------------------------------------

        x = VGGBlock(features=256, n_convs=3)(x)
        feats.append(x)

        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='SAME'
        )

        # ---------------------------------------------------------------------
        # VGG Block 4
        # ---------------------------------------------------------------------

        x = VGGBlock(features=512, n_convs=3)(x)
        feats.append(x)

        return feats


# =============================================================================
# RECONSTRUCTION LOSS
# =============================================================================

def l_recon(
    x_hat: jnp.ndarray,
    x_ref: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-example reconstruction MSE.

    Args:
        x_hat: (B, H, W, C)
        x_ref: (B, H, W, C)

    Returns:
        (B,)
    """

    return jnp.mean(
        (x_hat - x_ref) ** 2,
        axis=(1, 2, 3)
    )


# =============================================================================
# DISTRIBUTION LOSS
# =============================================================================

def l_dis(
    feats_gen: List[jnp.ndarray],
    feats_ref: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    Distribution matching loss.

    Computes feature mean differences at multiple scales.

    Returns:
        (B,)
    """

    losses = []

    for fg, fr in zip(feats_gen, feats_ref):

        # Mean over spatial dimensions only
        mu_gen = jnp.mean(fg, axis=(1, 2))
        mu_ref = jnp.mean(fr, axis=(1, 2))

        layer_loss = jnp.sum(
            (mu_gen - mu_ref) ** 2,
            axis=-1
        )

        losses.append(layer_loss)

    return sum(losses)


# =============================================================================
# STYLE LOSS
# =============================================================================

def gram_matrix(
    feat: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Gram matrix.

    Args:
        feat: (B, H, W, C)

    Returns:
        (B, C, C)
    """

    B, H, W, C = feat.shape

    feat = feat.reshape(B, H * W, C)

    gram = jnp.einsum(
        'bic,bid->bcd',
        feat,
        feat
    )

    # Use Python float to avoid int32 overflow in JAX's default x32 mode.
    # (C * H * W)^2 can exceed 2^31 for large feature maps (e.g. 64×64×64).
    norm = float(C * H * W) ** 2

    return gram / norm


def l_style(
    feats_gen: List[jnp.ndarray],
    feats_ref: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    Gram-matrix style loss.

    Returns:
        (B,)
    """

    losses = []

    for fg, fr in zip(feats_gen, feats_ref):

        Gg = gram_matrix(fg)
        Gr = gram_matrix(fr)

        layer_loss = jnp.mean(
            (Gg - Gr) ** 2,
            axis=(1, 2)
        )

        losses.append(layer_loss)

    return sum(losses)


# =============================================================================
# SPECTRAL ANGLE MAPPER
# =============================================================================

def spectral_angle_mapper(
    x_hat: jnp.ndarray,
    x_ref: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Mean spectral angle per example.

    Args:
        x_hat: (B, H, W, C)
        x_ref: (B, H, W, C)

    Returns:
        (B,)
    """

    dot = jnp.sum(
        x_hat * x_ref,
        axis=-1
    )

    nx = jnp.linalg.norm(
        x_hat,
        axis=-1
    )

    ny = jnp.linalg.norm(
        x_ref,
        axis=-1
    )

    nx = jnp.clip(nx, eps)
    ny = jnp.clip(ny, eps)

    cos = dot / (nx * ny)

    cos = jnp.clip(cos, -1.0, 1.0)

    sam = jnp.arccos(cos)

    return jnp.mean(
        sam,
        axis=(1, 2)
    )


# =============================================================================
# COMBINED CONSTRAINT LOSS
# =============================================================================

class ConstraintLoss(nn.Module):
    """
    Combined SEN12MS constraint loss.

    l_total =
        recon_weight * reconstruction
        + dis_weight * distribution
        + style_weight * style
        + sam_weight * SAM
    """

    n_input_channels: int = 13

    recon_weight: float = 1.0
    dis_weight: float = 1.0
    style_weight: float = 100.0
    sam_weight: float = 0.0

    @nn.compact
    def __call__(
        self,
        x_hat: jnp.ndarray,
        x_ref: jnp.ndarray,
    ) -> jnp.ndarray:

        # ---------------------------------------------------------------------
        # Clamp reflectance range
        # ---------------------------------------------------------------------

        x_hat = jnp.clip(x_hat, 0.0, 1.0)

        # ---------------------------------------------------------------------
        # Feature extractor
        # ---------------------------------------------------------------------

        vgg = FlaxVGGFeatureExtractor(
            n_input_channels=self.n_input_channels
        )

        feats_hat = vgg(x_hat)
        feats_ref = vgg(x_ref)

        # ---------------------------------------------------------------------
        # Individual losses
        # ---------------------------------------------------------------------

        recon = l_recon(x_hat, x_ref)

        dis = l_dis(
            feats_hat,
            feats_ref
        )

        style = l_style(
            feats_hat,
            feats_ref
        )

        sam = spectral_angle_mapper(
            x_hat,
            x_ref
        )

        # ---------------------------------------------------------------------
        # Total
        # ---------------------------------------------------------------------

        total = (
            self.recon_weight * recon
            + self.dis_weight * dis
            + self.style_weight * style
            # + self.sam_weight * sam
        )

        return total


# =============================================================================
# METRICS
# =============================================================================

def reconstruction_metrics(
    x_hat: jnp.ndarray,
    x_ref: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Reconstruction quality metrics.

    Returns:
        dict of scalar batch means
    """

    x_hat = jnp.clip(x_hat, 0.0, 1.0)

    mae = jnp.mean(
        jnp.abs(x_hat - x_ref)
    )

    mse = jnp.mean(
        (x_hat - x_ref) ** 2
    )

    psnr = -10.0 * jnp.log10(mse + 1e-8)

    sam = jnp.mean(
        spectral_angle_mapper(x_hat, x_ref)
    )

    return {
        'mae': mae,
        'mse': mse,
        'psnr': psnr,
        'sam': sam,
    }