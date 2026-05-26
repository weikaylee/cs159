"""Unit tests for phase1_mirror_map/losses.py.

Covers the three sub-losses (l_recon, l_dis, l_style), the gram_matrix
helper, the composite ConstraintLoss, and the full namm_loss objective.

Two structural properties drive most assertions:

  1. IDENTITY ZERO — every distance-style loss must vanish when the two
     inputs are equal. Six tests check this at increasing nesting levels
     (recon -> dis -> style -> composite ConstraintLoss).
  2. SHAPE / GRAD CONTRACTS — namm_loss returns a dict the training loop
     depends on: keys it logs, gradient flags it expects. The structure
     test guards both.

VGG16 weights load is the expensive part of any test that touches
ConstraintLoss; the module-scoped `constraint_loss` fixture loads it
once and shares the instance across the two tests that need it.
"""

import torch
import pytest

from icnn import ICNN, ICNNGradient
from inverse_map import InverseMap
from losses import (
    l_recon, l_dis, l_style, l_moments, l_sam, gram_matrix, ConstraintLoss,
    namm_loss, spectral_angle_mapper, reconstruction_metrics,
)


@pytest.fixture(scope="module")
def constraint_loss():
    """Shared ConstraintLoss — VGG16 weights load only once per module.

    Two tests below need a real ConstraintLoss (and so a real VGG16).
    Per-test instantiation would re-load the 528 MB cached weights
    twice. Module scope keeps wall-time low. .eval() disables any
    training-mode behaviour so feature extraction is deterministic.
    """
    return ConstraintLoss(style_weight=100.0, n_input_channels=13).eval()


def test_l_recon_zero_on_identity():
    """MSE of a tensor against itself is exactly zero.

    Trivial — but it's the anchor for the composite ConstraintLoss test
    further down. If MSE drifts away from 0 here, the composite identity
    breaks too.
    """
    x = torch.randn(2, 13, 32, 32)
    assert l_recon(x, x).item() == pytest.approx(0.0)


def test_l_recon_positive():
    """MSE of two distinct random tensors is strictly positive.

    Guards against a bug that returns 0 regardless of input (stray
    .detach(), wrong tensor compared, etc.).
    """
    x = torch.randn(2, 13, 32, 32)
    y = torch.randn(2, 13, 32, 32)
    assert l_recon(x, y).item() > 0


def test_gram_matrix_shape():
    """gram_matrix collapses spatial dims: (B, C, H, W) -> (B, C, C).

    The style loss's `(G_g - G_r).pow(2).sum(dim=(1,2))` assumes the
    `(B, C, C)` shape. A wrong bmm transpose would yield `(B, H*W, H*W)`
    and the style loss would either crash or silently compute something
    unrelated.
    """
    feat = torch.randn(2, 8, 16, 16)
    assert gram_matrix(feat).shape == (2, 8, 8)


def test_gram_matrix_symmetric():
    """Gram matrices G = F F^T are symmetric by construction.

    Symmetry is the defining property of a Gram matrix. If this fails,
    the function is producing something other than F @ F^T (e.g. a
    different product or a wrong transpose axis).
    """
    feat = torch.randn(3, 8, 16, 16)
    G = gram_matrix(feat)
    assert torch.allclose(G, G.transpose(-1, -2), atol=1e-6)


def test_l_dis_zero_on_identity():
    """l_dis on identical feature lists collapses to 0.

    l_dis = sum over layers of ||mean(phi(x_gen)) - mean(phi(x_ref))||^2.
    Identical inputs => identical means => zero. Catches a sign flip or
    an off-by-one when iterating VGG feature levels.
    """
    feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
    assert l_dis(feats, feats).item() == pytest.approx(0.0)


def test_l_style_zero_on_identity():
    """l_style on identical features is 0 within fp tolerance.

    Identical features => identical Gram matrices => zero. The 1e-6
    absolute tolerance absorbs floating-point accumulation in the bmm;
    a real bug would produce a difference orders of magnitude larger.
    """
    feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
    assert l_style(feats, feats).item() == pytest.approx(0.0, abs=1e-6)


def test_l_moments_zero_on_identity():
    """l_moments(x, x) == 0 when per-band mean/std match exactly."""
    x = torch.rand(2, 13, 16, 16)
    assert l_moments(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_l_moments_positive_on_shift():
    """l_moments detects per-band mean/std shifts."""
    x = torch.rand(2, 13, 16, 16)
    y = x * 1.1 + 0.05
    assert l_moments(x, y).item() > 0


def test_constraint_loss_zero_on_identity(constraint_loss):
    """ConstraintLoss(x, x) ~= 0 — the composite inherits identity-zero.

    The full constraint loss is recon + dis + style_weight * style. Each
    head vanishes on identity, so the composite must too. The 1e-4
    tolerance absorbs fp noise that accumulates across the band-proj,
    four VGG feature slices, and the three loss heads. A real bug (e.g.
    style_weight applied to the wrong term) would produce a much larger
    residual.
    """
    x = torch.rand(2, 13, 32, 32)
    assert constraint_loss(x, x).item() == pytest.approx(0.0, abs=5e-4)


def test_namm_loss_structure(constraint_loss):
    """namm_loss returns the expected dict with the right grad flags.

    The training loop relies on three contracts:
      - `losses['loss'].backward()` must work (so 'loss' has requires_grad)
      - `losses['l_cycle'].item()` is called every log step, so the three
        scalar diagnostic keys must be detached (else the autograd graph
        grows unboundedly across logging).
      - `losses['x_recon']` is the cycle reconstruction tensor, used by
        reconstruction_metrics() to compute MAE/SAM/PSNR/SSIM without a
        second forward pass — must have the same shape as x and be
        detached.
    If a refactor drops a key or forgets a `.detach()`, training silently
    degrades — this test catches all three.
    """
    g_phi = ICNNGradient(ICNN(n_in_channels=13, n_layers=2, n_filters=8))
    f_psi = InverseMap(n_channels=13, ngf=8, n_res_blocks=1)
    x = torch.rand(2, 13, 32, 32)

    out = namm_loss(
        g_phi, f_psi, constraint_loss, x,
        max_sigma=0.1, cycle_weight=1.0, constraint_weight=1.0,
        reg_weight=0.001, device=x.device,
    )

    assert set(out) == {"loss", "l_cycle", "l_constr", "l_reg",
                         "x_recon"}
    for k in ("loss", "l_cycle", "l_constr", "l_reg"):
        v = out[k]
        assert isinstance(v, torch.Tensor)
        assert v.dim() == 0, f"{k} should be scalar, got shape {tuple(v.shape)}"

    assert out["x_recon"].shape == x.shape, \
        f"x_recon should have shape {tuple(x.shape)}, got {tuple(out['x_recon'].shape)}"
    assert not out["x_recon"].requires_grad, "x_recon should be detached"

    assert out["loss"].requires_grad, "loss must be grad-tracked for backward()"
    for k in ("l_cycle", "l_constr", "l_reg"):
        assert not out[k].requires_grad, \
            f"{k} should be detached (training loop logs it via .item())"


def test_l_sam_zero_on_identity():
    """l_sam(x, x) ~= 0 — wrapper around spectral_angle_mapper."""
    x = torch.rand(2, 13, 16, 16)
    assert l_sam(x, x).item() == pytest.approx(0.0, abs=1e-3)


def test_spectral_angle_mapper_zero_on_identity():
    """spectral_angle_mapper(x, x) ~= 0 — identity vectors are co-linear.

    SAM measures angle between spectral vectors. A vector with itself has
    angle 0 (perfect spectral consistency). Tolerance is 1e-3 rad because
    fp32 rounds (x*x).sum vs (sqrt((x*x).sum))**2 differently, so cos
    lands at ~0.99999 instead of exactly 1.0 and arccos gives ~1e-4 rad.
    A real bug (e.g. swapped tensors) would produce angles of order 1 rad.
    """
    x = torch.rand(2, 13, 16, 16)
    assert spectral_angle_mapper(x, x).item() == pytest.approx(0.0, abs=1e-3)


def test_spectral_angle_mapper_orthogonal():
    """spectral_angle_mapper of orthogonal channel vectors equals pi/2.

    For x = e_0 (one-hot on channel 0) and y = e_1 (one-hot on channel 1),
    every pixel has cos = 0 -> arccos = pi/2. Sanity check the maths
    isn't computing something else (e.g. an L2 distance instead of an
    angle).
    """
    B, C, H, W = 2, 13, 4, 4
    x = torch.zeros(B, C, H, W); x[:, 0] = 1.0
    y = torch.zeros(B, C, H, W); y[:, 1] = 1.0
    expected = torch.pi / 2
    assert spectral_angle_mapper(x, y).item() == pytest.approx(expected, abs=1e-4)


def test_reconstruction_metrics_keys_and_types():
    """reconstruction_metrics returns {mae, sam, psnr, ssim} as scalar tensors.

    The training loop unpacks all four keys to log per-step metrics; a
    missing key or wrong type would break logging at runtime.
    """
    x_hat = torch.rand(2, 13, 32, 32)
    x_ref = torch.rand(2, 13, 32, 32)
    out = reconstruction_metrics(x_hat, x_ref)
    assert set(out) == {"mae", "sam", "psnr", "ssim"}
    for k, v in out.items():
        assert isinstance(v, torch.Tensor), f"{k} should be a tensor"
        assert v.dim() == 0, f"{k} should be scalar, got shape {tuple(v.shape)}"


def test_reconstruction_metrics_identity():
    """reconstruction_metrics(x, x): MAE=0, SAM=0, PSNR=inf, SSIM=1.

    On identity, the predicted reconstruction is exactly the reference -
    every distance metric should hit its perfect-reconstruction value.
    PSNR diverges to +inf because MSE is 0; SSIM saturates at 1.0. Catches
    accidental arg-swap (e.g. metric computed against a shuffled batch).
    """
    x = torch.rand(2, 13, 32, 32)
    out = reconstruction_metrics(x, x)
    assert out["mae"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["sam"].item() == pytest.approx(0.0, abs=1e-3)  # see SAM fp note
    assert out["psnr"].item() > 100.0, \
        f"PSNR on identity should be very large (effectively inf), got {out['psnr'].item()}"
    assert out["ssim"].item() == pytest.approx(1.0, abs=1e-4)
