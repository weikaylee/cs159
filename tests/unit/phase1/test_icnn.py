"""Unit tests for phase1_mirror_map/icnn.py.

The ICNN family implements the forward mirror map g_phi = grad_x Phi(x).
The correctness of every downstream Phase 1 quantity (cycle loss,
mirror-space EMRDM target, ICNN regulariser) depends on three invariants:

  1. SHAPE — gradients of a per-pixel field must come out the same shape
     as the input. Three tests cover this at three nesting levels.
  2. DIFFERENTIABILITY UNDER no_grad — validation calls ICNNGradient
     inside a torch.no_grad() context. The recent enable_grad fix in
     icnn.py:135 makes this work; test_icnn_gradient_under_no_grad
     locks it in.
  3. CONVEXITY of the scalar potential — required for g_phi to be a
     monotone (invertible) map. Checked numerically by
     test_icnn_convexity for several interpolation parameters.

All tests run on CPU with deliberately small (n_layers=2-3, n_filters=8-16)
networks. Instances are constructed inline because they're cheap.
"""

import torch
import pytest

from icnn import ICNN, ICNNLayer, ICNNGradient


def test_icnn_potential_shape():
    """ICNN.potential reduces a 4-D input (B, C, H, W) to a 1-D (B,).

    Phi is a *scalar* convex potential per sample — its gradient is
    what becomes the mirror map. If a future refactor of `potential`
    forgets the final `.mean(dim=(1,2,3))`, this test catches the
    shape mismatch before training silently broadcasts.
    """
    model = ICNN(n_in_channels=13, n_layers=3, n_filters=16)
    x = torch.randn(4, 13, 32, 32)
    assert model.potential(x).shape == (4,)


def test_icnn_layer_shape():
    """ICNNLayer(z, x) -> (B, hidden_channels, H, W).

    Spatial dims must be preserved (kernel=3, padding=1). Output channel
    count must equal `hidden_channels`. Catches a misconfigured Conv2d
    on either the convex (z) or shortcut (x) path.
    """
    layer = ICNNLayer(in_channels=16, hidden_channels=8, x_channels=13)
    z = torch.randn(2, 16, 32, 32)
    x = torch.randn(2, 13, 32, 32)
    assert layer(z, x).shape == (2, 8, 32, 32)


def test_icnn_gradient_shape():
    """ICNNGradient(x).shape == x.shape — the mirror map is a per-pixel field.

    `torch.autograd.grad(phi, x)` returns a tensor matching the input;
    this test guards against accidental reduction (e.g. someone summing
    the gradient before return).
    """
    g_phi = ICNNGradient(ICNN(n_in_channels=13, n_layers=2, n_filters=8))
    g_phi.eval()
    x = torch.randn(2, 13, 32, 32)
    assert g_phi(x).shape == x.shape


def test_icnn_gradient_under_no_grad():
    """Regression: ICNNGradient must work inside `torch.no_grad()`.

    The Phase 1 `validate()` function in train_mirror_map.py is decorated
    with @torch.no_grad(), so ICNNGradient gets called with autograd
    globally disabled. `requires_grad_(True)` alone does NOT bypass
    no_grad — only the `with torch.enable_grad():` block inside
    icnn.py:133-137 does. If someone removes that wrapper, validation
    crashes with "element 0 does not require grad and does not have a
    grad_fn". This test will fail loudly first.
    """
    g_phi = ICNNGradient(ICNN(n_in_channels=13, n_layers=2, n_filters=8))
    g_phi.eval()
    x = torch.randn(2, 13, 32, 32)
    with torch.no_grad():
        g_x = g_phi(x)
    assert g_x.shape == x.shape


def test_icnn_convexity():
    """ICNN.potential is convex in x: phi(tx + (1-t)y) <= t phi(x) + (1-t) phi(y).

    Convexity is the load-bearing mathematical property of ICNN. It guarantees
    that grad Phi is a monotone (hence invertible) map between constrained
    and mirror spaces — without it, the NAMM framework breaks. The check is
    numerical: random x, y, and five interpolation parameters t. Tolerance
    1e-4 absorbs fp noise but catches any qualitative violation (e.g. a
    leaky_relu with negative slope, a missing F.relu(Wz.weight) clamp).
    """
    torch.manual_seed(0)
    model = ICNN(n_in_channels=13, n_layers=3, n_filters=16,
                 strong_convexity=0.5)
    model.eval()
    x = torch.randn(8, 13, 16, 16)
    y = torch.randn(8, 13, 16, 16)
    phi_x = model.potential(x)
    phi_y = model.potential(y)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        phi_mid = model.potential(t * x + (1 - t) * y)
        phi_combo = t * phi_x + (1 - t) * phi_y
        assert (phi_mid <= phi_combo + 1e-4).all(), (
            f"convexity violated at t={t}: "
            f"max excess {(phi_mid - phi_combo).max().item():.2e}"
        )
