"""Unit tests for phase1_mirror_map/inverse_map.py.

The inverse mirror map f_psi maps mirror-space tensors (unconstrained, can
be negative) back to constrained reflectance space (non-negative). Two
invariants matter:

  1. SHAPE: encoder-decoder symmetry. f_psi(y) must have y's shape so it
     can be composed with g_phi for the cycle loss.
  2. NON-NEGATIVITY when residual=False: the final ReLU enforces that
     reflectances stay physical. With residual=True the property doesn't
     hold (y can be negative) and we only check shape.

ResNetBlock's shape-identity is tested separately because it's the unit
of repetition inside the bottleneck.
"""

import torch

from inverse_map import ResNetBlock, InverseMap


def test_resnet_block_shape():
    """ResNetBlock preserves shape exactly — required for the residual sum.

    The block is `x + block(x)` where `block` is conv -> norm -> relu ->
    dropout -> conv -> norm. For the addition to work at all, block(x)
    must have the same shape as x. .eval() disables Dropout2d so the
    forward pass is deterministic.
    """
    block = ResNetBlock(features=16, dropout_rate=0.0).eval()
    x = torch.randn(2, 16, 32, 32)
    assert block(x).shape == x.shape


def test_inverse_map_shape():
    """InverseMap is shape-identity: (B, C, H, W) -> (B, C, H, W).

    The encoder downsamples 2x per stage (n_downsample_layers=2 by default
    -> 4x total), and the decoder upsamples symmetrically via
    ConvTranspose2d with output_padding=1. An off-by-one there would
    show up as a shape mismatch; this test catches it.
    """
    model = InverseMap(n_channels=13, ngf=8, n_res_blocks=1).eval()
    y = torch.randn(2, 13, 32, 32)
    assert model(y).shape == y.shape


def test_inverse_map_residual_false_nonneg():
    """With residual=False, output is non-negative (final ReLU).

    Sentinel-2 surface reflectance is physically >= 0. The decoder's last
    layer is Conv2d -> ReLU(inplace), which enforces that constraint on
    the pure decoder output. If a future refactor removes that ReLU,
    f_psi could emit negative "reflectance" that downstream losses would
    process incorrectly.
    """
    model = InverseMap(n_channels=13, ngf=8, n_res_blocks=1,
                       residual=False).eval()
    y = torch.randn(2, 13, 32, 32)
    assert (model(y) >= 0).all()


def test_inverse_map_residual_true_keeps_shape():
    """With residual=True, shape is preserved even for negative-valued y.

    Real mirror-space tensors have negative entries; f_psi adds y (the
    residual) to the decoded output, so the result can go negative. We
    only assert shape here — non-negativity is impossible when y is
    unbounded. A failure would mean the residual addition isn't wired
    correctly.
    """
    model = InverseMap(n_channels=13, ngf=8, n_res_blocks=1,
                       residual=True).eval()
    y = torch.randn(2, 13, 32, 32) * 5  # mirror-space-ish: both signs, large
    assert model(y).shape == y.shape
