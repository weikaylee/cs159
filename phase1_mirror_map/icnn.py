"""
Input-Convex Neural Network (ICNN) for the forward mirror map g_phi.

Architecture ported from the JAX NAMM repo (berthyf96/namm) to PyTorch.
g_phi is parameterised as the gradient of a convex function, which guarantees
that g_phi is a valid (monotone) map.  We implement the ICNN itself and expose
its gradient via autograd.

Fixes applied vs. the original:
  1. Post-step weight clipping: ICNNLayer and ICNN both expose clip_weights(),
     which must be called after every optimizer.step(). This ensures Wz weights
     remain non-negative throughout training, not just during the forward pass.
  2. Removed dead `strong_convexity` parameter from ICNNLayer — it was stored
     but never used there. Strong convexity is handled solely in ICNN.potential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNNLayer(nn.Module):
    """Single ICNN layer.

    Passes the input x through two paths:
      - A convex path from the previous hidden state z (weights kept non-negative).
      - A free path directly from the original input x (unconstrained weights).

    Fix 1: clip_weights() added. The forward pass applies F.relu(Wz.weight)
    inline as a safety net, but this alone is insufficient: the stored weights
    can drift arbitrarily negative over training because gradients flow through
    the stored Wz.weight, not through the relu-projected copy. Over many steps
    the stored weights become large and negative, the relu clips most of them
    to zero, and the z-path effectively dies. Calling clip_weights() after every
    optimizer.step() keeps the stored weights non-negative, so the inline relu
    and the stored weights stay consistent.

    Fix 2: strong_convexity parameter removed. It was accepted in __init__ and
    stored as self.strong_convexity but never used anywhere in this class.
    Strong convexity is handled in ICNN.potential via the (alpha/2)||x||^2 term.
    Keeping it here was misleading and cluttered the interface.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 x_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2

        # Convex path: z -> z_next (weights must be non-negative at eval time).
        self.Wz = nn.Conv2d(in_channels, hidden_channels, kernel_size,
                            padding=pad, bias=False)

        # Shortcut path: x -> z_next (unconstrained).
        self.Wx = nn.Conv2d(x_channels, hidden_channels, kernel_size,
                            padding=pad, bias=True)

        nn.init.kaiming_uniform_(self.Wz.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.Wx.weight, nonlinearity='relu')

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Inline relu on Wz.weight: safety net for the forward pass.
        # This ensures convexity holds even if clip_weights() was somehow
        # skipped, but it is not a substitute for clip_weights() — see above.
        Wz_pos = F.relu(self.Wz.weight)
        z_conv = F.conv2d(z, Wz_pos, bias=None,
                          padding=self.Wz.padding[0])
        x_proj = self.Wx(x)
        return F.leaky_relu(z_conv + x_proj, negative_slope=0.2)

    def clip_weights(self):
        """Clamp Wz weights to non-negative. Call after every optimizer.step()."""
        self.Wz.weight.data.clamp_(min=0)


class ICNN(nn.Module):
    """Input-Convex Neural Network.

    Computes a scalar convex potential Phi(x).  The forward mirror map
    g_phi(x) = grad_x Phi(x) is obtained via autograd (see ICNNGradient).

    Args:
        n_in_channels:    Number of input image channels (13 for Sentinel-2).
        n_layers:         Number of hidden ICNN layers.
        n_filters:        Number of feature channels per hidden layer.
        kernel_size:      Spatial kernel size for all convolutions.
        strong_convexity: Coefficient alpha for strong-convexity regularisation.
                          Adds alpha/2 * ||x||^2 to the potential, ensuring
                          the gradient map is injective.
        negative_slope:   Leaky-ReLU slope (must be >= 0 to preserve convexity).
    """

    def __init__(self, n_in_channels: int = 13, n_layers: int = 6,
                 n_filters: int = 64, kernel_size: int = 3,
                 strong_convexity: float = 0.3, negative_slope: float = 0.2):
        super().__init__()
        assert negative_slope >= 0, "Negative slope must be >= 0 to preserve convexity."
        self.strong_convexity = strong_convexity
        pad = kernel_size // 2

        # First layer: x -> z_1 (no convex path yet, just a free projection).
        self.input_layer = nn.Conv2d(n_in_channels, n_filters, kernel_size,
                                     padding=pad, bias=True)

        # Hidden ICNN layers.
        # Fix 2: strong_convexity no longer passed to ICNNLayer (removed there).
        self.layers = nn.ModuleList([
            ICNNLayer(n_filters, n_filters, n_in_channels, kernel_size)
            for _ in range(n_layers - 1)
        ])

        # Final layer: z -> scalar potential (1 output channel, then mean).
        self.Wz_final = nn.Conv2d(n_filters, 1, kernel_size,
                                  padding=pad, bias=False)
        self.Wx_final = nn.Conv2d(n_in_channels, 1, kernel_size,
                                  padding=pad, bias=True)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the scalar convex potential Phi(x). Shape: (B,)"""
        z = F.leaky_relu(self.input_layer(x), negative_slope=0.2)
        for layer in self.layers:
            z = layer(z, x)

        # Final aggregation to scalar per sample.
        Wz_pos = F.relu(self.Wz_final.weight)
        z_out = F.conv2d(z, Wz_pos, bias=None,
                         padding=self.Wz_final.padding[0])
        x_out = self.Wx_final(x)
        phi = (z_out + x_out).mean(dim=(1, 2, 3))  # (B,)

        # Strong-convexity term: alpha/2 * ||x||^2.
        x_norm = (x ** 2).mean(dim=(1, 2, 3))
        phi = phi + (self.strong_convexity / 2.0) * x_norm
        return phi

    def clip_weights(self):
        """Clamp all Wz weights to non-negative. Call after every optimizer.step().

        Fix 1: this method propagates clipping to every ICNNLayer and to the
        final Wz_final layer. A single call from the training loop is sufficient:

            optimizer.step()
            icnn.clip_weights()  # <-- add this line after every step

        Wz_final is also clamped here for the same reason as ICNNLayer.Wz:
        the inline F.relu(Wz_final.weight) in potential() protects the forward
        pass but not the stored weights.
        """
        for layer in self.layers:
            layer.clip_weights()
        self.Wz_final.weight.data.clamp_(min=0)


class ICNNGradient(nn.Module):
    """Forward mirror map g_phi(x) = grad_x Phi(x).

    Wraps an ICNN and returns its gradient with respect to the input.
    This is a valid (monotone) map from the constrained space to the
    unconstrained mirror space.
    """

    def __init__(self, icnn: ICNN):
        super().__init__()
        self.icnn = icnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) cloud-free Sentinel-2 image in constrained space.
        Returns:
            g_x: (B, C, H, W) image in unconstrained mirror space.
        """
        # enable_grad: requires_grad_(True) does not bypass an outer
        # torch.no_grad() context — without this, autograd.grad fails because
        # phi has no grad_fn. Callers (e.g. validate()) may freely use no_grad.
        with torch.enable_grad():
            x = x.requires_grad_(True)
            phi = self.icnn.potential(x).sum()
            g_x = torch.autograd.grad(phi, x, create_graph=self.training)[0]
        return g_x

    def clip_weights(self):
        """Convenience passthrough so the training loop only needs one call."""
        self.icnn.clip_weights()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    B, C, H, W = 4, 13, 64, 64

    icnn = ICNN(n_in_channels=C, n_layers=3, n_filters=64,
                strong_convexity=0.9).to(device)
    g_phi = ICNNGradient(icnn).to(device)

    x = torch.randn(B, C, H, W, device=device)

    # --- forward pass ---
    g_x = g_phi(x)
    assert g_x.shape == x.shape, f"Shape mismatch: {g_x.shape} vs {x.shape}"
    print(f"Output shape: {g_x.shape} ✓")

    # --- backward pass through g_phi ---
    loss = g_x.mean()
    loss.backward()
    print("Backward pass through g_phi ✓")

    # --- simulate optimizer step + clip ---
    opt = torch.optim.Adam(g_phi.parameters(), lr=1e-3)
    opt.step()
    g_phi.clip_weights()  # Fix 1: must be called after every optimizer.step()

    # --- verify Wz weights are non-negative after clipping ---
    for i, layer in enumerate(g_phi.icnn.layers):
        min_w = layer.Wz.weight.data.min().item()
        assert min_w >= 0, f"layers[{i}].Wz has negative weights: {min_w:.4f}"
    min_final = g_phi.icnn.Wz_final.weight.data.min().item()
    assert min_final >= 0, f"Wz_final has negative weights: {min_final:.4f}"
    print("All Wz weights non-negative after clip_weights() ✓")

    print("Smoke test passed.")