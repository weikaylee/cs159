"""
Input-Convex Neural Network (ICNN) for the forward mirror map g_phi.

Architecture ported from the JAX NAMM repo (berthyf96/namm) to PyTorch.
g_phi is parameterised as the gradient of a convex function, which guarantees
that g_phi is a valid (monotone) map.  We implement the ICNN itself and expose
its gradient via autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNNLayer(nn.Module):
    """Single ICNN layer.

    Passes the input x through two paths:
      - A convex path from the previous hidden state z (weights kept non-negative).
      - A free path directly from the original input x (unconstrained weights).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 x_channels: int, kernel_size: int = 3,
                 strong_convexity: float = 0.3):
        super().__init__()
        pad = kernel_size // 2

        # Convex path: z -> z_next (weights must be non-negative at eval time)
        self.Wz = nn.Conv2d(in_channels, hidden_channels, kernel_size,
                            padding=pad, bias=False)

        # Shortcut path: x -> z_next (unconstrained)
        self.Wx = nn.Conv2d(x_channels, hidden_channels, kernel_size,
                            padding=pad, bias=True)

        self.strong_convexity = strong_convexity
        nn.init.kaiming_uniform_(self.Wz.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.Wx.weight, nonlinearity='relu')

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Clamp Wz weights to be non-negative (convexity constraint)
        Wz_pos = F.relu(self.Wz.weight)
        z_conv = F.conv2d(z, Wz_pos, bias=None,
                          padding=self.Wz.padding[0])
        x_proj = self.Wx(x)
        out = F.leaky_relu(z_conv + x_proj, negative_slope=0.2)
        # Strong-convexity regularisation: add alpha/2 * ||x||^2 implicitly
        # via the shortcut path — handled in the loss, not here.
        return out


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
        negative_slope:   Leaky-ReLU slope (must be ≥ 0 to preserve convexity).
    """

    def __init__(self, n_in_channels: int = 13, n_layers: int = 6,
                 n_filters: int = 64, kernel_size: int = 3,
                 strong_convexity: float = 0.3, negative_slope: float = 0.2):
        super().__init__()
        assert negative_slope >= 0, "Negative slope must be >= 0 to preserve convexity."
        self.strong_convexity = strong_convexity
        pad = kernel_size // 2

        # First layer: x -> z_1 (no convex path yet, just a free projection)
        self.input_layer = nn.Conv2d(n_in_channels, n_filters, kernel_size,
                                     padding=pad, bias=True)

        # Hidden ICNN layers
        self.layers = nn.ModuleList([
            ICNNLayer(n_filters, n_filters, n_in_channels, kernel_size,
                      strong_convexity)
            for _ in range(n_layers - 1)
        ])

        # Final layer: z -> scalar potential (1 output channel, then mean)
        self.Wz_final = nn.Conv2d(n_filters, 1, kernel_size,
                                  padding=pad, bias=False)
        self.Wx_final = nn.Conv2d(n_in_channels, 1, kernel_size,
                                  padding=pad, bias=True)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the scalar convex potential Phi(x). Shape: (B,)"""
        z = F.leaky_relu(self.input_layer(x), negative_slope=0.2)
        for layer in self.layers:
            z = layer(z, x)

        # Final aggregation to scalar per sample
        Wz_pos = F.relu(self.Wz_final.weight)
        z_out = F.conv2d(z, Wz_pos, bias=None,
                         padding=self.Wz_final.padding[0])
        x_out = self.Wx_final(x)
        phi = (z_out + x_out).mean(dim=(1, 2, 3))  # (B,)

        # Strong-convexity term: alpha/2 * ||x||^2
        x_norm = (x ** 2).mean(dim=(1, 2, 3))
        phi = phi + (self.strong_convexity / 2.0) * x_norm
        return phi


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
        x = x.requires_grad_(True)
        phi = self.icnn.potential(x).sum()
        g_x = torch.autograd.grad(phi, x, create_graph=self.training)[0]
        return g_x
