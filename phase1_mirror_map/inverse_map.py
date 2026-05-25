"""
ResNet-based inverse mirror map f_psi.

Ported from the JAX NAMM Generator architecture (berthyf96/namm model.py).
f_psi maps unconstrained mirror-space images back to the constrained
(spectrally consistent) space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.GroupNorm(1, features),   # instance norm (group_size=1)
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.GroupNorm(1, features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class InverseMap(nn.Module):
    """Inverse mirror map f_psi.

    Encoder-decoder (ResNet generator) that maps unconstrained mirror-space
    images back to the constrained optical reflectance space.

    Args:
        n_channels:          Number of image channels (13 for Sentinel-2).
        ngf:                 Base number of feature maps.
        n_res_blocks:        Number of ResNet transformation blocks.
        n_downsample_layers: Number of encoder/decoder stages.
        dropout_rate:        Dropout rate inside ResNet blocks.
        residual:            If True, add a skip connection from input to output.
    """

    def __init__(self, n_channels: int = 13, ngf: int = 64,
                 n_res_blocks: int = 6, n_downsample_layers: int = 2,
                 dropout_rate: float = 0.5, residual: bool = True):
        super().__init__()
        self.residual = residual

        # ── Encoder ──────────────────────────────────────────────────────────
        encoder = [
            nn.Conv2d(n_channels, ngf, 7, padding=3, bias=False),
            nn.GroupNorm(1, ngf),
            nn.ReLU(inplace=True),
        ]
        for i in range(n_downsample_layers):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2,
                          padding=1, bias=False),
                nn.GroupNorm(1, ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*encoder)

        # ── Bottleneck ────────────────────────────────────────────────────────
        bottleneck_features = ngf * (2 ** n_downsample_layers)
        self.bottleneck = nn.Sequential(*[
            ResNetBlock(bottleneck_features, dropout_rate)
            for _ in range(n_res_blocks)
        ])

        # ── Decoder ──────────────────────────────────────────────────────────
        decoder = []
        for i in range(n_downsample_layers):
            mult = 2 ** (n_downsample_layers - i)
            decoder += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.GroupNorm(1, ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        decoder += [
            nn.Conv2d(ngf, n_channels, 7, padding=3),
        ]
        self.decoder = nn.Sequential(*decoder)
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, C, H, W) image in unconstrained mirror space.
        Returns:
            x_hat: (B, C, H, W) image in constrained space.
        """
        z = self.encoder(y)
        z = self.bottleneck(z)
        x_hat = self.decoder(z)
        if self.residual:
            x_hat = x_hat + y
        if self.final_activation is not None:
            x_hat = self.final_activation(x_hat)
        return x_hat
