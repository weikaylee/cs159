"""
PDE-Constrained Diffusion Model
================================
Forward process  : structured noise via Navier-Stokes (fluid) + Maxwell (EM) PDEs
Reverse process  : score-matching denoiser conditioned on PDE residuals
Training         : DDPM loss + NS residual loss + Maxwell residual loss + divergence penalty

Spatial domain  : 2-D periodic grid (H × W)
Field layout    : [u, v, p, Ex, Ey, Bz]  ← velocity (u,v), pressure p, E-field, B-field
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class DiffusionConfig:
    # Grid
    H: int = 64
    W: int = 64
    C: int = 6           # channels: [u, v, p, Ex, Ey, Bz]

    # Diffusion schedule
    T: int = 1000
    beta_start: float = 1e-4
    beta_end:   float = 0.02
    schedule:   str   = "cosine"   # "linear" | "cosine"

    # PDE parameters
    nu:  float = 1e-3   # kinematic viscosity  (NS)
    eps: float = 1.0    # permittivity         (EM)
    mu:  float = 1.0    # permeability         (EM)
    dt:  float = 1e-2   # PDE integration step

    # Loss weights
    lambda_ns:  float = 0.1
    lambda_em:  float = 0.1
    lambda_div: float = 0.05

    # Score network
    base_channels: int = 64
    channel_mults: list = field(default_factory=lambda: [1, 2, 4, 8])
    num_res_blocks: int = 2
    attention_res:  list = field(default_factory=lambda: [16, 8])

    # Training
    batch_size: int = 8
    lr:         float = 1e-4
    num_steps:  int = 100_000
    grad_clip:  float = 1.0
    device:     str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
#  Noise Schedule
# ─────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Improved cosine schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DiffusionSchedule(nn.Module):
    """Pre-computes all schedule quantities on the given device."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        T = cfg.T
        if cfg.schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, cfg.beta_start, cfg.beta_end)

        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",                betas)
        self.register_buffer("alphas_cumprod",       alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",  alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",  alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas",    alphas.rsqrt())
        self.register_buffer("posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε."""
        s_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0)
        s_one   = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return s_alpha * x0 + s_one * noise

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(B, *([1] * (x.ndim - 1)))


# ─────────────────────────────────────────────
#  PDE Operators  (finite-difference, periodic BC)
# ─────────────────────────────────────────────

def _ddx(f: torch.Tensor) -> torch.Tensor:
    """Central-difference ∂/∂x on last dimension (periodic)."""
    return (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) * 0.5

def _ddy(f: torch.Tensor) -> torch.Tensor:
    """Central-difference ∂/∂y on second-to-last dimension (periodic)."""
    return (torch.roll(f, -1, -2) - torch.roll(f, 1, -2)) * 0.5

def _laplacian(f: torch.Tensor) -> torch.Tensor:
    """Five-point Laplacian (periodic)."""
    return (torch.roll(f, 1, -1) + torch.roll(f, -1, -1) +
            torch.roll(f, 1, -2) + torch.roll(f, -1, -2) - 4.0 * f)


class NavierStokesKernel(nn.Module):
    """
    Incompressible 2-D Navier-Stokes PDE noise kernel.

    Channels assumed:  x[:, 0] = u (x-velocity)
                       x[:, 1] = v (y-velocity)
                       x[:, 2] = p (pressure)

    One explicit Euler step:
        u* = u − dt[(u·∂u/∂x + v·∂u/∂y) − ν∇²u + ∂p/∂x] + √βₜ · fᵤ
        v* = v − dt[(u·∂v/∂x + v·∂v/∂y) − ν∇²v + ∂p/∂y] + √βₜ · f_v
    Pressure is updated via pseudo-spectral projection to enforce ∇·u = 0.
    """

    def __init__(self, nu: float, dt: float):
        super().__init__()
        self.nu = nu
        self.dt = dt

    def forward(self, x: torch.Tensor, beta_t: torch.Tensor,
                rng: Optional[torch.Generator] = None) -> torch.Tensor:
        u, v, p = x[:, 0], x[:, 1], x[:, 2]
        nu, dt  = self.nu, self.dt

        # Advection
        adv_u = u * _ddx(u) + v * _ddy(u)
        adv_v = u * _ddx(v) + v * _ddy(v)

        # Diffusion
        diff_u = nu * _laplacian(u)
        diff_v = nu * _laplacian(v)

        # Pressure gradient
        dp_dx, dp_dy = _ddx(p), _ddy(p)

        # Structured stochastic forcing ~ Matérn-like (via repeated Laplacian smoothing)
        sqrt_beta = beta_t.sqrt().reshape(-1, 1, 1)
        f_u = torch.randn_like(u, generator=rng)
        f_v = torch.randn_like(v, generator=rng)
        for _ in range(3):       # smooth → divergence-reduced forcing
            f_u = f_u + 0.1 * _laplacian(f_u)
            f_v = f_v + 0.1 * _laplacian(f_v)
        f_u = F.normalize(f_u.flatten(1), dim=1).reshape_as(u)
        f_v = F.normalize(f_v.flatten(1), dim=1).reshape_as(v)

        # Explicit Euler step
        u_new = u + dt * (-adv_u + diff_u - dp_dx) + sqrt_beta * f_u
        v_new = v + dt * (-adv_v + diff_v - dp_dy) + sqrt_beta * f_v

        # Pressure correction: project to divergence-free via spectral Poisson solve
        div = _ddx(u_new) + _ddy(v_new)
        p_corr = self._poisson_fft(div)          # ∇²p_corr = ∇·u*
        u_new = u_new - _ddx(p_corr)
        v_new = v_new - _ddy(p_corr)
        p_new = p + p_corr

        out = x.clone()
        out[:, 0], out[:, 1], out[:, 2] = u_new, v_new, p_new
        return out

    @staticmethod
    def _poisson_fft(rhs: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver on periodic domain: ∇²φ = rhs  →  φ via FFT."""
        B, H, W = rhs.shape
        rhs_f = torch.fft.rfft2(rhs)
        ky = torch.fft.fftfreq(H, d=1.0/H, device=rhs.device).reshape(H, 1)
        kx = torch.fft.rfftfreq(W, d=1.0/W, device=rhs.device).reshape(1, W // 2 + 1)
        k2 = (ky**2 + kx**2).clamp(min=1e-10)
        k2[0, 0] = 1.0                           # avoid DC division by zero
        phi_f = rhs_f / (-4.0 * math.pi**2 * k2)
        phi_f[:, 0, 0] = 0.0                     # zero mean
        return torch.fft.irfft2(phi_f, s=(H, W))


class MaxwellKernel(nn.Module):
    """
    2-D transverse-magnetic (TM) Maxwell noise kernel.

    Channels assumed:  x[:, 3] = Ex
                       x[:, 4] = Ey
                       x[:, 5] = Bz

    Yee-scheme half-step:
        ∂Bz/∂t = −(∂Ey/∂x − ∂Ex/∂y)          (Faraday)
        ∂Ex/∂t =  (1/ε) ∂Bz/∂y  − Jx/ε       (Ampère)
        ∂Ey/∂t = −(1/ε) ∂Bz/∂x  − Jy/ε
    Random current J injects stochastic EM energy (curl-free component removed).
    """

    def __init__(self, eps: float, mu: float, dt: float):
        super().__init__()
        self.eps, self.mu, self.dt = eps, mu, dt

    def forward(self, x: torch.Tensor, beta_t: torch.Tensor,
                rng: Optional[torch.Generator] = None) -> torch.Tensor:
        Ex, Ey, Bz = x[:, 3], x[:, 4], x[:, 5]
        eps, mu, dt = self.eps, self.mu, self.dt

        sqrt_beta = beta_t.sqrt().reshape(-1, 1, 1)

        # Stochastic current (Helmholtz: remove curl-free part → divergence-free J)
        Jx = torch.randn_like(Ex, generator=rng) * sqrt_beta
        Jy = torch.randn_like(Ey, generator=rng) * sqrt_beta
        Jx, Jy = self._project_divergence_free(Jx, Jy)

        # Faraday: ∂Bz/∂t = −(∂Ey/∂x − ∂Ex/∂y)
        curl_E = _ddx(Ey) - _ddy(Ex)
        Bz_new = Bz - dt / mu * curl_E

        # Ampère: ∂E/∂t = (1/ε)(∇×B) − J/ε
        Ex_new = Ex + dt / eps * _ddy(Bz_new) - dt / eps * Jx
        Ey_new = Ey - dt / eps * _ddx(Bz_new) - dt / eps * Jy

        out = x.clone()
        out[:, 3], out[:, 4], out[:, 5] = Ex_new, Ey_new, Bz_new
        return out

    @staticmethod
    def _project_divergence_free(Jx: torch.Tensor, Jy: torch.Tensor):
        """Remove irrotational component via spectral Helmholtz decomposition."""
        B, H, W = Jx.shape
        Jxf = torch.fft.rfft2(Jx)
        Jyf = torch.fft.rfft2(Jy)
        ky = torch.fft.fftfreq(H, d=1.0/H, device=Jx.device).reshape(H, 1)
        kx = torch.fft.rfftfreq(W, d=1.0/W, device=Jx.device).reshape(1, W // 2 + 1)
        k2 = (kx**2 + ky**2).clamp(min=1e-10)
        # div(J) in Fourier: kx*Jx + ky*Jy
        div_f = kx * Jxf + ky * Jyf
        # subtract gradient of scalar potential
        Jxf = Jxf - kx * div_f / k2
        Jyf = Jyf - ky * div_f / k2
        return torch.fft.irfft2(Jxf, s=(H, W)), torch.fft.irfft2(Jyf, s=(H, W))


# ─────────────────────────────────────────────
#  PDE Residual Encoder  (conditioning signal c)
# ─────────────────────────────────────────────

class PDEResidualEncoder(nn.Module):
    """
    Computes NS and EM residuals on a predicted x̂₀ and encodes them
    into a conditioning vector c used by the score network.
    """

    def __init__(self, C: int, out_dim: int, nu: float, eps: float, mu: float):
        super().__init__()
        self.nu, self.eps, self.mu = nu, eps, mu
        # project residual channels to latent
        residual_C = 4   # [∂u/∂t-residual, ∂v/∂t-residual, ∂Ex/∂t-residual, ∂Bz/∂t-residual]
        self.proj = nn.Sequential(
            nn.Conv2d(residual_C, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, out_dim),
        )

    def _ns_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Steady-state NS residual (momentum imbalance per channel)."""
        u, v = x[:, 0], x[:, 1]
        res_u = u * _ddx(u) + v * _ddy(u) - self.nu * _laplacian(u) + _ddx(x[:, 2])
        res_v = u * _ddx(v) + v * _ddy(v) - self.nu * _laplacian(v) + _ddy(x[:, 2])
        return torch.stack([res_u, res_v], dim=1)

    def _em_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Faraday + Ampère residuals."""
        Ex, Ey, Bz = x[:, 3], x[:, 4], x[:, 5]
        faraday = _ddx(Ey) - _ddy(Ex)                             # should ≈ 0 at equilibrium
        ampere_x = _ddy(Bz) / self.eps - Ex * 0.0                # simplified: J=0 baseline
        return torch.stack([faraday, ampere_x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r_ns = self._ns_residual(x)   # (B, 2, H, W)
        r_em = self._em_residual(x)   # (B, 2, H, W)
        residuals = torch.cat([r_ns, r_em], dim=1)   # (B, 4, H, W)
        return self.proj(residuals)   # (B, out_dim)


# ─────────────────────────────────────────────
#  Score Network  (U-Net with PDE conditioning)
# ─────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                  nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return self.proj(emb)


class ResBlock(nn.Module):
    """Residual block with AdaLN conditioning from (time_emb + pde_emb)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.adaLN = nn.Linear(cond_dim, out_ch * 2)   # scale + shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.adaLN(cond).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h * (1 + scale) + shift     # AdaLN
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm  = nn.GroupNorm(8, ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(q.shape[-2])
        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class PDEScoreUNet(nn.Module):
    """
    U-Net score network conditioned on (t, PDE residuals).

    sθ(xₜ, t, c) ≈ ∇_{xₜ} log q(xₜ | x₀)
    Parameterised as noise predictor ε̂θ.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        C   = cfg.C
        ch  = cfg.base_channels
        mults = cfg.channel_mults
        chs   = [ch * m for m in mults]
        pde_out_dim = 256

        # Time + PDE conditioning
        self.time_emb  = SinusoidalTimeEmbed(ch)
        self.pde_enc   = PDEResidualEncoder(C, pde_out_dim,
                                             cfg.nu, cfg.eps, cfg.mu)
        self.pde_proj  = nn.Linear(pde_out_dim, ch)
        cond_dim = ch + ch      # time + pde concatenated

        # Input projection
        self.in_proj = nn.Conv2d(C, ch, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs      = nn.ModuleList()
        in_ch = ch
        for i, out_ch in enumerate(chs):
            blocks = nn.ModuleList([
                ResBlock(in_ch if j == 0 else out_ch, out_ch, cond_dim)
                for j in range(cfg.num_res_blocks)
            ])
            attn = SelfAttention2D(out_ch) if (cfg.H >> i) in cfg.attention_res else nn.Identity()
            self.enc_blocks.append(nn.ModuleList([blocks, attn]))
            self.downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                              if i < len(chs) - 1 else nn.Identity())
            in_ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock(in_ch, in_ch, cond_dim)
        self.mid_attn   = SelfAttention2D(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, cond_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups        = nn.ModuleList()
        for i, out_ch in enumerate(reversed(chs)):
            skip_ch = chs[-(i + 1)]
            blocks = nn.ModuleList([
                ResBlock(in_ch + skip_ch if j == 0 else out_ch, out_ch, cond_dim)
                for j in range(cfg.num_res_blocks)
            ])
            attn = SelfAttention2D(out_ch) if (cfg.H >> (len(chs) - 1 - i)) in cfg.attention_res else nn.Identity()
            self.dec_blocks.append(nn.ModuleList([blocks, attn]))
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
                            if i < len(chs) - 1 else nn.Identity())
            in_ch = out_ch

        # Output projection
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_proj = nn.Conv2d(ch, C, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                x_for_pde: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : (B, C, H, W) — noised field xₜ
        t          : (B,)         — diffusion timestep
        x_for_pde  : (B, C, H, W) — field for PDE residual (usually xₜ or x̂₀)
        Returns    : ε̂θ(xₜ, t, c)  same shape as x
        """
        if x_for_pde is None:
            x_for_pde = x

        # Conditioning
        t_emb   = self.time_emb(t)               # (B, ch)
        pde_emb = self.pde_proj(self.pde_enc(x_for_pde))  # (B, ch)
        cond    = torch.cat([t_emb, pde_emb], dim=1)       # (B, cond_dim)

        # Encoder
        h = self.in_proj(x)
        skips = []
        for (blocks, attn), down in zip(self.enc_blocks, self.downs):
            for blk in blocks:
                h = blk(h, cond)
            h = attn(h) if isinstance(attn, SelfAttention2D) else h
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder
        for (blocks, attn), up in zip(self.dec_blocks, self.ups):
            h = up(h)
            s = skips.pop()
            if h.shape != s.shape:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], dim=1)
            for blk in blocks:
                h = blk(h, cond)
            h = attn(h) if isinstance(attn, SelfAttention2D) else h

        return self.out_proj(F.silu(self.out_norm(h)))


# ─────────────────────────────────────────────
#  PDE Residual Losses  (physics constraints on x̂₀)
# ─────────────────────────────────────────────

def ns_residual_loss(x0_hat: torch.Tensor, nu: float) -> torch.Tensor:
    """
    Penalise momentum imbalance in the predicted clean field.
    ℒ_NS = ‖u·∇u − ν∇²u + ∇p‖² + ‖∇·u‖²
    """
    u, v, p = x0_hat[:, 0], x0_hat[:, 1], x0_hat[:, 2]
    res_u = u * _ddx(u) + v * _ddy(u) - nu * _laplacian(u) + _ddx(p)
    res_v = u * _ddx(v) + v * _ddy(v) - nu * _laplacian(v) + _ddy(p)
    div   = _ddx(u) + _ddy(v)
    return (res_u**2 + res_v**2 + div**2).mean()


def maxwell_residual_loss(x0_hat: torch.Tensor, eps: float, mu: float) -> torch.Tensor:
    """
    Penalise steady-state EM residuals and Gauss's law.
    ℒ_EM = ‖∇×E‖² + ‖∇·E‖² (source-free)
    """
    Ex, Ey, Bz = x0_hat[:, 3], x0_hat[:, 4], x0_hat[:, 5]
    curl_E  = _ddx(Ey) - _ddy(Ex)    # should vanish in static regime
    div_E   = _ddx(Ex) + _ddy(Ey)    # Gauss: ∇·E = 0 (charge-free)
    return (curl_E**2 + div_E**2).mean()


def divergence_penalty(x0_hat: torch.Tensor) -> torch.Tensor:
    """Enforce ∇·u = 0 (incompressibility)."""
    return ((_ddx(x0_hat[:, 0]) + _ddy(x0_hat[:, 1]))**2).mean()


# ─────────────────────────────────────────────
#  Complete PDE Diffusion Model
# ─────────────────────────────────────────────

class PDEDiffusionModel(nn.Module):
    """
    Wraps the schedule, PDE kernels, and score network into
    a single trainable module.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg      = cfg
        self.schedule = DiffusionSchedule(cfg)
        self.ns_kern  = NavierStokesKernel(cfg.nu, cfg.dt)
        self.em_kern  = MaxwellKernel(cfg.eps, cfg.mu, cfg.dt)
        self.score_net = PDEScoreUNet(cfg)

    # ── forward q(xₜ | x₀) with PDE-structured noise ──

    def pde_noise(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply PDE-structured noise injection.
        The PDE kernels advect/scatter the base Gaussian noise so the
        noise manifold respects fluid / EM structure.
        Returns ε ~ PDE-correlated noise (same shape as x0).
        """
        # Start from pure Gaussian
        eps = torch.randn_like(x0)
        # Apply PDE kernels to structure the noise field
        beta_t = self.schedule.betas[t]               # (B,)
        eps = self.ns_kern(eps, beta_t)               # NS advection on noise
        eps = self.em_kern(eps, beta_t)               # EM scattering on noise
        # Re-normalise to unit variance per sample
        eps = eps - eps.mean(dim=[1, 2, 3], keepdim=True)
        std = eps.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        return eps / std

    def q_sample_pde(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Sample xₜ using PDE-structured noise."""
        eps = self.pde_noise(x0, t)
        xt  = self.schedule.q_sample(x0, t, eps)
        return xt, eps

    # ── reverse pθ(xₜ₋₁ | xₜ) ──

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """One DDPM reverse step."""
        cfg  = self.cfg
        sched = self.schedule
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps_hat = self.score_net(xt, t_batch)

        alpha   = 1.0 - sched.betas[t]
        alpha_b = sched.alphas_cumprod[t]
        coef1   = 1.0 / alpha.sqrt()
        coef2   = sched.betas[t] / (1.0 - alpha_b).sqrt()
        mean    = coef1 * (xt - coef2 * eps_hat)

        if t > 0:
            noise = torch.randn_like(xt)
            var   = sched.posterior_variance[t]
            return mean + var.sqrt() * noise
        return mean

    @torch.no_grad()
    def sample(self, shape: tuple, device: str) -> torch.Tensor:
        """Full reverse chain: xT → x0."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.cfg.T)):
            x = self.p_sample(x, t)
        return x


# ─────────────────────────────────────────────
#  Training Step
# ─────────────────────────────────────────────

class TrainingStep:
    """
    Encapsulates one gradient update step.

    ℒ = ℒ_DDPM  +  λ_NS · ℒ_NS  +  λ_EM · ℒ_EM  +  λ_div · ℒ_div

    where x̂₀ is recovered from ε̂θ via DDPM:
        x̂₀ = (xₜ − √(1−ᾱₜ) · ε̂θ) / √ᾱₜ
    """

    def __init__(self, model: PDEDiffusionModel, cfg: DiffusionConfig):
        self.model = model
        self.cfg   = cfg
        self.opt   = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
        self.sched = CosineAnnealingLR(self.opt, T_max=cfg.num_steps)

    def step(self, x0: torch.Tensor) -> dict:
        """
        x0 : (B, C, H, W) — batch of clean fields
        Returns dict of loss components.
        """
        model  = self.model
        cfg    = self.cfg
        sched  = model.schedule
        device = x0.device

        # ── Sample t ~ Uniform[0, T) ──
        t = torch.randint(0, cfg.T, (x0.shape[0],), device=device)

        # ── Forward: apply PDE-structured noise ──
        xt, eps = model.q_sample_pde(x0, t)

        # ── Predict noise with score network ──
        eps_hat = model.score_net(xt, t, x_for_pde=xt)

        # ── DDPM loss ──
        loss_ddpm = F.mse_loss(eps_hat, eps)

        # ── Reconstruct x̂₀ from ε̂θ ──
        sqrt_alpha = sched._extract(sched.sqrt_alphas_cumprod, t, xt)
        sqrt_om    = sched._extract(sched.sqrt_one_minus_alphas_cumprod, t, xt)
        x0_hat     = (xt - sqrt_om * eps_hat) / sqrt_alpha.clamp(min=1e-5)
        x0_hat     = x0_hat.clamp(-4.0, 4.0)    # numerical safety

        # ── PDE physics losses on x̂₀ ──
        loss_ns  = ns_residual_loss(x0_hat, cfg.nu)
        loss_em  = maxwell_residual_loss(x0_hat, cfg.eps, cfg.mu)
        loss_div = divergence_penalty(x0_hat)

        # ── Total loss ──
        loss = (loss_ddpm
                + cfg.lambda_ns  * loss_ns
                + cfg.lambda_em  * loss_em
                + cfg.lambda_div * loss_div)

        # ── Optimise ──
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.sched.step()

        return {
            "loss":       loss.item(),
            "loss_ddpm":  loss_ddpm.item(),
            "loss_ns":    loss_ns.item(),
            "loss_em":    loss_em.item(),
            "loss_div":   loss_div.item(),
        }


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train(cfg: Optional[DiffusionConfig] = None, dataset=None):
    """
    Main training loop.

    dataset : iterable yielding (B, C, H, W) tensors.
              If None, uses synthetic random fields for demonstration.
    """
    if cfg is None:
        cfg = DiffusionConfig()

    device = torch.device(cfg.device)
    model  = PDEDiffusionModel(cfg).to(device)
    trainer = TrainingStep(model, cfg)

    print(f"PDE Diffusion Model")
    print(f"  Grid:      {cfg.H}×{cfg.W}, C={cfg.C}")
    print(f"  T:         {cfg.T}  ({cfg.schedule} schedule)")
    print(f"  ν={cfg.nu}, ε={cfg.eps}, μ={cfg.mu}")
    print(f"  Loss λ: NS={cfg.lambda_ns}, EM={cfg.lambda_em}, div={cfg.lambda_div}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Device:     {device}")
    print()

    log_interval = 100
    running = {k: 0.0 for k in ["loss", "loss_ddpm", "loss_ns", "loss_em", "loss_div"]}

    for step in range(1, cfg.num_steps + 1):
        # Fetch batch (real dataset or synthetic)
        if dataset is not None:
            x0 = next(iter(dataset)).to(device)
        else:
            # Synthetic: divergence-free velocity + EM field initialised from curl
            x0 = _synthetic_batch(cfg, device)

        losses = trainer.step(x0)

        for k, v in losses.items():
            running[k] += v

        if step % log_interval == 0:
            avg = {k: running[k] / log_interval for k in running}
            print(f"[{step:6d}/{cfg.num_steps}]  "
                  f"loss={avg['loss']:.4f}  "
                  f"ddpm={avg['loss_ddpm']:.4f}  "
                  f"ns={avg['loss_ns']:.4f}  "
                  f"em={avg['loss_em']:.4f}  "
                  f"div={avg['loss_div']:.5f}")
            running = {k: 0.0 for k in running}

    return model


# ─────────────────────────────────────────────
#  Synthetic data helper
# ─────────────────────────────────────────────

def _synthetic_batch(cfg: DiffusionConfig, device: torch.device) -> torch.Tensor:
    """
    Generate a divergence-free velocity + consistent EM field for testing.
    Uses stream-function ψ so that u = ∂ψ/∂y, v = −∂ψ/∂x  →  ∇·u = 0.
    """
    B, H, W = cfg.batch_size, cfg.H, cfg.W
    psi  = torch.randn(B, H, W, device=device)
    # Smooth with Laplacian damping
    for _ in range(5):
        psi = psi + 0.1 * _laplacian(psi)
    u = _ddy(psi)
    v = -_ddx(psi)
    p = torch.randn(B, H, W, device=device) * 0.1

    # EM: generate from a scalar potential A_z → curl-based E
    Az  = torch.randn(B, H, W, device=device)
    for _ in range(3):
        Az = Az + 0.1 * _laplacian(Az)
    Ex = _ddy(Az)
    Ey = -_ddx(Az)
    Bz = torch.randn(B, H, W, device=device) * 0.05

    return torch.stack([u, v, p, Ex, Ey, Bz], dim=1)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = DiffusionConfig(
        H=64, W=64, C=6,
        T=1000,
        schedule="cosine",
        base_channels=64,
        channel_mults=[1, 2, 4, 8],
        num_res_blocks=2,
        attention_res=[16, 8],
        nu=1e-3, eps=1.0, mu=1.0, dt=1e-2,
        lambda_ns=0.1, lambda_em=0.1, lambda_div=0.05,
        batch_size=4,
        lr=1e-4,
        num_steps=10_000,
    )
    model = train(cfg)
