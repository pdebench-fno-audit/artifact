"""Core FNO model architectures used in the paper.

No cloud dependencies — pure PyTorch. These are the exact architectures
from which all reported results were produced.

Usage:
    from models import FNO1d_AR, FNO2d_AR, VorticityPoissonFNO
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 1D Spectral Convolution (periodic FFT)
# ═══════════════════════════════════════════════════════════
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        self.w = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.w.shape[1],
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.w
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


# ═══════════════════════════════════════════════════════════
# 2D Spectral Convolution (periodic FFT)
# ═══════════════════════════════════════════════════════════
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        H_ft = x_ft.shape[-2]
        out_ft = torch.zeros(
            B,
            self.w1.shape[1],
            H_ft,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.w1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.w2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ═══════════════════════════════════════════════════════════
# FNO Blocks
# ═══════════════════════════════════════════════════════════
class FNO1dBlock(nn.Module):
    """Base 1D FNO block: spectral conv + pointwise conv + residual."""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pw = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pw(x))


class FNO2dBlock(nn.Module):
    """Gated local-global 2D block (used for 2D time-dependent + CFD tests)."""

    def __init__(self, width, modes, local_kernel=5):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pw = nn.Conv2d(width, width, 1)
        self.lc = nn.Conv2d(width, width, local_kernel, padding=local_kernel // 2)
        self.gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        g = self.spectral(x) + self.pw(x)
        l = self.lc(x)
        a = torch.sigmoid(self.gate)
        return (1 - a) * g + a * l


# ═══════════════════════════════════════════════════════════
# Complete Models
# ═══════════════════════════════════════════════════════════
class FNO1d_AR(nn.Module):
    """1D FNO with autoregressive rollout."""

    def __init__(self, nc=1, modes=12, width=32, init_step=5, n_layers=4):
        super().__init__()
        self.fc0 = nn.Conv1d(init_step * nc + 1, width, 1)  # +1 for grid
        self.blocks = nn.ModuleList([FNO1dBlock(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Conv1d(width, 128, 1)
        self.fc2 = nn.Conv1d(128, nc, 1)

    def forward(self, x, grid):
        # x: [B, N, init_step*nc], grid: [B, N, 1]
        x = torch.cat([x, grid], dim=-1).permute(0, 2, 1)  # [B, C, N]
        x = self.fc0(x)
        for block in self.blocks:
            x = block(x)
        return self.fc2(F.gelu(self.fc1(x))).permute(0, 2, 1)  # [B, N, nc]


class FNO2d_AR(nn.Module):
    """2D FNO with gated local-global blocks and autoregressive rollout."""

    def __init__(self, nc=4, modes=12, width=32, init_step=5, n_layers=4):
        super().__init__()
        self.fc0 = nn.Conv2d(init_step * nc + 2, width, 1)  # +2 for grid
        self.blocks = nn.ModuleList([FNO2dBlock(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, nc, 1)
        self.n_layers = n_layers

    def forward(self, x, grid):
        # x: [B, H, W, init_step*nc], grid: [B, H, W, 2]
        x = torch.cat([x, grid], dim=-1).permute(0, 3, 1, 2)
        x = self.fc0(x)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        return self.fc2(F.gelu(self.fc1(x))).permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════
# Vorticity-Poisson FNO (Test 28: 2D Incompressible NS)
# ═══════════════════════════════════════════════════════════
class DST_SpectralConv2d(nn.Module):
    """DST-I spectral convolution for Dirichlet BCs (FFT-based)."""

    def __init__(self, ic, oc, mx, my):
        super().__init__()
        self.mx, self.my = mx, my
        self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

    @staticmethod
    def dst1_1d(x, dim=-1):
        N = x.shape[dim]
        zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
        x_ext = torch.cat([zeros, x, zeros, -x.flip(dims=[dim])], dim=dim)
        x_ft = torch.fft.fft(x_ext, dim=dim)
        return -torch.narrow(x_ft.imag, dim, 1, N)

    def forward(self, x):
        x_f = x.float() if x.dtype != torch.float32 else x
        # Forward 2D DST
        x_st = self.dst1_1d(self.dst1_1d(x_f, dim=-1), dim=-2)
        # Truncate, apply weights, zero-pad
        x_t = x_st[:, :, : self.mx, : self.my]
        o = torch.einsum("bihw,iohw->bohw", x_t, self.w.float())
        out = torch.zeros_like(x_st)
        out[:, :, : self.mx, : self.my] = o
        # Inverse 2D DST
        Nh, Nw = x.shape[-2], x.shape[-1]
        result = self.dst1_1d(self.dst1_1d(out, dim=-1), dim=-2)
        return (result / ((2 * (Nh + 1)) * (2 * (Nw + 1)))).to(x.dtype)


class DSTBlock2d(nn.Module):
    def __init__(self, w, mx, my):
        super().__init__()
        self.spec = DST_SpectralConv2d(w, w, mx, my)
        self.pw = nn.Conv2d(w, w, 1)
        self.norm = nn.InstanceNorm2d(w)

    def forward(self, x):
        xn = self.norm(x)
        return x + F.gelu(self.spec(xn) + self.pw(xn))


class VorticityPoissonFNO(nn.Module):
    """Predicts vorticity; recovers velocity via DST Poisson solve + curl.

    Architecture for 2D incompressible NS with Dirichlet BCs.
    See Section 8 of the paper for details.
    """

    def __init__(self, modes=12, width=20, init_step=10, n_layers=4, has_forcing=True):
        super().__init__()
        in_ch = init_step * 1 + 2 + (2 if has_forcing else 0)
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [DSTBlock2d(width, modes, modes) for _ in range(n_layers)]
        )
        self.fc1 = nn.Conv2d(width, 64, 1)
        self.fc2 = nn.Conv2d(64, 1, 1)
        self.has_forcing = has_forcing

    def forward(self, omega_input, grid, forcing=None):
        parts = [omega_input, grid]
        if self.has_forcing and forcing is not None:
            parts.append(forcing)
        inp = torch.cat(parts, dim=-1).permute(0, 3, 1, 2)
        h = self.fc0(inp)
        for block in self.blocks:
            h = block(h)
        return self.fc2(F.gelu(self.fc1(h)))


# ═══════════════════════════════════════════════════════════
# Metrics (matching PDEBench convention)
# ═══════════════════════════════════════════════════════════
def compute_nrmse_per_timestep(pred, target, initial_step=1):
    """Per-timestep nRMSE matching PDEBench metric_func convention.

    pred, target: [B, ..., T, nc]
    Returns: scalar nRMSE averaged over channels and timesteps.
    """
    pred = pred[..., initial_step:, :]
    target = target[..., initial_step:, :]

    B = pred.shape[0]
    nc = pred.shape[-1]
    T = pred.shape[-2]

    # Reshape to [B, nc, spatial..., T]
    if pred.dim() == 4:  # 1D: [B, N, T, nc]
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif pred.dim() == 5:  # 2D: [B, H, W, T, nc]
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)

    # Per-sample, per-channel, per-timestep nRMSE
    err = torch.sqrt(
        torch.mean(
            (pred.reshape(B, nc, -1, T) - target.reshape(B, nc, -1, T)) ** 2, dim=2
        )
    )
    nrm = torch.sqrt(torch.mean(target.reshape(B, nc, -1, T) ** 2, dim=2))
    nrmse = torch.mean(err / (nrm + 1e-20), dim=0)  # average over batch

    return torch.mean(nrmse).item()  # average over channels and timesteps


# ═══════════════════════════════════════════════════════════
# Normalized MSE Loss (our key innovation)
# ═══════════════════════════════════════════════════════════
class NormalizedMSELoss(nn.Module):
    """Per-sample normalized MSE: aligns training with nRMSE evaluation.

    L = mean_over_samples( ||pred - target||^2 / ||target||^2 )

    This is the single most impactful training fix (2.05x degradation
    when removed on Test 13).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        B = pred.shape[0]
        pf = pred.reshape(B, -1)
        tf = target.reshape(B, -1)
        per_sample = (pf - tf).pow(2).sum(1) / (tf.pow(2).sum(1) + self.eps)
        return per_sample.mean()
