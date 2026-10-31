"""Experimental residual directional-SSM unit for the VDMamba reproduction.

This is a prototype, NOT a validated reproduction of the paper's RMU.
It reuses the project's tested DirectionalSelectiveSSM reference implementation.
"""
import torch
from torch import nn

from vision.deraining.vdmamba.directional_ssm import DirectionalSelectiveSSM


class ChannelAttention(nn.Module):
    """Squeeze/excitation attention, without assuming a particular spatial size."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ExperimentalRMU(nn.Module):
    """Residual attention + directional SSM + local feedforward prototype.

    Input/output: [B,C,H,W]. Spatial size unchanged.
    """

    def __init__(self, channels: int = 8, state_size: int = 4, expansion: int = 2):
        super().__init__()
        if channels < 1 or state_size < 1 or expansion < 1:
            raise ValueError("channels, state_size and expansion must be positive")
        inner = channels * expansion
        self.channels = channels
        self.norm1 = ChannelLayerNorm(channels)
        self.attention = ChannelAttention(channels)
        self.directional_ssm = DirectionalSelectiveSSM(
            channels=channels, state_size=state_size
        )
        self.norm2 = ChannelLayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, inner, 1),
            nn.GELU(),
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1),
        )
        # Residual scales stabilize a randomly initialized prototype.
        self.gamma1 = nn.Parameter(torch.tensor(0.1))
        self.gamma2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a tensor")
        if x.ndim != 4 or x.shape[1] != self.channels or any(d < 1 for d in x.shape):
            raise ValueError(f"Expected [B,{self.channels},H,W] with positive dimensions")
        if not x.is_floating_point():
            raise TypeError("Expected floating-point features")

        z = self.norm1(x)
        attended = self.attention(z)
        # DirectionalSelectiveSSM itself has a residual path; subtract its
        # input to avoid adding another unscaled identity in this residual.
        processed = self.directional_ssm(attended) - attended
        x = x + self.gamma1 * processed
        return x + self.gamma2 * self.ffn(self.norm2(x))
