"""Experimental direction-aware projection, informed by VDMamba Fig. 2.

The paper shows direction-aware average pooling and a 1x1 -> Norm ->
Swish -> 1x1 -> sigmoid gate. Pool axes, channels and normalization
are implementation choices until official specifications are verified.
NOT an exact reproduction of the published DPU.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DirectionAwareProjection(nn.Module):
    """Compute axis-conditioned spatial gating without changing [B,C,H,W]."""

    def __init__(self, channels: int, direction: str, reduction: int = 4):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        if direction not in {"vertical", "horizontal"}:
            raise ValueError("direction must be 'vertical' or 'horizontal'")
        self.channels = channels
        self.direction = direction
        hidden = max(2, channels // reduction)
        self.conv_in = nn.Conv2d(channels, hidden, 1)
        self.norm = nn.GroupNorm(1, hidden)
        self.conv_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("expected a PyTorch Tensor")
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape) < 1:
            raise ValueError(f"expected [B,{self.channels},H,W] with positive sizes")
        if not x.is_floating_point():
            raise TypeError("expected floating-point features")
        # Preserve vertical coordinate when learning vertical variation;
        # preserve horizontal coordinate for horizontal variation.
        pool_dimension = 3 if self.direction == "vertical" else 2
        descriptor = x.mean(dim=pool_dimension, keepdim=True)
        gate = torch.sigmoid(self.conv_out(F.silu(self.norm(self.conv_in(descriptor)))))
        return x * gate
