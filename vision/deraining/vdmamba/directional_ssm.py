
"""
Experimental directional selective SSM image block.

Combines:
- Horizontal feature sequences
- Vertical feature sequences
- Two independent selective state-space processors
- Learnable convolutional feature fusion
- Residual connection

Input and output: [B, C, H, W]

This is a research prototype, NOT a faithful implementation
of the published VDMamba VDSM or projection units.
"""

import torch
from torch import nn

from vision.deraining.vdmamba.directional import (
    horizontal_sequences,
    vertical_sequences,
    restore_horizontal,
    restore_vertical,
)
from vision.deraining.vdmamba.selective_ssm import (
    SelectiveSSM,
)


class DirectionalSelectiveSSM(nn.Module):
    def __init__(
        self,
        channels=16,
        state_size=8,
        kernel_size=3,
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be positive")

        self.channels = channels

        self.norm = nn.LayerNorm(channels)

        self.horizontal_ssm = SelectiveSSM(
            channels=channels,
            state_size=state_size,
            kernel_size=kernel_size,
        )

        self.vertical_ssm = SelectiveSSM(
            channels=channels,
            state_size=state_size,
            kernel_size=kernel_size,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

        # Small residual contribution at initialization.
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a tensor")

        if x.ndim != 4:
            raise ValueError(
                "Expected [B, C, H, W]"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels"
            )

        if not x.is_floating_point():
            raise TypeError(
                "Expected floating-point features"
            )

        if any(d < 1 for d in x.shape):
            raise ValueError(
                "All tensor dimensions must be positive"
            )

        original_shape = x.shape

        # Layer normalization over channels.
        features = x.permute(0, 2, 3, 1)
        features = self.norm(features)
        features = features.permute(0, 3, 1, 2).contiguous()

        # Horizontal scan.
        horizontal = horizontal_sequences(features)
        horizontal = self.horizontal_ssm(horizontal)

        horizontal_features = restore_horizontal(
            horizontal,
            original_shape,
        )

        # Vertical scan.
        vertical = vertical_sequences(features)
        vertical = self.vertical_ssm(vertical)

        vertical_features = restore_vertical(
            vertical,
            original_shape,
        )

        # Fuse directional representations.
        combined = torch.cat(
            [horizontal_features, vertical_features],
            dim=1,
        )

        fused = self.fusion(combined)

        return x + self.residual_scale * fused
