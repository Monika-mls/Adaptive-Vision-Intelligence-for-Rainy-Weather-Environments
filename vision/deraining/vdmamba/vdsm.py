"""Experimental VDSM-inspired *feature* block, NOT official VDMamba.

The module produces direction-specific residue estimates, exchanges information
between them, synthesizes a residue and subtracts it from background features.
Dimensions and fusion equations are implementation choices; they are not
claimed to reproduce the paper exactly. Untrained outputs are NOT derained images.
"""

import torch
from torch import nn

from vision.deraining.vdmamba.dpu import DirectionAwareProjection
from vision.deraining.vdmamba.rmu import ExperimentalRMU


class ExperimentalVDSM(nn.Module):
    """Input feature F -> (background F - R, residue R, vertical, horizontal).

    All tensors have [B,C,H,W] shape. No interpolation or spatial downsampling.
    Requires the Step 7 DPU and Step 8 RMU modules.
    """

    def __init__(self, channels: int = 8, state_size: int = 4):
        super().__init__()
        if channels < 1 or state_size < 1:
            raise ValueError("channels and state_size must be positive")
        self.channels = channels
        self.vertical_projection = DirectionAwareProjection(channels, "vertical")
        self.horizontal_projection = DirectionAwareProjection(channels, "horizontal")
        self.vertical_refine = ExperimentalRMU(channels, state_size)
        self.horizontal_refine = ExperimentalRMU(channels, state_size)

        # Explicit bidirectional exchange; per-direction gates depend on BOTH streams.
        self.vertical_exchange = nn.Conv2d(channels * 2, channels, 1)
        self.horizontal_exchange = nn.Conv2d(channels * 2, channels, 1)
        self.synthesis = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Small initial correction. A learnable scalar allows data-driven magnitude.
        self.residue_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor):
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 4 or features.shape[1] != self.channels or any(n < 1 for n in features.shape):
            raise ValueError(f"expected [B,{self.channels},H,W] with positive dimensions")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")

        vertical = self.vertical_refine(self.vertical_projection(features))
        horizontal = self.horizontal_refine(self.horizontal_projection(features))
        combined = torch.cat((vertical, horizontal), dim=1)
        vertical = vertical + torch.sigmoid(self.vertical_exchange(combined)) * horizontal
        horizontal = horizontal + torch.sigmoid(self.horizontal_exchange(combined)) * vertical
        residue = self.residue_scale * self.synthesis(torch.cat((vertical, horizontal), dim=1))
        background = features - residue
        return {
            "background": background,
            "residue": residue,
            "vertical": vertical,
            "horizontal": horizontal,
        }
