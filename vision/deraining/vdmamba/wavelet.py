
"""
Haar wavelet transform for the VDMamba reproduction.

Pure PyTorch implementation:
- Forward discrete wavelet transform (DWT)
- Inverse discrete wavelet transform (IDWT)
- Supports CPU and CUDA
- Supports autograd
- Handles odd image dimensions using replicate padding

This is a foundational wavelet component, not the
complete FEM or VDMamba architecture.
"""

import torch
import torch.nn.functional as F
from torch import nn


class HaarDWT(nn.Module):
    """
    Orthogonal 2D Haar decomposition.

    Input:
        [B, C, H, W]

    Outputs:
        LL, LH, HL, HH: [B, C, ceil(H/2), ceil(W/2)]
        original_size: (H, W)
    """

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W] input.")

        if not x.is_floating_point():
            raise TypeError("Expected floating-point input.")

        height, width = x.shape[-2:]

        if height < 1 or width < 1:
            raise ValueError("Spatial dimensions must be positive.")

        pad_h = height % 2
        pad_w = width % 2

        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh, (height, width)


class HaarIDWT(nn.Module):
    """
    Inverse orthogonal 2D Haar reconstruction.
    """

    def forward(self, ll, lh, hl, hh, original_size):
        shapes = [x.shape for x in (ll, lh, hl, hh)]

        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("Wavelet bands must have equal shapes.")

        if ll.ndim != 4:
            raise ValueError("Expected 4D wavelet bands.")

        height, width = original_size

        if not (1 <= height <= ll.shape[-2] * 2):
            raise ValueError("Invalid original height.")

        if not (1 <= width <= ll.shape[-1] * 2):
            raise ValueError("Invalid original width.")

        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        batch, channels, half_h, half_w = ll.shape

        output = torch.empty(
            batch,
            channels,
            half_h * 2,
            half_w * 2,
            device=ll.device,
            dtype=ll.dtype,
        )

        output[..., 0::2, 0::2] = x00
        output[..., 0::2, 1::2] = x01
        output[..., 1::2, 0::2] = x10
        output[..., 1::2, 1::2] = x11

        return output[..., :height, :width]


def wavelet_roundtrip(x):
    dwt = HaarDWT()
    idwt = HaarIDWT()

    ll, lh, hl, hh, size = dwt(x)

    return idwt(ll, lh, hl, hh, size)
