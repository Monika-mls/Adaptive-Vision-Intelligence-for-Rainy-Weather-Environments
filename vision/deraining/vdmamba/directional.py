
"""
Directional feature-sequence utilities for VDMamba research.

Input image features:
    [B, C, H, W]

Horizontal sequences:
    [B * H, W, C]

Vertical sequences:
    [B * W, H, C]

These operations are reversible and differentiable.

IMPORTANT:
This module provides directional sequence layouts.
It is not the published VDMamba Direction-Aware
Projection Unit or complete VDSM.
"""

import torch
from torch import nn


def validate_feature_tensor(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if x.ndim != 4:
        raise ValueError("Expected [B, C, H, W].")

    if not x.is_floating_point():
        raise TypeError("Expected floating-point features.")

    if any(d < 1 for d in x.shape):
        raise ValueError("All dimensions must be positive.")


def horizontal_sequences(x):
    """
    [B, C, H, W] -> [B*H, W, C]
    """
    validate_feature_tensor(x)

    b, c, h, w = x.shape

    sequences = (
        x.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(b * h, w, c)
    )

    return sequences


def vertical_sequences(x):
    """
    [B, C, H, W] -> [B*W, H, C]
    """
    validate_feature_tensor(x)

    b, c, h, w = x.shape

    sequences = (
        x.permute(0, 3, 2, 1)
        .contiguous()
        .reshape(b * w, h, c)
    )

    return sequences


def restore_horizontal(sequences, original_shape):
    b, c, h, w = original_shape

    if sequences.shape != (b * h, w, c):
        raise ValueError(
            "Horizontal sequence shape mismatch."
        )

    return (
        sequences.reshape(b, h, w, c)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def restore_vertical(sequences, original_shape):
    b, c, h, w = original_shape

    if sequences.shape != (b * w, h, c):
        raise ValueError(
            "Vertical sequence shape mismatch."
        )

    return (
        sequences.reshape(b, w, h, c)
        .permute(0, 3, 2, 1)
        .contiguous()
    )


class DirectionalFeatureLayout(nn.Module):
    """
    Produces and reconstructs horizontal/vertical
    feature sequences without learning parameters.
    """

    def forward(self, x):
        validate_feature_tensor(x)

        horizontal = horizontal_sequences(x)
        vertical = vertical_sequences(x)

        return horizontal, vertical

    def reconstruct(self, horizontal, vertical, shape):
        x_horizontal = restore_horizontal(
            horizontal, shape
        )

        x_vertical = restore_vertical(
            vertical, shape
        )

        return (x_horizontal + x_vertical) * 0.5
