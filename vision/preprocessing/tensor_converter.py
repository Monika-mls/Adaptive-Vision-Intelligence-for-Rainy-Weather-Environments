
"""
Tensor conversion utilities for Adaptive Vision Intelligence.

Converts RGB PIL images and NumPy arrays to PyTorch tensors.
Preserves spatial dimensions and supports optional normalization.
"""

from typing import Literal

import numpy as np
import torch
from PIL import Image


NormalizationMode = Literal["zero_one", "imagenet"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TensorConversionError(ValueError):
    """Raised for invalid image tensor input."""


def image_to_tensor(
    image: Image.Image | np.ndarray,
    normalization: NormalizationMode = "zero_one",
) -> torch.Tensor:
    """
    Convert an RGB image into a float32 CHW tensor.

    Input:
        RGB PIL image or uint8 HWC NumPy array.

    Output:
        Float32 tensor with shape (3, H, W).

    Normalization:
        zero_one: pixel values divided by 255.
        imagenet: ImageNet channel normalization.
    """

    if normalization not in ("zero_one", "imagenet"):
        raise TensorConversionError(
            f"Unsupported normalization: {normalization}"
        )

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise TensorConversionError(
                "PIL image must be in RGB mode."
            )
        array = np.asarray(image)

    elif isinstance(image, np.ndarray):
        array = image

    else:
        raise TensorConversionError(
            "Input must be a PIL image or NumPy array."
        )

    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
    ):
        raise TensorConversionError(
            "Expected a nonempty RGB image with shape (H, W, 3)."
        )

    if array.dtype != np.uint8:
        raise TensorConversionError(
            "Expected uint8 image data."
        )

    array = np.ascontiguousarray(array)

    tensor = (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
    )

    if normalization == "imagenet":
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


def tensor_to_image(
    tensor: torch.Tensor,
    normalization: NormalizationMode = "zero_one",
) -> Image.Image:
    """
    Convert a CHW tensor back to an RGB PIL image.

    Reverses the specified normalization.
    Clips floating-point rounding differences to [0, 1].
    """

    if normalization not in ("zero_one", "imagenet"):
        raise TensorConversionError(
            f"Unsupported normalization: {normalization}"
        )

    if not isinstance(tensor, torch.Tensor):
        raise TensorConversionError(
            "Expected a PyTorch tensor."
        )

    if (
        tensor.ndim != 3
        or tensor.shape[0] != 3
        or tensor.shape[1] == 0
        or tensor.shape[2] == 0
    ):
        raise TensorConversionError(
            "Expected tensor with shape (3, H, W)."
        )

    data = tensor.detach().to(
        device="cpu", dtype=torch.float32
    ).clone()

    if not torch.isfinite(data).all().item():
        raise TensorConversionError(
            "Tensor contains non-finite values."
        )

    if normalization == "imagenet":
        mean = data.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = data.new_tensor(IMAGENET_STD).view(3, 1, 1)
        data = data * std + mean

    array = (
        data.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    return Image.fromarray(array, mode="RGB")


def add_batch_dimension(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Convert a CHW tensor to a BCHW tensor."""

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 3
        or tensor.shape[0] != 3
    ):
        raise TensorConversionError(
            "Expected CHW RGB tensor."
        )

    return tensor.unsqueeze(0)
