
"""
Deterministic inference preprocessing.

Provides MobileNetV3-Small severity-classification
preprocessing and generic restoration input preparation.

Actual restoration model adapters must verify their
specific preprocessing requirements independently.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

from vision.preprocessing.tensor_converter import (
    add_batch_dimension,
    image_to_tensor,
)


@dataclass(frozen=True)
class InferenceConfig:
    task: Literal["severity", "restoration"] = "restoration"
    device: str = "cpu"


class InferencePreprocessingError(ValueError):
    """Raised when inference preprocessing is invalid."""


def prepare_inference_image(
    image: Image.Image,
    config: InferenceConfig | None = None,
) -> torch.Tensor:
    """
    Prepare a single RGB image for inference.

    Returns:
        Float32 tensor with shape (1, 3, H, W).

    Severity:
        Uses official MobileNetV3-Small ImageNet
        inference preprocessing.

    Restoration:
        Preserves resolution and converts RGB
        values into the [0, 1] range.
    """

    config = config or InferenceConfig()

    if not isinstance(image, Image.Image):
        raise InferencePreprocessingError(
            "Expected a PIL image."
        )

    if image.mode != "RGB":
        raise InferencePreprocessingError(
            "Expected RGB image. Use load_image first."
        )

    if config.task == "severity":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

        transform = weights.transforms()

        tensor = transform(image)

    elif config.task == "restoration":
        tensor = image_to_tensor(
            image, normalization="zero_one"
        )

    else:
        raise InferencePreprocessingError(
            f"Unsupported inference task: {config.task}"
        )

    batch = add_batch_dimension(tensor)

    try:
        device = torch.device(config.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise InferencePreprocessingError(
                "CUDA was requested but is unavailable."
            )

        batch = batch.to(device)

    except (RuntimeError, ValueError) as error:
        raise InferencePreprocessingError(
            f"Cannot prepare tensor on {config.device}: {error}"
        ) from error

    return batch


def describe_tensor(tensor: torch.Tensor) -> dict:
    """Return basic tensor characteristics."""

    if not isinstance(tensor, torch.Tensor):
        raise InferencePreprocessingError(
            "Expected a PyTorch tensor."
        )

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }
