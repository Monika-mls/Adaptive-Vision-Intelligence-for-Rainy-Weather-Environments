
"""
Image loading utilities for Adaptive Vision Intelligence.

Provides safe loading, RGB conversion, image-pair validation,
and NumPy array conversion without unnecessary resizing.
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError


ImagePath = Union[str, Path]


class ImageLoadingError(Exception):
    """Raised when an image cannot be loaded or validated."""


def load_image(path: ImagePath) -> Image.Image:
    """
    Load an image and return a fully decoded RGB PIL image.

    Preserves original image width and height.
    """

    image_path = Path(path)

    if not image_path.is_file():
        raise ImageLoadingError(
            f"Image file does not exist: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image.load()

            if image.width <= 0 or image.height <= 0:
                raise ImageLoadingError(
                    f"Invalid dimensions: {image_path}"
                )

            return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise ImageLoadingError(
            f"Cannot load image {image_path}: {error}"
        ) from error


def load_image_array(path: ImagePath) -> np.ndarray:
    """
    Load an image as an RGB uint8 NumPy array.

    Output shape: (height, width, 3).
    """

    image = load_image(path)
    return np.asarray(image, dtype=np.uint8).copy()


def load_image_pair(
    rainy_path: ImagePath,
    clean_path: ImagePath,
) -> tuple[Image.Image, Image.Image]:
    """
    Load a paired rainy and clean image.

    Both images must have identical dimensions.
    """

    rainy = load_image(rainy_path)
    clean = load_image(clean_path)

    if rainy.size != clean.size:
        raise ImageLoadingError(
            "Rainy and clean image dimensions do not match: "
            f"{rainy.size} vs {clean.size}"
        )

    return rainy, clean


def validate_image(path: ImagePath) -> dict:
    """
    Return basic validated image information.
    """

    image = load_image(path)

    return {
        "path": str(Path(path)),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "valid": True,
    }
