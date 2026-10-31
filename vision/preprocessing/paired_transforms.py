
"""
Synchronized paired image augmentations.

Applies identical spatial transformations to rainy and
clean images to preserve pixel-level correspondence.
"""

import random
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class PairedTransformConfig:
    crop_size: int = 256
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True


class PairedTransformError(ValueError):
    """Raised when paired transformation inputs are invalid."""


def validate_pair(
    rainy: Image.Image,
    clean: Image.Image,
) -> None:
    """Validate image types, modes and dimensions."""

    if not isinstance(rainy, Image.Image) or not isinstance(
        clean, Image.Image
    ):
        raise PairedTransformError(
            "Both inputs must be PIL images."
        )

    if rainy.mode != "RGB" or clean.mode != "RGB":
        raise PairedTransformError(
            "Both images must be in RGB mode."
        )

    if rainy.size != clean.size:
        raise PairedTransformError(
            "Paired images must have matching dimensions."
        )


class PairedTrainingTransform:
    """Random synchronized crop, flips and 90-degree rotations."""

    def __init__(
        self,
        config: PairedTransformConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or PairedTransformConfig()

        if (
            not isinstance(self.config.crop_size, int)
            or isinstance(self.config.crop_size, bool)
            or self.config.crop_size <= 0
        ):
            raise PairedTransformError(
                "crop_size must be a positive integer."
            )

        self.rng = random.Random(seed)

    def __call__(
        self,
        rainy: Image.Image,
        clean: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        validate_pair(rainy, clean)

        width, height = rainy.size
        crop_size = self.config.crop_size

        if width < crop_size or height < crop_size:
            raise PairedTransformError(
                f"Image size {rainy.size} is smaller "
                f"than crop size {crop_size}."
            )

        left = self.rng.randint(0, width - crop_size)
        top = self.rng.randint(0, height - crop_size)

        crop_box = (
            left,
            top,
            left + crop_size,
            top + crop_size,
        )

        rainy = rainy.crop(crop_box)
        clean = clean.crop(crop_box)

        if (
            self.config.horizontal_flip
            and self.rng.random() < 0.5
        ):
            rainy = rainy.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            clean = clean.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if (
            self.config.vertical_flip
            and self.rng.random() < 0.5
        ):
            rainy = rainy.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            clean = clean.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        if self.config.rotate_90:
            rotations = self.rng.randint(0, 3)

            for _ in range(rotations):
                rainy = rainy.transpose(
                    Image.Transpose.ROTATE_90
                )
                clean = clean.transpose(
                    Image.Transpose.ROTATE_90
                )

        return rainy, clean


def prepare_validation_pair(
    rainy: Image.Image,
    clean: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """
    Prepare a pair without stochastic augmentation.

    Preserves original resolution and spatial alignment.
    """

    validate_pair(rainy, clean)

    return rainy.copy(), clean.copy()
