
import numpy as np
import pytest
from PIL import Image

from vision.preprocessing.paired_transforms import (
    PairedTransformConfig,
    PairedTrainingTransform,
    PairedTransformError,
    prepare_validation_pair,
)


def make_test_pair():
    rng = np.random.default_rng(123)

    pixels = rng.integers(
        0, 256, (100, 120, 3), dtype=np.uint8
    )

    image = Image.fromarray(pixels, mode="RGB")

    return image, image.copy()


def test_output_patch_dimensions():
    rainy, clean = make_test_pair()

    transform = PairedTrainingTransform(
        PairedTransformConfig(crop_size=64),
        seed=42,
    )

    output_rainy, output_clean = transform(
        rainy, clean
    )

    assert output_rainy.size == (64, 64)
    assert output_clean.size == (64, 64)


def test_identical_inputs_remain_identical():
    rainy, clean = make_test_pair()

    transform = PairedTrainingTransform(
        PairedTransformConfig(crop_size=64),
        seed=10,
    )

    output_rainy, output_clean = transform(
        rainy, clean
    )

    assert np.array_equal(
        np.asarray(output_rainy),
        np.asarray(output_clean),
    )


def test_reproducible_seed():
    rainy, clean = make_test_pair()

    config = PairedTransformConfig(crop_size=64)

    transform1 = PairedTrainingTransform(
        config, seed=123
    )
    transform2 = PairedTrainingTransform(
        config, seed=123
    )

    result1, _ = transform1(rainy, clean)
    result2, _ = transform2(rainy, clean)

    assert np.array_equal(
        np.asarray(result1),
        np.asarray(result2),
    )


def test_validation_preserves_dimensions():
    rainy, clean = make_test_pair()

    output_rainy, output_clean = (
        prepare_validation_pair(rainy, clean)
    )

    assert output_rainy.size == rainy.size
    assert output_clean.size == clean.size


def test_validation_preserves_pixels():
    rainy, clean = make_test_pair()

    output_rainy, _ = prepare_validation_pair(
        rainy, clean
    )

    assert np.array_equal(
        np.asarray(rainy),
        np.asarray(output_rainy),
    )


def test_reject_mismatched_dimensions():
    rainy = Image.new("RGB", (100, 100))
    clean = Image.new("RGB", (80, 80))

    transform = PairedTrainingTransform(
        PairedTransformConfig(crop_size=64)
    )

    with pytest.raises(PairedTransformError):
        transform(rainy, clean)


def test_reject_oversized_crop():
    rainy, clean = make_test_pair()

    transform = PairedTrainingTransform(
        PairedTransformConfig(crop_size=256)
    )

    with pytest.raises(PairedTransformError):
        transform(rainy, clean)


def test_reject_invalid_crop_size():
    with pytest.raises(PairedTransformError):
        PairedTrainingTransform(
            PairedTransformConfig(crop_size=0)
        )


def test_reject_grayscale_pair():
    rainy = Image.new("L", (100, 100))
    clean = Image.new("L", (100, 100))

    with pytest.raises(PairedTransformError):
        prepare_validation_pair(rainy, clean)
