
"""
Unit tests for image loading and validation.
"""

import numpy as np
import pytest
from PIL import Image

from vision.preprocessing.image_loader import (
    ImageLoadingError,
    load_image,
    load_image_array,
    load_image_pair,
    validate_image,
)


def test_load_valid_rgb_image(tmp_path):
    path = tmp_path / "valid.png"

    Image.new("RGB", (100, 80), "red").save(path)

    result = load_image(path)

    assert result.size == (100, 80)
    assert result.mode == "RGB"


def test_grayscale_image_converts_to_rgb(tmp_path):
    path = tmp_path / "gray.png"

    Image.new("L", (40, 30), 128).save(path)

    result = load_image(path)

    assert result.mode == "RGB"
    assert result.size == (40, 30)


def test_image_array_shape_and_dtype(tmp_path):
    path = tmp_path / "array.png"

    Image.new("RGB", (64, 32), "blue").save(path)

    result = load_image_array(path)

    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 64, 3)
    assert result.dtype == np.uint8


def test_load_matching_pair(tmp_path):
    rainy_path = tmp_path / "rainy.png"
    clean_path = tmp_path / "clean.png"

    Image.new("RGB", (100, 100), "gray").save(
        rainy_path
    )
    Image.new("RGB", (100, 100), "white").save(
        clean_path
    )

    rainy, clean = load_image_pair(
        rainy_path, clean_path
    )

    assert rainy.size == clean.size


def test_reject_mismatched_pair(tmp_path):
    rainy_path = tmp_path / "rainy.png"
    clean_path = tmp_path / "clean.png"

    Image.new("RGB", (100, 100)).save(rainy_path)
    Image.new("RGB", (80, 80)).save(clean_path)

    with pytest.raises(ImageLoadingError):
        load_image_pair(rainy_path, clean_path)


def test_reject_missing_image(tmp_path):
    with pytest.raises(ImageLoadingError):
        load_image(tmp_path / "missing.png")


def test_reject_corrupted_image(tmp_path):
    path = tmp_path / "corrupted.png"

    path.write_bytes(b"This is not a real image")

    with pytest.raises(ImageLoadingError):
        load_image(path)


def test_validate_image_metadata(tmp_path):
    path = tmp_path / "sample.png"

    Image.new("RGB", (120, 90)).save(path)

    result = validate_image(path)

    assert result["valid"] is True
    assert result["width"] == 120
    assert result["height"] == 90
    assert result["mode"] == "RGB"
