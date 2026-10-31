
import numpy as np
import pytest
import torch
from PIL import Image

from vision.preprocessing.tensor_converter import (
    TensorConversionError,
    add_batch_dimension,
    image_to_tensor,
    tensor_to_image,
)


def test_rgb_image_conversion():
    image = Image.new("RGB", (64, 32), "red")

    tensor = image_to_tensor(image)

    assert tensor.shape == (3, 32, 64)
    assert tensor.dtype == torch.float32
    assert torch.all((tensor >= 0) & (tensor <= 1))


def test_numpy_conversion():
    array = np.zeros((20, 30, 3), dtype=np.uint8)

    tensor = image_to_tensor(array)

    assert tensor.shape == (3, 20, 30)


def test_imagenet_normalization():
    image = Image.new("RGB", (16, 16), "white")

    tensor = image_to_tensor(
        image, normalization="imagenet"
    )

    expected = torch.tensor([
        (1 - 0.485) / 0.229,
        (1 - 0.456) / 0.224,
        (1 - 0.406) / 0.225,
    ])

    assert torch.allclose(
        tensor[:, 0, 0], expected, atol=1e-5
    )


def test_round_trip_zero_one():
    array = np.random.default_rng(42).integers(
        0, 256, (32, 40, 3), dtype=np.uint8
    )

    tensor = image_to_tensor(array)
    restored = tensor_to_image(tensor)

    assert np.array_equal(
        array, np.asarray(restored)
    )


def test_round_trip_imagenet():
    image = Image.new("RGB", (20, 10), (40, 90, 150))

    tensor = image_to_tensor(
        image, normalization="imagenet"
    )

    restored = tensor_to_image(
        tensor, normalization="imagenet"
    )

    assert np.array_equal(
        np.asarray(image), np.asarray(restored)
    )


def test_batch_dimension():
    tensor = torch.zeros((3, 32, 64))

    batch = add_batch_dimension(tensor)

    assert batch.shape == (1, 3, 32, 64)


def test_reject_grayscale():
    image = Image.new("L", (32, 32))

    with pytest.raises(TensorConversionError):
        image_to_tensor(image)


def test_reject_invalid_numpy_dtype():
    array = np.zeros((32, 32, 3), dtype=np.float32)

    with pytest.raises(TensorConversionError):
        image_to_tensor(array)


def test_reject_invalid_normalization():
    image = Image.new("RGB", (10, 10))

    with pytest.raises(TensorConversionError):
        image_to_tensor(image, normalization="invalid")


def test_reject_nonfinite_tensor():
    tensor = torch.zeros((3, 10, 10))
    tensor[0, 0, 0] = float("nan")

    with pytest.raises(TensorConversionError):
        tensor_to_image(tensor)
