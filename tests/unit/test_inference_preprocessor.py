
import pytest
import torch
from PIL import Image

from vision.preprocessing.inference_preprocessor import (
    InferenceConfig,
    InferencePreprocessingError,
    describe_tensor,
    prepare_inference_image,
)


def make_image():
    return Image.new("RGB", (320, 280), "gray")


def test_severity_tensor_shape():
    result = prepare_inference_image(
        make_image(),
        InferenceConfig(task="severity"),
    )

    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == torch.float32


def test_restoration_preserves_dimensions():
    result = prepare_inference_image(
        make_image(),
        InferenceConfig(task="restoration"),
    )

    assert result.shape == (1, 3, 280, 320)


def test_restoration_value_range():
    result = prepare_inference_image(
        make_image(),
        InferenceConfig(task="restoration"),
    )

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_preprocessing_is_deterministic():
    image = make_image()

    first = prepare_inference_image(
        image, InferenceConfig(task="severity")
    )

    second = prepare_inference_image(
        image, InferenceConfig(task="severity")
    )

    assert torch.equal(first, second)


def test_reject_invalid_task():
    with pytest.raises(InferencePreprocessingError):
        prepare_inference_image(
            make_image(),
            InferenceConfig(task="invalid"),
        )


def test_reject_grayscale_image():
    image = Image.new("L", (100, 100))

    with pytest.raises(InferencePreprocessingError):
        prepare_inference_image(image)


def test_tensor_description():
    result = prepare_inference_image(
        make_image(),
        InferenceConfig(task="restoration"),
    )

    description = describe_tensor(result)

    assert description["shape"] == [1, 3, 280, 320]
    assert description["dtype"] == "torch.float32"


def test_cpu_device():
    result = prepare_inference_image(
        make_image(),
        InferenceConfig(
            task="restoration",
            device="cpu",
        ),
    )

    assert result.device.type == "cpu"
