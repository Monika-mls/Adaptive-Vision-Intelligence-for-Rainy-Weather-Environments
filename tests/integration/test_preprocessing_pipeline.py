
"""
End-to-end preprocessing integration tests.

Uses existing RealRain-1k sample images.
No downloads, training or model weights required.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from vision.preprocessing.image_loader import (
    load_image_pair,
)
from vision.preprocessing.tensor_converter import (
    image_to_tensor,
    tensor_to_image,
)
from vision.preprocessing.paired_transforms import (
    PairedTrainingTransform,
    PairedTransformConfig,
    prepare_validation_pair,
)
from vision.preprocessing.inference_preprocessor import (
    InferenceConfig,
    prepare_inference_image,
)


SAMPLE_DIR = Path(
    "datasets/sample/RealRain-1k-H/validation"
)

RAINY_PATH = SAMPLE_DIR / "input/1007.png"
CLEAN_PATH = SAMPLE_DIR / "target/1007.png"


@pytest.fixture
def real_image_pair():
    if not RAINY_PATH.exists() or not CLEAN_PATH.exists():
        pytest.fail(
            "RealRain-1k sample pair is missing. "
            "Complete Phase 1 sample validation first."
        )

    return load_image_pair(RAINY_PATH, CLEAN_PATH)


def test_real_image_pair_loading(real_image_pair):
    rainy, clean = real_image_pair

    assert rainy.size == clean.size
    assert rainy.mode == clean.mode == "RGB"
    assert rainy.size == (1032, 1000)


def test_tensor_roundtrip(real_image_pair):
    rainy, _ = real_image_pair

    tensor = image_to_tensor(rainy)
    restored = tensor_to_image(tensor)

    assert tensor.shape == (3, 1000, 1032)
    assert np.array_equal(
        np.asarray(rainy),
        np.asarray(restored),
    )


def test_paired_augmentation(real_image_pair):
    rainy, clean = real_image_pair

    transform = PairedTrainingTransform(
        PairedTransformConfig(crop_size=256),
        seed=42,
    )

    rainy_patch, clean_patch = transform(
        rainy, clean
    )

    assert rainy_patch.size == (256, 256)
    assert clean_patch.size == (256, 256)

    rainy_tensor = image_to_tensor(rainy_patch)
    clean_tensor = image_to_tensor(clean_patch)

    assert rainy_tensor.shape == (3, 256, 256)
    assert clean_tensor.shape == (3, 256, 256)


def test_validation_pair_is_unchanged(real_image_pair):
    rainy, clean = real_image_pair

    result_rainy, result_clean = (
        prepare_validation_pair(rainy, clean)
    )

    assert np.array_equal(
        np.asarray(rainy),
        np.asarray(result_rainy),
    )

    assert np.array_equal(
        np.asarray(clean),
        np.asarray(result_clean),
    )


def test_severity_preprocessing(real_image_pair):
    rainy, _ = real_image_pair

    tensor = prepare_inference_image(
        rainy,
        InferenceConfig(
            task="severity",
            device="cpu",
        ),
    )

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all().item()


def test_restoration_preprocessing(real_image_pair):
    rainy, _ = real_image_pair

    tensor = prepare_inference_image(
        rainy,
        InferenceConfig(
            task="restoration",
            device="cpu",
        ),
    )

    assert tensor.shape == (1, 3, 1000, 1032)
    assert torch.all(tensor >= 0).item()
    assert torch.all(tensor <= 1).item()


def test_cpu_gpu_tensor_consistency(real_image_pair):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    rainy, _ = real_image_pair

    cpu_tensor = prepare_inference_image(
        rainy,
        InferenceConfig(
            task="restoration",
            device="cpu",
        ),
    )

    gpu_tensor = prepare_inference_image(
        rainy,
        InferenceConfig(
            task="restoration",
            device="cuda",
        ),
    )

    assert gpu_tensor.device.type == "cuda"

    assert torch.allclose(
        cpu_tensor,
        gpu_tensor.cpu(),
        atol=1e-6,
    )
