
"""
Directional sequence-layout unit tests.

No datasets, checkpoint files, or training.
Tests reversibility, shape preservation, gradients,
and CUDA compatibility.
"""

import pytest
import torch

from vision.deraining.vdmamba.directional import (
    DirectionalFeatureLayout,
    horizontal_sequences,
    vertical_sequences,
    restore_horizontal,
    restore_vertical,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 64, 64),
        (1, 16, 32, 48),
        (2, 8, 17, 21),
        (1, 1, 1, 1),
    ],
)
def test_directional_roundtrip(shape):
    x = torch.randn(*shape)

    h = horizontal_sequences(x)
    v = vertical_sequences(x)

    restored_h = restore_horizontal(h, shape)
    restored_v = restore_vertical(v, shape)

    assert torch.equal(restored_h, x)
    assert torch.equal(restored_v, x)


def test_sequence_shapes():
    x = torch.randn(2, 16, 32, 48)

    h = horizontal_sequences(x)
    v = vertical_sequences(x)

    assert h.shape == (64, 48, 16)
    assert v.shape == (96, 32, 16)


def test_directional_module():
    x = torch.randn(1, 8, 32, 48)

    module = DirectionalFeatureLayout()

    horizontal, vertical = module(x)

    reconstructed = module.reconstruct(
        horizontal, vertical, x.shape
    )

    assert reconstructed.shape == x.shape
    assert torch.allclose(
        reconstructed, x, atol=1e-6
    )


def test_gradient_flow():
    x = torch.randn(
        1, 8, 16, 16,
        requires_grad=True,
    )

    module = DirectionalFeatureLayout()

    h, v = module(x)

    reconstructed = module.reconstruct(
        h, v, x.shape
    )

    loss = reconstructed.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_shape():
    x = torch.randn(3, 32, 32)

    with pytest.raises(ValueError):
        horizontal_sequences(x)


def test_shape_mismatch():
    sequences = torch.randn(16, 16, 8)

    with pytest.raises(ValueError):
        restore_horizontal(
            sequences, (1, 8, 32, 32)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_directional_processing():
    device = torch.device("cuda")

    x = torch.randn(
        1, 16, 128, 128,
        device=device,
        requires_grad=True,
    )

    module = DirectionalFeatureLayout()

    torch.cuda.reset_peak_memory_stats()

    horizontal, vertical = module(x)

    restored = module.reconstruct(
        horizontal, vertical, x.shape
    )

    assert torch.allclose(
        restored, x, atol=1e-6
    )

    restored.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print(
        "\nPeak PyTorch GPU allocation:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        ),
        "MiB",
    )
