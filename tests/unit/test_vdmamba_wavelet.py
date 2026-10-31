
"""
Tests for the VDMamba Haar wavelet building block.

No datasets, downloads, checkpoints, or training.
"""

import pytest
import torch

from vision.deraining.vdmamba.wavelet import (
    HaarDWT,
    HaarIDWT,
    wavelet_roundtrip,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 64, 64),
        (1, 3, 65, 63),
        (2, 8, 32, 48),
        (1, 1, 1, 1),
    ],
)
def test_roundtrip_cpu(shape):
    x = torch.randn(*shape)

    restored = wavelet_roundtrip(x)

    assert restored.shape == x.shape

    assert torch.allclose(
        restored,
        x,
        atol=1e-5,
        rtol=1e-5,
    )


def test_subband_shapes():
    x = torch.randn(1, 16, 65, 63)

    ll, lh, hl, hh, size = HaarDWT()(x)

    assert size == (65, 63)

    for band in (ll, lh, hl, hh):
        assert band.shape == (1, 16, 33, 32)


def test_gradient_flow():
    x = torch.randn(
        1, 3, 32, 32,
        requires_grad=True,
    )

    restored = wavelet_roundtrip(x)

    loss = restored.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_input():
    with pytest.raises(ValueError):
        HaarDWT()(torch.randn(3, 32, 32))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_roundtrip_and_gradients():
    device = torch.device("cuda")

    x = torch.randn(
        1, 16, 128, 128,
        device=device,
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats()

    restored = wavelet_roundtrip(x)

    error = (
        restored.detach() - x.detach()
    ).abs().max().item()

    assert error < 1e-5

    restored.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("\nCUDA wavelet reconstruction error:", error)

    print(
        "Peak PyTorch GPU memory:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        ),
        "MiB",
    )
