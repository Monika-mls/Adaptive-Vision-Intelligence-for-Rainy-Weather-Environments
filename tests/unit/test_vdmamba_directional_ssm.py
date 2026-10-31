
"""
Unit tests for directional selective SSM processing.

No model weights, datasets, or training required.
"""

import pytest
import torch

from vision.deraining.vdmamba.directional_ssm import (
    DirectionalSelectiveSSM,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 8, 8),
        (1, 8, 7, 11),
        (2, 4, 5, 9),
    ],
)
def test_output_shapes(shape):
    model = DirectionalSelectiveSSM(
        channels=shape[1],
        state_size=4,
    )

    x = torch.randn(*shape)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


def test_backward_gradients():
    torch.manual_seed(42)

    model = DirectionalSelectiveSSM(
        channels=8,
        state_size=4,
    )

    x = torch.randn(
        1, 8, 8, 8,
        requires_grad=True,
    )

    output = model(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(
            parameter.grad
        ).all().item(), name


def test_output_changes_with_input():
    torch.manual_seed(7)

    model = DirectionalSelectiveSSM(
        channels=8,
        state_size=4,
    )
    model.eval()

    x1 = torch.randn(1, 8, 8, 8)
    x2 = x1.clone()
    x2[:, :, 2, 3] += 1.0

    with torch.inference_mode():
        y1 = model(x1)
        y2 = model(x2)

    assert not torch.allclose(y1, y2)


def test_both_directional_modules_exist():
    model = DirectionalSelectiveSSM(
        channels=8,
        state_size=4,
    )

    assert model.horizontal_ssm is not model.vertical_ssm

    horizontal_ids = {
        id(p) for p in model.horizontal_ssm.parameters()
    }

    vertical_ids = {
        id(p) for p in model.vertical_ssm.parameters()
    }

    assert horizontal_ids.isdisjoint(vertical_ids)


def test_invalid_input():
    model = DirectionalSelectiveSSM(
        channels=8,
        state_size=4,
    )

    with pytest.raises(ValueError):
        model(torch.randn(1, 8, 8))

    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 8, 8))

    with pytest.raises(TypeError):
        model(torch.ones(1, 8, 8, 8, dtype=torch.int32))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_forward_backward():
    device = torch.device("cuda")

    model = DirectionalSelectiveSSM(
        channels=8,
        state_size=4,
    ).to(device)

    x = torch.randn(
        1, 8, 32, 32,
        device=device,
        requires_grad=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all().item()

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()

    print(
        "\nPeak PyTorch allocated memory:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        ),
        "MiB",
    )
