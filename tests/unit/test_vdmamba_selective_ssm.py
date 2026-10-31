
"""
Tests for the reference selective state-space block.

CPU/CUDA forward and backward checks.
Causal behavior and input-dependent parameter tests.

No dataset access, training runs or checkpoint changes.
"""

import pytest
import torch

from vision.deraining.vdmamba.selective_ssm import (
    SelectiveSSM,
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 8),
        (2, 32, 16),
        (1, 64, 16),
        (1, 1, 8),
    ],
)
def test_output_shapes(shape):
    batch, length, channels = shape

    model = SelectiveSSM(
        channels=channels,
        state_size=4,
    )

    x = torch.randn(*shape)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


def test_gradients():
    model = SelectiveSSM(
        channels=8,
        state_size=4,
    )

    x = torch.randn(
        2, 16, 8,
        requires_grad=True,
    )

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(
            parameter.grad
        ).all().item()


def test_causal_processing():
    torch.manual_seed(42)

    model = SelectiveSSM(
        channels=8,
        state_size=4,
    )

    model.eval()

    original = torch.randn(1, 12, 8)
    modified = original.clone()

    # Change only positions after token 5.
    modified[:, 6:, :] += 10.0

    with torch.inference_mode():
        y1 = model(original)
        y2 = model(modified)

    # Earlier outputs should not depend
    # on tokens from the future.
    assert torch.allclose(
        y1[:, :6],
        y2[:, :6],
        atol=1e-6,
        rtol=1e-6,
    )


def test_input_dependent_parameters():
    model = SelectiveSSM(
        channels=8,
        state_size=4,
    )

    x1 = torch.randn(1, 12, 8)
    x2 = x1 + 0.5

    with torch.no_grad():
        p1 = model.parameter_projection(
            torch.nn.functional.silu(
                model.depthwise_conv(
                    torch.nn.functional.pad(
                        model.input_projection(x1)
                        .chunk(2, dim=-1)[0]
                        .transpose(1, 2),
                        (model.kernel_size - 1, 0),
                    )
                ).transpose(1, 2)
            )
        )

        p2 = model.parameter_projection(
            torch.nn.functional.silu(
                model.depthwise_conv(
                    torch.nn.functional.pad(
                        model.input_projection(x2)
                        .chunk(2, dim=-1)[0]
                        .transpose(1, 2),
                        (model.kernel_size - 1, 0),
                    )
                ).transpose(1, 2)
            )
        )

    assert not torch.allclose(p1, p2)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (1, 8, 4, 4),
    ],
)
def test_invalid_shapes(shape):
    model = SelectiveSSM(
        channels=8,
        state_size=4,
    )

    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_invalid_channels():
    model = SelectiveSSM(
        channels=8,
        state_size=4,
    )

    with pytest.raises(ValueError):
        model(torch.randn(1, 8, 16))


def test_invalid_configuration():
    with pytest.raises(ValueError):
        SelectiveSSM(channels=0)

    with pytest.raises(ValueError):
        SelectiveSSM(state_size=0)

    with pytest.raises(ValueError):
        SelectiveSSM(kernel_size=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_forward_backward():
    device = torch.device("cuda")

    model = SelectiveSSM(
        channels=16,
        state_size=8,
    ).to(device)

    x = torch.randn(
        1, 64, 16,
        device=device,
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats()

    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()

    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()

    print(
        "\nPeak PyTorch GPU allocation:",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2,
            2,
        ),
        "MiB",
    )
