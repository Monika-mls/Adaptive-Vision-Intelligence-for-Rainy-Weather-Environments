
"""
Phase 6: VDMamba preliminary state-space feasibility test.

This is a simplified, differentiable state-space
recurrence implemented in pure PyTorch.

IMPORTANT:
- This is NOT the complete Mamba block.
- This is NOT the VDMamba architecture.
- It does not perform image deraining.
- It does not load or modify any checkpoints.

Purpose:
Verify sequence processing, backward gradients,
and GPU memory on the current Windows laptop.
"""

import time

import pytest
import torch
from torch import nn


class SimpleStateSpace(nn.Module):
    def __init__(self, channels=16, state_size=8):
        super().__init__()

        self.channels = channels
        self.state_size = state_size

        self.log_decay = nn.Parameter(
            torch.zeros(channels, state_size)
        )

        self.input_projection = nn.Linear(
            channels,
            channels * state_size,
        )

        self.output_projection = nn.Linear(
            channels * state_size,
            channels,
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                "Expected [batch, length, channels]."
            )

        batch, length, _ = x.shape

        state = x.new_zeros(
            batch,
            self.channels,
            self.state_size,
        )

        decay = torch.sigmoid(
            self.log_decay
        ).unsqueeze(0)

        outputs = []

        for index in range(length):
            current = x[:, index]

            projected = self.input_projection(
                current
            ).reshape(
                batch,
                self.channels,
                self.state_size,
            )

            state = (
                decay * state + projected
            )

            output = self.output_projection(
                state.reshape(batch, -1)
            )

            outputs.append(output)

        return torch.stack(outputs, dim=1)


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_state_space_forward_backward(device_name):
    device = torch.device(device_name)

    model = SimpleStateSpace(
        channels=16,
        state_size=8,
    ).to(device)

    x = torch.randn(
        1, 64, 16,
        device=device,
        requires_grad=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all().item()

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(
            parameter.grad
        ).all().item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nDevice: {device}")
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Loss:", round(loss.item(), 6))
    print("Forward + backward:", round(elapsed, 4), "s")

    if device.type == "cuda":
        print(
            "Peak PyTorch allocated memory:",
            round(
                torch.cuda.max_memory_allocated()
                / 1024**2,
                2,
            ),
            "MiB",
        )

    print("RESULT: PASSED")
