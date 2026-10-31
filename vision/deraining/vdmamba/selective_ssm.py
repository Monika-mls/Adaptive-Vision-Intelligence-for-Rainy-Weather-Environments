
"""
Reference selective state-space block.

Part of the experimental VDMamba reproduction.

Implements:
- Input-dependent delta, B and C parameters
- Stable negative diagonal state transition
- Causal depthwise 1D convolution
- Recurrent selective state updates
- Skip connection and output gating
- CPU/CUDA autograd support

This is a straightforward PyTorch reference implementation,
not the optimized official Mamba implementation.

No pretrained weights or restoration claims.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SelectiveSSM(nn.Module):
    def __init__(
        self,
        channels=16,
        state_size=8,
        kernel_size=3,
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be positive.")

        if state_size < 1:
            raise ValueError("state_size must be positive.")

        if kernel_size < 1:
            raise ValueError("kernel_size must be positive.")

        self.channels = channels
        self.state_size = state_size
        self.kernel_size = kernel_size

        # Separate processing and gating branches.
        self.input_projection = nn.Linear(
            channels,
            channels * 2,
        )

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=0,
        )

        # Input-dependent delta, B and C.
        self.parameter_projection = nn.Linear(
            channels,
            channels + 2 * state_size,
        )

        # Stable diagonal transition parameters.
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    state_size + 1,
                    dtype=torch.float32,
                )
            ).repeat(channels, 1)
        )

        self.D = nn.Parameter(
            torch.ones(channels)
        )

        self.delta_bias = nn.Parameter(
            torch.full((channels,), -2.0)
        )

        self.output_projection = nn.Linear(
            channels,
            channels,
        )

    def forward(self, x):
        """
        Args:
            x: [batch, length, channels]

        Returns:
            [batch, length, channels]
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a tensor.")

        if x.ndim != 3:
            raise ValueError(
                "Expected [batch, length, channels]."
            )

        if x.shape[-1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels."
            )

        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError(
                "Batch and sequence length must be positive."
            )

        if not x.is_floating_point():
            raise TypeError(
                "Input must contain floating-point values."
            )

        batch, length, channels = x.shape

        projected = self.input_projection(x)

        u, gate = projected.chunk(2, dim=-1)

        # Causal depthwise convolution:
        # left-pad only, preventing future-token access.
        u_conv = u.transpose(1, 2)

        u_conv = F.pad(
            u_conv,
            (self.kernel_size - 1, 0),
        )

        u_conv = self.depthwise_conv(u_conv)

        u = F.silu(
            u_conv.transpose(1, 2)
        )

        parameters = self.parameter_projection(u)

        delta_raw, b_values, c_values = torch.split(
            parameters,
            [
                channels,
                self.state_size,
                self.state_size,
            ],
            dim=-1,
        )

        delta = F.softplus(
            delta_raw + self.delta_bias
        )

        # Negative state transition helps stability.
        A = -torch.exp(self.A_log)

        state = x.new_zeros(
            batch,
            channels,
            self.state_size,
        )

        outputs = []

        for index in range(length):
            dt = delta[:, index, :].unsqueeze(-1)

            current_input = u[:, index, :]

            current_B = (
                b_values[:, index, :]
                .unsqueeze(1)
            )

            current_C = (
                c_values[:, index, :]
                .unsqueeze(1)
            )

            # Zero-order-hold transition for A.
            discrete_A = torch.exp(
                dt * A.unsqueeze(0)
            )

            # Reference input update using dt * B * u.
            # This is a common selective-scan
            # discretization approximation.
            discrete_input = (
                dt
                * current_B
                * current_input.unsqueeze(-1)
            )

            state = (
                discrete_A * state
                + discrete_input
            )

            output = (
                state * current_C
            ).sum(dim=-1)

            output = (
                output
                + self.D * current_input
            )

            outputs.append(output)

        y = torch.stack(outputs, dim=1)

        y = y * F.silu(gate)

        return self.output_projection(y)
