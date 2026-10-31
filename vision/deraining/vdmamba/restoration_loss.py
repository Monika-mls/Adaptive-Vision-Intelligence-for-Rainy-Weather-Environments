
"""
Restoration training losses for the experimental VDMamba model.

Supported losses:
1. Charbonnier loss
2. Differentiable SSIM loss
3. Gradient reconstruction loss

Inputs:
    [B, 3, H, W] floating-point tensors

Expected range:
    [0, 1]

Existing behavior is preserved when gradient_weight=0.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()

        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                "window_size must be odd and >= 3"
            )

        if sigma <= 0:
            raise ValueError(
                "sigma must be positive"
            )

        positions = torch.arange(
            window_size, dtype=torch.float32
        ) - window_size // 2

        gaussian = torch.exp(
            -(positions ** 2) / (2 * sigma ** 2)
        )

        gaussian = gaussian / gaussian.sum()

        window = torch.outer(
            gaussian, gaussian
        )

        window = window / window.sum()

        self.register_buffer(
            "window",
            window[None, None],
        )

        self.window_size = window_size

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                "Image shapes must match"
            )

        if prediction.ndim != 4:
            raise ValueError(
                "Expected [B, C, H, W]"
            )

        if not (
            prediction.is_floating_point()
            and target.is_floating_point()
        ):
            raise TypeError(
                "Images must be floating-point"
            )

        if (
            prediction.device != target.device
            or prediction.dtype != target.dtype
        ):
            raise ValueError(
                "Images must share device and dtype"
            )

        if (
            prediction.shape[-2] < self.window_size
            or prediction.shape[-1] < self.window_size
        ):
            raise ValueError(
                "Image smaller than SSIM window"
            )

        channels = prediction.shape[1]

        window = self.window.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).expand(channels, 1, -1, -1)

        padding = self.window_size // 2

        def filter_image(x):
            x = F.pad(
                x,
                (padding, padding, padding, padding),
                mode="reflect",
            )

            return F.conv2d(
                x,
                window,
                groups=channels,
            )

        mu_x = filter_image(prediction)
        mu_y = filter_image(target)

        mu_x2 = mu_x.square()
        mu_y2 = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x2 = (
            filter_image(prediction.square())
            - mu_x2
        ).clamp_min(0)

        sigma_y2 = (
            filter_image(target.square())
            - mu_y2
        ).clamp_min(0)

        sigma_xy = (
            filter_image(prediction * target)
            - mu_xy
        )

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = (
            (2 * mu_xy + c1)
            * (2 * sigma_xy + c2)
        ) / (
            (mu_x2 + mu_y2 + c1)
            * (sigma_x2 + sigma_y2 + c2)
        )

        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    """
    Compare horizontal and vertical image gradients.

    Encourages the predicted image's local changes
    to match those of the clean target.

    This is not a dedicated rain-mask loss.
    """

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                "Image shapes must match"
            )

        if prediction.ndim != 4:
            raise ValueError(
                "Expected [B, C, H, W]"
            )

        if not (
            prediction.is_floating_point()
            and target.is_floating_point()
        ):
            raise TypeError(
                "Images must be floating-point"
            )

        if (
            prediction.device != target.device
            or prediction.dtype != target.dtype
        ):
            raise ValueError(
                "Images must share device and dtype"
            )

        if (
            prediction.shape[-2] < 2
            or prediction.shape[-1] < 2
        ):
            raise ValueError(
                "Image must be at least 2x2"
            )

        prediction_dx = (
            prediction[:, :, :, 1:]
            - prediction[:, :, :, :-1]
        )

        prediction_dy = (
            prediction[:, :, 1:, :]
            - prediction[:, :, :-1, :]
        )

        target_dx = (
            target[:, :, :, 1:]
            - target[:, :, :, :-1]
        )

        target_dy = (
            target[:, :, 1:, :]
            - target[:, :, :-1, :]
        )

        loss_x = (
            prediction_dx - target_dx
        ).abs().mean()

        loss_y = (
            prediction_dy - target_dy
        ).abs().mean()

        return 0.5 * (loss_x + loss_y)


class RestorationLoss(nn.Module):
    def __init__(
        self,
        ssim_weight=0.02,
        epsilon=1e-3,
        gradient_weight=0.0,
    ):
        super().__init__()

        if not 0 <= ssim_weight <= 1:
            raise ValueError(
                "ssim_weight must be between 0 and 1"
            )

        if not 0 <= gradient_weight <= 1:
            raise ValueError(
                "gradient_weight must be between 0 and 1"
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.epsilon = epsilon

        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                "Image shapes must match"
            )

        charbonnier = torch.sqrt(
            (prediction - target).square()
            + self.epsilon ** 2
        ).mean()

        total_loss = charbonnier

        if self.ssim_weight > 0:
            total_loss = (
                total_loss
                + self.ssim_weight
                * self.ssim_loss(prediction, target)
            )

        if self.gradient_weight > 0:
            total_loss = (
                total_loss
                + self.gradient_weight
                * self.gradient_loss(prediction, target)
            )

        return total_loss
