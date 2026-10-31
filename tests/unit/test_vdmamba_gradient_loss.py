
import pytest
import torch

from vision.deraining.vdmamba.restoration_loss import (
    GradientLoss,
    RestorationLoss,
)


def test_gradient_loss_identical_images():
    image = torch.rand(1, 3, 16, 16)

    loss = GradientLoss()(image, image)

    assert torch.allclose(
        loss,
        torch.zeros_like(loss),
        atol=1e-7,
    )


def test_gradient_loss_constant_brightness_shift():
    target = torch.rand(1, 3, 16, 16) * 0.5

    prediction = target + 0.2

    loss = GradientLoss()(
        prediction,
        target,
    )

    # A uniform brightness shift does not
    # change horizontal or vertical gradients.
    assert loss.item() < 1e-6


def test_gradient_loss_detects_local_difference():
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()

    prediction[:, :, 5:10, 5:10] = 0.5

    loss = GradientLoss()(
        prediction,
        target,
    )

    assert loss.item() > 0


def test_gradient_loss_is_differentiable():
    prediction = torch.rand(
        1, 3, 16, 16,
        requires_grad=True,
    )

    target = torch.rand(1, 3, 16, 16)

    loss = GradientLoss()(
        prediction,
        target,
    )

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all().item()

    assert prediction.grad.abs().sum().item() > 0


def test_combined_restoration_loss_backward():
    prediction = torch.rand(
        1, 3, 16, 16,
        requires_grad=True,
    )

    target = torch.rand(1, 3, 16, 16)

    criterion = RestorationLoss(
        ssim_weight=0.02,
        gradient_weight=0.05,
    )

    loss = criterion(
        prediction,
        target,
    )

    assert torch.isfinite(loss).item()

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all().item()


def test_zero_gradient_weight_preserves_old_loss():
    prediction = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)

    old_behavior = RestorationLoss(
        ssim_weight=0.02
    )(prediction, target)

    new_behavior = RestorationLoss(
        ssim_weight=0.02,
        gradient_weight=0.0,
    )(prediction, target)

    assert torch.allclose(
        old_behavior,
        new_behavior,
        atol=1e-7,
    )


def test_invalid_gradient_weight():
    with pytest.raises(ValueError):
        RestorationLoss(
            gradient_weight=-0.1
        )

    with pytest.raises(ValueError):
        RestorationLoss(
            gradient_weight=1.5
        )


def test_gradient_loss_invalid_shape():
    with pytest.raises(ValueError):
        GradientLoss()(
            torch.rand(1, 3, 16, 16),
            torch.rand(1, 3, 12, 12),
        )
