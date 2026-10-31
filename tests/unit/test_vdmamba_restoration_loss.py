
import pytest
import torch

from vision.deraining.vdmamba.restoration_loss import (
    SSIMLoss,
    RestorationLoss,
)


def test_ssim_identical_images():
    image = torch.rand(1, 3, 16, 16)
    loss = SSIMLoss()(image, image)
    assert abs(loss.item()) < 1e-5


def test_ssim_different_images():
    x = torch.zeros(1, 3, 16, 16)
    y = torch.ones(1, 3, 16, 16)

    assert SSIMLoss()(x, y).item() > 0


def test_combined_loss_gradients():
    prediction = torch.rand(
        1, 3, 16, 16, requires_grad=True
    )
    target = torch.rand(1, 3, 16, 16)

    loss = RestorationLoss(0.02)(
        prediction, target
    )
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all().item()


def test_zero_weight_is_charbonnier():
    x = torch.rand(1, 3, 16, 16)
    y = torch.rand(1, 3, 16, 16)

    expected = torch.sqrt(
        (x - y).square() + 1e-6
    ).mean()

    actual = RestorationLoss(0)(x, y)

    assert torch.allclose(actual, expected)


def test_invalid_weight():
    with pytest.raises(ValueError):
        RestorationLoss(-0.1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_cuda_combined_loss():
    x = torch.rand(
        1, 3, 64, 64,
        device="cuda",
        requires_grad=True,
    )
    y = torch.rand(1, 3, 64, 64, device="cuda")

    loss = RestorationLoss(0.02).cuda()(x, y)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(x.grad).all().item()
