"""Tests for experimental RMU; fast input sizes for 2 GB GPUs."""
import pytest
import torch

from vision.deraining.vdmamba.rmu import ExperimentalRMU, ChannelAttention


@pytest.mark.parametrize('shape', [(1, 8, 8, 8), (1, 8, 7, 9), (2, 4, 4, 5)])
def test_shape_and_finite(shape):
    model = ExperimentalRMU(channels=shape[1], state_size=4)
    y = model(torch.randn(*shape))
    assert y.shape == shape
    assert torch.isfinite(y).all().item()


def test_gradient_flow():
    model = ExperimentalRMU(channels=8, state_size=4)
    x = torch.randn(1, 8, 6, 6, requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all().item(), name


def test_identity_when_residual_scales_zero():
    model = ExperimentalRMU(channels=8, state_size=4)
    with torch.no_grad():
        model.gamma1.zero_()
        model.gamma2.zero_()
    x = torch.randn(1, 8, 6, 6)
    assert torch.equal(model(x), x)


def test_attention_shape_and_bounds():
    model = ChannelAttention(8)
    x = torch.ones(1, 8, 5, 7)
    y = model(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0).item() and torch.all(y <= 1).item()


@pytest.mark.parametrize('bad', [0, -1])
def test_invalid_channels(bad):
    with pytest.raises(ValueError):
        ExperimentalRMU(channels=bad)


def test_invalid_input():
    model = ExperimentalRMU(channels=8)
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 8, 8))
    with pytest.raises(TypeError):
        model(torch.ones(1, 8, 8, 8, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_backward():
    model = ExperimentalRMU(channels=8, state_size=4).cuda()
    x = torch.randn(1, 8, 8, 8, device='cuda', requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    print('Peak allocated MiB:', round(torch.cuda.max_memory_allocated()/1024**2, 2))
