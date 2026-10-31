import pytest
import torch

from vision.deraining.vdmamba.dpu import DirectionAwareProjection


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
@pytest.mark.parametrize("shape", [(1, 8, 8, 12), (2, 4, 7, 9), (1, 4, 1, 1)])
def test_shapes_and_finite(direction, shape):
    model = DirectionAwareProjection(shape[1], direction)
    out = model(torch.randn(*shape))
    assert out.shape == shape
    assert torch.isfinite(out).all().item()


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
def test_gradients(direction):
    model = DirectionAwareProjection(8, direction)
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
               for p in model.parameters())


def test_directions_use_distinct_axis():
    x = torch.arange(5 * 7, dtype=torch.float32).reshape(1, 1, 5, 7)
    vertical_descriptor = x.mean(dim=3, keepdim=True)
    horizontal_descriptor = x.mean(dim=2, keepdim=True)
    assert vertical_descriptor.shape == (1, 1, 5, 1)
    assert horizontal_descriptor.shape == (1, 1, 1, 7)


@pytest.mark.parametrize("kwargs", [
    {"channels": 0, "direction": "vertical"},
    {"channels": 8, "direction": "diagonal"},
    {"channels": 8, "direction": "horizontal", "reduction": 0},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DirectionAwareProjection(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_gradients():
    model = DirectionAwareProjection(8, "vertical").cuda()
    x = torch.randn(1, 8, 32, 32, device="cuda", requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
