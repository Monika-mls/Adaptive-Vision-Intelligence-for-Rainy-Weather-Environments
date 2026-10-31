"""Tiny-input checks only: these do not assess deraining quality."""
import pytest
import torch
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel, ExperimentalFEM, ExperimentalReconstruction
)

@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (1, 3, 7, 9)])
def test_image_output_shape(shape):
    model = ExperimentalVDMambaImageModel(channels=4, state_size=2)
    with torch.no_grad():
        x = torch.rand(shape)
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


def test_outputs_contract():
    model = ExperimentalVDMambaImageModel(channels=4, state_size=2)
    with torch.no_grad():
        result = model(torch.rand(1, 3, 8, 8), return_details=True)
    assert set(result) == {"prediction", "background", "residue", "vertical", "horizontal"}
    assert result["background"].shape == (1, 4, 4, 4)
    assert result["residue"].shape == result["background"].shape
    assert result["vertical"].shape == result["horizontal"].shape


def test_gradients():
    model = ExperimentalVDMambaImageModel(channels=4, state_size=2)
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all().item(), name


def test_bad_inputs():
    fem = ExperimentalFEM(channels=4, state_size=2)
    with pytest.raises(ValueError):
        fem(torch.rand(1, 1, 8, 8))
    with pytest.raises(TypeError):
        fem(torch.ones(1, 3, 8, 8, dtype=torch.uint8))
    rm = ExperimentalReconstruction(channels=4)
    with pytest.raises(ValueError):
        rm(torch.randn(1, 4, 3, 3), torch.rand(1, 3, 8, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_small_image():
    model = ExperimentalVDMambaImageModel(channels=4, state_size=2).cuda()
    x = torch.rand(1, 3, 8, 8, device="cuda", requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    y = model(x)
    y.square().mean().backward()
    assert y.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    print("peak PyTorch allocation MiB:", round(torch.cuda.max_memory_allocated() / (1024**2), 2))
