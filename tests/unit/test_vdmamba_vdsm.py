"""Unit tests for experimental VDSM feature block (not a trained derainer)."""
import pytest
import torch
from vision.deraining.vdmamba.vdsm import ExperimentalVDSM


@pytest.mark.parametrize("shape", [(1, 4, 4, 6), (1, 8, 5, 7)])
def test_shapes_and_residual_identity(shape):
    model = ExperimentalVDSM(channels=shape[1], state_size=2)
    x = torch.randn(*shape)
    result = model(x)
    assert set(result) == {"background", "residue", "vertical", "horizontal"}
    for y in result.values():
        assert y.shape == x.shape
        assert torch.isfinite(y).all().item()
    assert torch.allclose(result["background"] + result["residue"], x, atol=1e-6)


def test_backward_reaches_all_parameters():
    torch.manual_seed(5)
    model = ExperimentalVDSM(channels=4, state_size=2)
    x = torch.randn(1, 4, 4, 5, requires_grad=True)
    out = model(x)
    # Force both streams, fusion, and input to participate in the loss.
    loss = (out["background"].square().mean() +
            0.1 * out["vertical"].square().mean() +
            0.1 * out["horizontal"].square().mean())
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all().item(), name


def test_unique_directional_branches():
    model = ExperimentalVDSM(4, 2)
    left = {id(p) for p in model.vertical_refine.parameters()}
    right = {id(p) for p in model.horizontal_refine.parameters()}
    assert left.isdisjoint(right)


def test_zero_residue_scale_restores_input_exactly():
    model = ExperimentalVDSM(4, 2)
    with torch.no_grad():
        model.residue_scale.zero_()
    x = torch.randn(1, 4, 4, 4)
    output = model(x)
    assert torch.equal(output["background"], x)
    assert torch.count_nonzero(output["residue"]).item() == 0


def test_validation_errors():
    with pytest.raises(ValueError):
        ExperimentalVDSM(channels=0)
    model = ExperimentalVDSM(4, 2)
    with pytest.raises(TypeError):
        model(None)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 4, 4))
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 4))
    with pytest.raises(TypeError):
        model(torch.ones(1, 4, 4, 4, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_backward_and_memory():
    model = ExperimentalVDSM(4, 2).cuda()
    x = torch.randn(1, 4, 8, 8, device="cuda", requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    output["background"].square().mean().backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    print("Peak PyTorch allocated MiB:", round(torch.cuda.max_memory_allocated() / 2**20, 2))
