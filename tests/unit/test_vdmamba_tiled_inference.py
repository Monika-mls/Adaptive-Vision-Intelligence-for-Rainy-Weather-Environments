
import pytest
import torch
from torch import nn

from vision.deraining.vdmamba.inference_adapter import (
    tile_positions,
    blending_window,
    restore_tiled,
)


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize(
    "length,tile,overlap",
    [
        (64, 128, 32),
        (128, 128, 32),
        (129, 128, 32),
        (256, 128, 32),
        (313, 128, 32),
    ],
)
def test_tile_positions(length, tile, overlap):
    positions = tile_positions(length, tile, overlap)

    assert positions[0] == 0

    if length > tile:
        assert positions[-1] == length - tile

    assert positions == sorted(set(positions))


def test_blending_weights_positive():
    window = blending_window(128)

    assert window.shape == (1, 128, 128)
    assert torch.all(window > 0).item()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 64, 71),
        (1, 3, 128, 128),
        (1, 3, 197, 231),
        (1, 3, 256, 257),
    ],
)
def test_identity_tiled_restoration(shape):
    image = torch.rand(*shape)

    output, tile_count = restore_tiled(
        IdentityModel(),
        image,
        torch.device("cpu"),
        tile_size=128,
        overlap=32,
    )

    assert output.shape == image.shape
    assert tile_count >= 1
    assert torch.allclose(
        output, image, atol=1e-5, rtol=1e-5
    )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        tile_positions(256, 128, 128)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_identity_cuda():
    image = torch.rand(1, 3, 160, 192)

    output, count = restore_tiled(
        IdentityModel().cuda(),
        image,
        torch.device("cuda"),
        tile_size=128,
        overlap=32,
    )

    assert output.shape == image.shape
    assert count > 1
    assert torch.allclose(output, image, atol=1e-5)
