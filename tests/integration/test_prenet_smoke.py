
"""
Phase 4: PReNet pretrained model smoke test.

Tests:
1. Original architecture import.
2. Exact checkpoint compatibility.
3. CPU forward pass.
4. GPU forward pass if CUDA is available.
5. Output shape, finite values and memory usage.

No dataset images used.
No training performed.
No checkpoints modified.
"""

import importlib.util
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPO_ROOT = PROJECT_ROOT / "external_models" / "PReNet"

CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "Rain100L"
    / "PReNet6"
    / "net_latest.pth"
)


def load_original_prenet_class():
    source = REPO_ROOT / "networks.py"

    if not source.is_file():
        raise FileNotFoundError(source)

    spec = importlib.util.spec_from_file_location(
        "original_prenet_networks",
        source,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.PReNet


import pytest

@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_checkpoint_and_inference(device_name):
    print("\n" + "=" * 60)
    print(f"PRENET SMOKE TEST: {device_name.upper()}")
    print("=" * 60)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    PReNet = load_original_prenet_class()

    device = torch.device(device_name)

    model = PReNet(
        recurrent_iter=6,
        use_GPU=(device.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    # Strict loading ensures every expected parameter
    # matches the original architecture.
    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    print("Checkpoint compatibility: PASSED")

    model = model.to(device)
    model.eval()

    # Small synthetic RGB image.
    # No RealRain test or validation images accessed.
    torch.manual_seed(42)

    image = torch.rand(
        1, 3, 128, 128,
        dtype=torch.float32,
    ).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output, stages = model(image)
        output = output.clamp(0.0, 1.0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    assert output.shape == image.shape
    assert len(stages) == 6
    assert torch.isfinite(output).all().item()
    assert output.min().item() >= 0
    assert output.max().item() <= 1

    print("Input shape:", tuple(image.shape))
    print("Output shape:", tuple(output.shape))
    print("Recurrent stages:", len(stages))
    print("Output finite:", True)
    print("Output range:", (
        round(output.min().item(), 4),
        round(output.max().item(), 4),
    ))
    print(f"Inference time: {elapsed:.4f} seconds")

    if device.type == "cuda":
        peak = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
        print(f"Peak GPU memory: {peak:.2f} MiB")

    print("RESULT: PASSED")


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-v", str(Path(__file__).resolve())])
    )