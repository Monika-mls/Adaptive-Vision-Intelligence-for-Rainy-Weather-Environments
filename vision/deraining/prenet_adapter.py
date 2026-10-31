
"""
PReNet restoration adapter.

Project:
Adaptive Vision Intelligence for Rainy Weather Environments

Uses the original PReNet implementation and pretrained
Rain100L PReNet6 weights.

Does not train or modify the original research model.
Does not access dataset manifests or held-out test images.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRENET_REPO = PROJECT_ROOT / "external_models" / "PReNet"

DEFAULT_CHECKPOINT = (
    PRENET_REPO
    / "logs"
    / "Rain100L"
    / "PReNet6"
    / "net_latest.pth"
)


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda.")

    return torch.device(name)


def import_original_prenet():
    source = PRENET_REPO / "networks.py"

    if not source.is_file():
        raise FileNotFoundError(
            f"Original PReNet source not found: {source}"
        )

    spec = importlib.util.spec_from_file_location(
        "original_prenet_networks",
        source,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import PReNet source.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.PReNet


def load_prenet(checkpoint_path=DEFAULT_CHECKPOINT, device="auto"):
    checkpoint_path = Path(checkpoint_path)
    selected_device = get_device(device)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    PReNet = import_original_prenet()

    model = PReNet(
        recurrent_iter=6,
        use_GPU=(selected_device.type == "cuda"),
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model = model.to(selected_device)
    model.eval()

    return model, selected_device


def load_rgb_image(path, max_side=256):
    """
    Load one RGB image.

    The default max_side=256 is a GPU-feasibility
    safeguard, not the original full-resolution image.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    if max_side < 1:
        raise ValueError("max_side must be positive.")

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    original_size = image.size

    if max(image.size) > max_side:
        image.thumbnail(
            (max_side, max_side),
            Image.Resampling.LANCZOS,
        )

    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    return image, tensor, original_size


def restore_image(
    input_path,
    output_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device="auto",
    max_side=256,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if input_path == output_path:
        raise ValueError(
            "Input and output paths must be different."
        )

    input_image, tensor, original_size = load_rgb_image(
        input_path,
        max_side=max_side,
    )

    model, selected_device = load_prenet(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    tensor = tensor.to(selected_device)

    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        restored, stages = model(tensor)
        restored = restored.clamp(0.0, 1.0)

    if selected_device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if restored.shape != tensor.shape:
        raise RuntimeError(
            "Restoration output shape does not match input."
        )

    if not torch.isfinite(restored).all().item():
        raise RuntimeError(
            "Restoration contains non-finite values."
        )

    output_array = (
        restored[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    output_array = (
        np.clip(output_array * 255.0, 0, 255)
        .round()
        .astype(np.uint8)
    )

    restored_image = Image.fromarray(
        output_array,
        mode="RGB",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    restored_image.save(output_path)

    peak_memory = None

    if selected_device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "device": str(selected_device),
        "original_size": list(original_size),
        "processed_size": list(input_image.size),
        "output_size": list(restored_image.size),
        "recurrent_stages": len(stages),
        "inference_seconds": round(elapsed, 4),
        "peak_gpu_allocated_mib": (
            round(peak_memory, 2)
            if peak_memory is not None
            else None
        ),
        "restoration_completed": True,
        "quality_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restore a rainy image with pretrained PReNet."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    result = restore_image(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_side=args.max_side,
    )

    print("=" * 60)
    print("PRENET RESTORATION RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
