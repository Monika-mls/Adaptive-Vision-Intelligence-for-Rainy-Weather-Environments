
"""
Tiled inference for the experimental VDMamba-inspired model.

Features:
- Loads an existing trained project checkpoint
- Accepts external rainy RGB photographs
- Processes overlapping image tiles
- Blends tile predictions to reduce seams
- Preserves original image dimensions
- Saves restored PNG and JSON metadata
- Works with CPU or CUDA

No training.
Does not modify checkpoints.
Does not use validation clean targets.

IMPORTANT:
This is our experimental VDMamba-inspired architecture,
not the official published VDMamba implementation.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
)

from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


DEFAULT_CHECKPOINT = (
    "outputs/vdmamba/multi_train/"
    "run_20261028_093414_860421/best.pt"
)


def tile_positions(length, tile_size, overlap):
    """Return tile start positions covering one image axis."""
    if length < 1:
        raise ValueError("Image dimension must be positive.")

    if tile_size < 1:
        raise ValueError("Tile size must be positive.")

    if not 0 <= overlap < tile_size:
        raise ValueError(
            "Overlap must be nonnegative and smaller than tile size."
        )

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap

    positions = list(
        range(0, length - tile_size + 1, stride)
    )

    final = length - tile_size

    if positions[-1] != final:
        positions.append(final)

    return positions


def blending_window(tile_size):
    """
    Positive feathering window.

    Positive weights at tile boundaries prevent
    division by zero along image borders.
    """
    if tile_size < 2:
        raise ValueError("Tile size must be >= 2.")

    coordinates = torch.arange(
        tile_size, dtype=torch.float32
    )

    window_1d = 0.5 - 0.5 * torch.cos(
        2 * math.pi * (coordinates + 0.5) / tile_size
    )

    window_1d = window_1d.clamp_min(0.05)

    window_2d = torch.outer(window_1d, window_1d)

    return window_2d.unsqueeze(0)


def restore_tiled(
    model,
    image,
    device,
    tile_size=128,
    overlap=32,
):
    """
    Restore a tensor in [1, 3, H, W] format.

    Image remains on CPU; only individual tiles
    are sent to the model's device.

    Returns:
        restored tensor on CPU, shape [1, 3, H, W]
        number of processed tiles
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected [1, 3, H, W].")

    if image.shape[1] != 3:
        raise ValueError("Expected RGB image.")

    if not image.is_floating_point():
        raise TypeError("Image must be floating-point.")

    if image.device.type != "cpu":
        raise ValueError("Input image must be on CPU.")

    if not 0 <= overlap < tile_size:
        raise ValueError("Invalid tile overlap.")

    height, width = image.shape[-2:]

    if height < 1 or width < 1:
        raise ValueError("Invalid image dimensions.")

    # We pad small images for tiled inference,
    # then crop the output back to original size.
    pad_h = max(0, tile_size - height)
    pad_w = max(0, tile_size - width)

    padded = F.pad(
        image,
        (0, pad_w, 0, pad_h),
        mode="replicate",
    )

    padded_h, padded_w = padded.shape[-2:]

    y_positions = tile_positions(
        padded_h, tile_size, overlap
    )
    x_positions = tile_positions(
        padded_w, tile_size, overlap
    )

    # Accumulate on CPU to conserve GPU memory.
    accumulated = torch.zeros(
        (1, 3, padded_h, padded_w),
        dtype=torch.float32,
    )

    weights = torch.zeros(
        (1, 1, padded_h, padded_w),
        dtype=torch.float32,
    )

    window = blending_window(tile_size)

    tiles_processed = 0

    model.eval()

    with torch.inference_mode():
        for top in y_positions:
            for left in x_positions:
                tile = padded[
                    :,
                    :,
                    top:top + tile_size,
                    left:left + tile_size,
                ].to(device)

                prediction = model(tile)

                if prediction.shape != tile.shape:
                    raise RuntimeError(
                        "Model output size differs from input tile."
                    )

                if not torch.isfinite(
                    prediction
                ).all().item():
                    raise RuntimeError(
                        "Model produced non-finite pixels."
                    )

                prediction = (
                    prediction.clamp(0, 1)
                    .float()
                    .cpu()
                )

                accumulated[
                    :,
                    :,
                    top:top + tile_size,
                    left:left + tile_size,
                ] += prediction * window

                weights[
                    :,
                    :,
                    top:top + tile_size,
                    left:left + tile_size,
                ] += window

                tiles_processed += 1

                del tile, prediction

    if torch.any(weights <= 0).item():
        raise RuntimeError(
            "Incomplete image coverage during tiling."
        )

    restored = accumulated / weights

    restored = restored[
        :,
        :,
        :height,
        :width,
    ].clamp(0, 1)

    return restored, tiles_processed


def load_model(checkpoint_path, device):
    """Load the project's trained experimental model."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("model_status") != (
        "experimental_vdmamba_inspired"
    ):
        raise ValueError(
            "Checkpoint is not marked as the "
            "expected experimental model."
        )

    config = checkpoint["config"]

    model = ExperimentalVDMambaImageModel(
        channels=int(config["channels"]),
        state_size=int(config["state_size"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, config, checkpoint.get("epoch")


def restore_image(
    input_path,
    output_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device_name="auto",
    tile_size=128,
    overlap=32,
):
    """Restore an external photograph and save results."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu, cuda or auto.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if tile_size < 8 or tile_size % 2:
        raise ValueError(
            "Tile size must be an even integer >= 8."
        )

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid overlap.")

    device = torch.device(device_name)

    model, config, checkpoint_epoch = load_model(
        checkpoint_path, device
    )

    with Image.open(input_path) as opened:
        original = opened.convert("RGB")

    width, height = original.size

    image = (
        pil_to_tensor(original)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    restored, tile_count = restore_tiled(
        model=model,
        image=image,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    inference_seconds = time.perf_counter() - start

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    restored_image = to_pil_image(restored[0])
    restored_image.save(output_path, format="PNG")

    peak_memory = (
        round(
            torch.cuda.max_memory_allocated(device)
            / 1024**2,
            2,
        )
        if device.type == "cuda"
        else None
    )

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": "ExperimentalVDMambaImageModel",
        "model_status": "experimental_vdmamba_inspired",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_dimensions": [width, height],
        "output_dimensions": list(restored_image.size),
        "tile_size": tile_size,
        "overlap": overlap,
        "tiles_processed": tile_count,
        "device": device_name,
        "inference_seconds": round(
            inference_seconds, 4
        ),
        "peak_pytorch_allocated_mib": peak_memory,
        "quality_evaluated": False,
        "trained_on_patch_size": config.get(
            "patch_size"
        ),
        "notes": (
            "Tiled inference preserves output dimensions, "
            "not necessarily visual quality. "
            "No clean reference is available."
        ),
    }

    report_path = output_path.with_suffix(".json")

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    report = restore_image(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
