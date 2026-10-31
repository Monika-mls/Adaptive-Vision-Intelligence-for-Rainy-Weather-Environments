
"""
Full-resolution tiled inference for fine-tuned Restormer.

- Uses official Restormer architecture.
- Loads our 784-pair fine-tuned checkpoint.
- Runs one tile at a time on CUDA.
- Blends overlapping tiles to reduce seams.
- Preserves original image resolution.
- Does not use training or test datasets.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from vision.deraining.restormer_adapter import load_model


DEFAULT_CHECKPOINT = Path(
    "outputs/restormer/finetune/"
    "20261030_210003_966091/best.pt"
)


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))

    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def blend_weights(height, width, y, x, full_h, full_w, device):
    """Feather tile boundaries without fading image boundaries."""

    def axis_weights(length, at_start, at_end):
        weight = torch.ones(length, device=device)
        ramp = min(24, length // 4)

        if ramp > 0:
            if not at_start:
                weight[:ramp] = torch.linspace(
                    0.01, 1.0, ramp, device=device
                )

            if not at_end:
                weight[-ramp:] = torch.linspace(
                    1.0, 0.01, ramp, device=device
                )

        return weight

    wy = axis_weights(
        height,
        y == 0,
        y + height == full_h,
    )

    wx = axis_weights(
        width,
        x == 0,
        x + width == full_w,
    )

    return wy[:, None] * wx[None, :]


def restore_image(
    input_path,
    output_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device_name="auto",
    tile_size=128,
    overlap=32,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if tile_size < 32 or tile_size % 8:
        raise ValueError(
            "tile_size must be at least 32 and divisible by 8"
        )

    if overlap < 24 or overlap >= tile_size:
        raise ValueError(
            "overlap must be at least 24 and less than tile_size"
        )

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    width, height = image.size

    pixels = np.asarray(image, dtype=np.float32) / 255.0

    # Keep the full source image and accumulation on CPU.
    input_tensor = (
        torch.from_numpy(pixels.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    print("Loading fine-tuned Restormer...", flush=True)

    model = load_model(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["params"],
        strict=True,
    )

    model.eval()

    del checkpoint

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)

    total_tiles = len(ys) * len(xs)

    accumulated = torch.zeros(
        (3, height, width),
        dtype=torch.float32,
    )

    weights_sum = torch.zeros(
        (1, height, width),
        dtype=torch.float32,
    )

    print("Device:", device, flush=True)
    print("Image dimensions:", width, "x", height, flush=True)
    print("Tile size:", tile_size, flush=True)
    print("Overlap:", overlap, flush=True)
    print("Total tiles:", total_tiles, flush=True)

    start = time.perf_counter()
    tile_number = 0

    with torch.inference_mode():
        for y in ys:
            for x in xs:
                tile_number += 1

                end_y = min(y + tile_size, height)
                end_x = min(x + tile_size, width)

                tile_h = end_y - y
                tile_w = end_x - x

                tile = input_tensor[
                    :, :, y:end_y, x:end_x
                ].to(device)

                pad_h = (-tile_h) % 8
                pad_w = (-tile_w) % 8

                if pad_h or pad_w:
                    tile = F.pad(
                        tile,
                        (0, pad_w, 0, pad_h),
                        mode="replicate",
                    )

                restored = model(tile)

                restored = restored[
                    :, :, :tile_h, :tile_w
                ].clamp(0, 1)

                if not torch.isfinite(restored).all().item():
                    raise RuntimeError(
                        f"Non-finite output in tile {tile_number}"
                    )

                restored_cpu = restored[0].cpu()

                weight = blend_weights(
                    tile_h,
                    tile_w,
                    y,
                    x,
                    height,
                    width,
                    device="cpu",
                )

                accumulated[
                    :, y:end_y, x:end_x
                ] += restored_cpu * weight

                weights_sum[
                    :, y:end_y, x:end_x
                ] += weight

                if (
                    tile_number == 1
                    or tile_number % 10 == 0
                    or tile_number == total_tiles
                ):
                    print(
                        f"Processed {tile_number}/{total_tiles} tiles",
                        flush=True,
                    )

                del tile, restored, restored_cpu

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = time.perf_counter() - start

    if (weights_sum <= 0).any().item():
        raise RuntimeError("Some pixels received zero blending weight")

    result = (
        accumulated / weights_sum
    ).clamp(0, 1)

    output_pixels = (
        result.permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(output_pixels).save(
        output_path,
        format="PNG",
    )

    peak_gpu_mib = (
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        )
        if device.type == "cuda"
        else None
    )

    report = {
        "model": "Fine-tuned Restormer",
        "model_status": "official_restormer_output_layer_finetuned",
        "checkpoint": str(checkpoint_path),
        "original_size": [width, height],
        "output_size": [width, height],
        "device": str(device),
        "tile_size": tile_size,
        "overlap": overlap,
        "tiles_processed": total_tiles,
        "inference_seconds": round(inference_seconds, 4),
        "peak_gpu_allocated_mib": peak_gpu_mib,
        "output_path": str(output_path),
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
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
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

    print("\nRESTORMER INFERENCE COMPLETE")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
