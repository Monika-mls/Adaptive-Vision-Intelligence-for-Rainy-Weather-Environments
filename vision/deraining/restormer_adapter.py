
"""
Official pretrained Restormer deraining inference test.

Uses the original architecture and checkpoint.
Small preview inference for low-memory NVIDIA GPUs.

Does not train or modify any model.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]

REPO = ROOT / "external_models" / "Restormer"

ARCHITECTURE = (
    REPO / "basicsr" / "models" / "archs"
    / "restormer_arch.py"
)

CHECKPOINT = (
    REPO / "Deraining" / "pretrained_models"
    / "deraining.pth"
)


def load_model(device):
    if not ARCHITECTURE.is_file():
        raise FileNotFoundError(ARCHITECTURE)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    spec = importlib.util.spec_from_file_location(
        "official_restormer_arch",
        ARCHITECTURE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load official Restormer architecture."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Matches the official Deraining_Restormer.yml.
    model = module.Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["params"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


def restore(input_path, output_path, max_side=128):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    model = load_model(device)

    print(
        "Official Restormer weights loaded.",
        flush=True,
    )

    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

    original_size = image.size

    # This is a preview, not full-resolution inference.
    image.thumbnail(
        (max_side, max_side),
        Image.Resampling.LANCZOS,
    )

    resized_size = image.size

    array = np.asarray(
        image, dtype=np.float32
    ).copy() / 255.0

    tensor = torch.from_numpy(array).permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    height, width = tensor.shape[-2:]

    pad_height = (-height) % 8
    pad_width = (-width) % 8

    # Replicate padding is safe for small image dimensions.
    tensor = F.pad(
        tensor,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    prediction = prediction[
        :, :, :height, :width
    ].clamp(0, 1)

    if not torch.isfinite(
        prediction
    ).all().item():
        raise RuntimeError(
            "Restormer generated non-finite values."
        )

    result = (
        prediction[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    result = np.uint8(
        np.clip(
            np.round(result * 255),
            0, 255,
        )
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(result).save(
        output_path,
        format="PNG",
    )

    print("Original size:", original_size)
    print("Preview size:", resized_size)
    print("Inference time:", round(elapsed, 4), "s")

    if device.type == "cuda":
        print(
            "Peak GPU allocated:",
            round(
                torch.cuda.max_memory_allocated()
                / 1024**2,
                2,
            ),
            "MiB",
        )

    print("Restored image:", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if args.max_side < 32:
        parser.error(
            "--max-side must be at least 32"
        )

    restore(
        args.input,
        args.output,
        args.max_side,
    )


if __name__ == "__main__":
    main()
