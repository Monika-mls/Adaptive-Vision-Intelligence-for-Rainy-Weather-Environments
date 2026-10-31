
"""Detailed validation of an experimental VDMamba-inspired checkpoint.

Uses only RealRain-1k Heavy validation images.
Computes per-patch MSE/PSNR and pooled PSNR.
Exports CSV and before/after comparisons.
Does not modify model weights or original checkpoints.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


def psnr_from_mse(mse):
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def make_comparison(rainy, restored, clean, path):
    images = [
        to_pil_image(x.detach().float().cpu().clamp(0, 1))
        .convert("RGB")
        for x in (rainy, restored, clean)
    ]

    w, h = images[0].size
    label_height = 25

    canvas = Image.new(
        "RGB", (w * 3, h + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)

    for i, (image, label) in enumerate(
        zip(images, ("Rainy", "Restored", "Clean"))
    ):
        canvas.paste(image, (i * w, label_height))
        draw.text((i * w + 4, 5), label, fill="black")

    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    device = torch.device(args.device)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]

    patch_size = int(config["patch_size"])

    model = ExperimentalVDMambaImageModel(
        channels=int(config["channels"]),
        state_size=int(config["state_size"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    model.eval()

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=patch_size,
        augment=False,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        Path("outputs/vdmamba/detailed_validation")
        / f"run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    total_rainy_mse = 0.0
    total_restored_mse = 0.0
    total_pixels = 0

    with torch.inference_mode():
        for sample in dataset:
            rainy = sample["rainy"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)
            restored = model(rainy).clamp(0, 1)

            if not torch.isfinite(restored).all().item():
                raise RuntimeError(
                    f"Invalid output: {sample['image_id']}"
                )

            rainy_mse = (
                (rainy - clean).float().square().mean().item()
            )
            restored_mse = (
                (restored - clean).float().square().mean().item()
            )

            rainy_psnr = psnr_from_mse(rainy_mse)
            restored_psnr = psnr_from_mse(restored_mse)

            gain = (
                restored_psnr - rainy_psnr
                if math.isfinite(rainy_psnr)
                else float("-inf")
            )

            image_id = str(sample["image_id"])
            improved = restored_mse < rainy_mse

            rows.append({
                "image_id": image_id,
                "rainy_mse": rainy_mse,
                "restored_mse": restored_mse,
                "rainy_psnr": rainy_psnr,
                "restored_psnr": restored_psnr,
                "psnr_gain": gain,
                "improved": improved,
                "perfect_input_crop": rainy_mse == 0,
            })

            pixels = clean.numel()
            total_rainy_mse += rainy_mse * pixels
            total_restored_mse += restored_mse * pixels
            total_pixels += pixels

            if not improved:
                make_comparison(
                    rainy[0],
                    restored[0],
                    clean[0],
                    output_dir / f"failure_{image_id}.png",
                )

    rainy_pooled = psnr_from_mse(
        total_rainy_mse / total_pixels
    )
    restored_pooled = psnr_from_mse(
        total_restored_mse / total_pixels
    )

    with (output_dir / "per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    failures = sorted(
        (r for r in rows if not r["improved"]),
        key=lambda r: r["restored_mse"] - r["rainy_mse"],
        reverse=True,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "validation_count": len(rows),
        "patch_size": patch_size,
        "pooled_rainy_psnr": rainy_pooled,
        "pooled_restored_psnr": restored_pooled,
        "pooled_psnr_gain": restored_pooled - rainy_pooled,
        "images_improved": sum(r["improved"] for r in rows),
        "images_not_improved": len(failures),
        "perfect_input_crops": [
            r["image_id"]
            for r in rows if r["perfect_input_crop"]
        ],
        "failure_ids": [
            r["image_id"] for r in failures
        ],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DETAILED HEAVY-RAIN VALIDATION")
    print("=" * 60)
    print("Pooled rainy PSNR:", round(rainy_pooled, 4))
    print("Pooled restored PSNR:", round(restored_pooled, 4))
    print(
        "PSNR gain:",
        round(restored_pooled - rainy_pooled, 4),
        "dB",
    )
    print("Images improved:", summary["images_improved"])
    print("Images not improved:", len(failures))
    print("Failures:", summary["failure_ids"])
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
