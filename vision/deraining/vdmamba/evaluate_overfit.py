
"""
Evaluate a single-patch overfit checkpoint on unseen
RealRain-1k Heavy validation patches.

No training or checkpoint modifications.
Test split remains untouched.

This is a diagnostic, not a full model evaluation.
"""

import argparse
import csv
import math
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


def psnr(prediction, target):
    prediction = prediction.clamp(0, 1)
    mse = torch.mean((prediction - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if not 1 <= args.count <= 112:
        parser.error("--count must be between 1 and 112")

    if args.device not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # Only load checkpoints created by our own project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = ExperimentalVDMambaImageModel(
        channels=config["channels"],
        state_size=config["state_size"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=config["patch_size"],
        augment=False,
        limit=args.count,
    )

    output_dir = (
        Path("outputs/vdmamba/overfit_validation")
        / checkpoint_path.parent.name
    )

    # Avoid overwriting a prior validation experiment.
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    output_dir.mkdir(parents=True)

    results = []

    print("=" * 65)
    print("UNSEEN HEAVY-RAIN VALIDATION")
    print("=" * 65)
    print("Checkpoint:", checkpoint_path)
    print("Validation samples:", len(dataset))

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]

            rainy = sample["rainy"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)

            restored = model(rainy)

            if not torch.isfinite(restored).all().item():
                raise RuntimeError("Non-finite restoration output")

            rainy_score = psnr(rainy, clean)
            restored_score = psnr(restored, clean)
            gain = restored_score - rainy_score

            result = {
                "image_id": sample["image_id"],
                "rainy_psnr": rainy_score,
                "restored_psnr": restored_score,
                "psnr_gain": gain,
            }
            results.append(result)

            if index < 3:
                images = [
                    rainy[0],
                    restored[0].clamp(0, 1),
                    clean[0],
                ]

                pil_images = [
                    to_pil_image(x.detach().cpu())
                    for x in images
                ]

                width, height = pil_images[0].size

                from PIL import Image

                comparison = Image.new(
                    "RGB", (width * 3, height)
                )

                for j, image in enumerate(pil_images):
                    comparison.paste(
                        image.convert("RGB"),
                        (j * width, 0),
                    )

                comparison.save(
                    output_dir /
                    f"comparison_{sample['image_id']}.png"
                )

            print(
                f"Image {sample['image_id']}: "
                f"rainy={rainy_score:.3f} dB, "
                f"restored={restored_score:.3f} dB, "
                f"gain={gain:+.3f} dB",
                flush=True,
            )

    csv_path = output_dir / "results.csv"

    with csv_path.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "rainy_psnr",
                "restored_psnr",
                "psnr_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    mean_gain = sum(
        result["psnr_gain"] for result in results
    ) / len(results)

    improved = sum(
        result["psnr_gain"] > 0
        for result in results
    )

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    print("Mean PSNR gain:", round(mean_gain, 4), "dB")
    print("Images improved:", improved, "/", len(results))
    print("Results:", output_dir)
    print("No validation training performed.")


if __name__ == "__main__":
    main()
