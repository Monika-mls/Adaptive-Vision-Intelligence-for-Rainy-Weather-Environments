
"""
PReNet single-pair validation experiment.

Dataset: RealRain-1k
Sample: Light validation image 1007
Model: Original Rain100L PReNet6

Outputs:
    rainy.png
    restored.png
    clean.png
    comparison.png
    metrics.json

No training.
No test split access.
No checkpoint modification.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity

from vision.deraining.prenet_adapter import restore_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "docs/severity_labels.csv"

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/prenet/realrain_light_1007"
)


def read_validation_pair(image_id="1007"):
    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                str(row["image_id"]) == image_id
                and row["split"] == "validation"
                and row["severity"] == "Light"
            ):
                return row

    raise ValueError(
        f"No Light validation record for image {image_id}"
    )


def resolve_dataset_path(relative_path):
    relative = Path(relative_path)

    if relative.is_absolute():
        raise ValueError(
            "Manifest paths should be dataset-relative."
        )

    if ".." in relative.parts:
        raise ValueError(
            "Parent-directory traversal is not allowed."
        )

    candidates = [
        PROJECT_ROOT / "datasets" / relative,
        PROJECT_ROOT / "data" / relative,
        PROJECT_ROOT / relative,
    ]

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Dataset image not found.\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_aligned_pair(rainy_path, clean_path, max_side):
    with Image.open(rainy_path) as file:
        rainy = ImageOps.exif_transpose(file).convert("RGB")

    with Image.open(clean_path) as file:
        clean = ImageOps.exif_transpose(file).convert("RGB")

    if rainy.size != clean.size:
        raise ValueError(
            "Rainy and clean source dimensions differ: "
            f"{rainy.size} vs {clean.size}. "
            "Alignment must be verified before metrics."
        )

    original_size = rainy.size

    rainy.thumbnail(
        (max_side, max_side),
        Image.Resampling.LANCZOS,
    )

    clean = clean.resize(
        rainy.size,
        Image.Resampling.LANCZOS,
    )

    return rainy, clean, original_size


def image_float(image):
    return np.asarray(
        image, dtype=np.float32
    ) / 255.0


def calculate_psnr(prediction, target):
    mse = float(
        np.mean(
            (prediction.astype(np.float64)
             - target.astype(np.float64)) ** 2
        )
    )

    if mse == 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(prediction, target):
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=1.0,
            channel_axis=-1,
        )
    )


def create_comparison(rainy, restored, clean, output_path):
    images = [
        ("Rainy input", rainy),
        ("PReNet restored", restored),
        ("Clean target", clean),
    ]

    panel_width, panel_height = rainy.size
    title_height = 36
    padding = 12

    canvas = Image.new(
        "RGB",
        (
            panel_width * 3 + padding * 4,
            panel_height + title_height + padding * 2,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, (title, image) in enumerate(images):
        x = padding + index * (
            panel_width + padding
        )

        draw.text(
            (x, padding + 5),
            title,
            fill="black",
        )

        canvas.paste(
            image,
            (x, padding + title_height),
        )

    canvas.save(output_path)


def run_experiment(image_id, device, max_side, output_dir):
    if max_side < 16:
        raise ValueError(
            "max_side must be at least 16."
        )

    row = read_validation_pair(image_id)

    rainy_path = resolve_dataset_path(
        row["rainy_path"]
    )

    clean_path = resolve_dataset_path(
        row["clean_path"]
    )

    rainy, clean, original_size = load_aligned_pair(
        rainy_path,
        clean_path,
        max_side,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rainy_output = output_dir / "rainy.png"
    clean_output = output_dir / "clean.png"
    restored_output = output_dir / "restored.png"
    comparison_output = output_dir / "comparison.png"

    rainy.save(rainy_output)
    clean.save(clean_output)

    # Runs the previously verified original-model adapter.
    restoration = restore_image(
        input_path=rainy_output,
        output_path=restored_output,
        device=device,
        max_side=max_side,
    )

    with Image.open(restored_output) as file:
        restored = file.convert("RGB").copy()

    if not (
        rainy.size == restored.size == clean.size
    ):
        raise RuntimeError(
            "Evaluation images have mismatched dimensions."
        )

    rainy_array = image_float(rainy)
    restored_array = image_float(restored)
    clean_array = image_float(clean)

    rainy_psnr = calculate_psnr(
        rainy_array, clean_array
    )

    restored_psnr = calculate_psnr(
        restored_array, clean_array
    )

    rainy_ssim = calculate_ssim(
        rainy_array, clean_array
    )

    restored_ssim = calculate_ssim(
        restored_array, clean_array
    )

    create_comparison(
        rainy,
        restored,
        clean,
        comparison_output,
    )

    metrics = {
        "dataset": "RealRain-1k",
        "image_id": image_id,
        "split": row["split"],
        "severity": row["severity"],
        "model": "PReNet6",
        "pretraining_dataset": "Rain100L",
        "source_image_size": list(original_size),
        "evaluation_size": list(rainy.size),
        "evaluation_note": (
            "Both rainy and clean references were "
            "resized identically; metrics are for the "
            "resized PNG images, not original resolution."
        ),
        "quality": {
            "rainy_psnr_db": rainy_psnr,
            "restored_psnr_db": restored_psnr,
            "psnr_change_db": (
                restored_psnr - rainy_psnr
            ),
            "rainy_ssim": rainy_ssim,
            "restored_ssim": restored_ssim,
            "ssim_change": (
                restored_ssim - rainy_ssim
            ),
        },
        "restoration": restoration,
        "outputs": {
            "rainy": str(rainy_output),
            "restored": str(restored_output),
            "clean": str(clean_output),
            "comparison": str(comparison_output),
        },
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=False,
        )

    print("=" * 65)
    print("PRENET REALRAIN-1K VALIDATION EXPERIMENT")
    print("=" * 65)

    print("Image ID:", image_id)
    print("Severity:", row["severity"])
    print("Split:", row["split"])
    print("Device:", restoration["device"])
    print("Original size:", original_size)
    print("Evaluation size:", rainy.size)

    print("\nQUALITY COMPARISON")
    print(f"Rainy PSNR:    {rainy_psnr:.4f} dB")
    print(f"Restored PSNR: {restored_psnr:.4f} dB")
    print(
        f"PSNR change:   "
        f"{restored_psnr - rainy_psnr:+.4f} dB"
    )

    print(f"\nRainy SSIM:    {rainy_ssim:.4f}")
    print(f"Restored SSIM: {restored_ssim:.4f}")
    print(
        f"SSIM change:   "
        f"{restored_ssim - rainy_ssim:+.4f}"
    )

    print("\nPERFORMANCE")
    print(
        "Inference seconds:",
        restoration["inference_seconds"],
    )
    print(
        "Peak GPU allocated MiB:",
        restoration["peak_gpu_allocated_mib"],
    )

    print("\nOUTPUTS")
    print("Comparison:", comparison_output)
    print("Metrics:", metrics_path)
    print("\nTest split: NOT ACCESSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-id",
        default="1007",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    run_experiment(
        image_id=args.image_id,
        device=args.device,
        max_side=args.max_side,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
