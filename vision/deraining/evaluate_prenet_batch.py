
"""
PReNet multi-image validation experiment.

Project: Adaptive Vision Intelligence for Rainy Weather Environments

Evaluates Light-rain RealRain-1k validation pairs using
the original pretrained Rain100L PReNet6 checkpoint.

Features:
- Deterministic sample selection
- Read-only dataset preflight
- Optional download of missing clean validation targets
- Single model load for all images
- PSNR/SSIM before and after restoration
- CUDA inference time and allocated GPU memory
- Individual CSV results, JSON summary, image comparisons

No training. No test-split access.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity

from vision.deraining.prenet_adapter import load_prenet


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/severity_labels.csv"
DATA_ROOT = ROOT / "datasets"

DEFAULT_OUTPUT = ROOT / "outputs/prenet/batch_validation"


def sort_key(row):
    image_id = str(row["image_id"])
    return (
        0, int(image_id)
    ) if image_id.isdigit() else (1, image_id)


def select_pairs(count):
    if count < 1:
        raise ValueError("count must be positive.")

    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    selected = sorted(
        [
            row for row in rows
            if row["split"] == "validation"
            and row["severity"] == "Light"
        ],
        key=sort_key,
    )[:count]

    if len(selected) != count:
        raise ValueError(
            f"Requested {count} validation pairs, "
            f"found {len(selected)}."
        )

    for row in selected:
        for column in ("rainy_path", "clean_path"):
            relative = Path(row[column])

            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "test" in relative.parts
                or "train" in relative.parts
            ):
                raise ValueError(
                    f"Invalid validation path: {relative}"
                )

            if "validation" not in relative.parts:
                raise ValueError(
                    f"Non-validation path: {relative}"
                )

    return selected


def local_file(relative_path):
    return DATA_ROOT / relative_path


def check_inputs(rows):
    missing_rainy = []
    missing_clean = []

    for row in rows:
        rainy = local_file(row["rainy_path"])
        clean = local_file(row["clean_path"])

        if not rainy.is_file():
            missing_rainy.append(row["rainy_path"])

        if not clean.is_file():
            missing_clean.append(row["clean_path"])

    return missing_rainy, missing_clean


def download_clean_targets(missing, revision):
    if not revision:
        raise ValueError(
            "A pinned dataset revision is required to "
            "download missing clean targets."
        )

    from huggingface_hub import hf_hub_download

    for index, relative in enumerate(missing, 1):
        print(
            f"Downloading clean validation target "
            f"{index}/{len(missing)}: {relative}",
            flush=True,
        )

        hf_hub_download(
            repo_id="dronefreak/RealRain-1k",
            repo_type="dataset",
            filename=relative,
            revision=revision,
            local_dir=str(DATA_ROOT),
        )


def read_aligned_images(rainy_path, clean_path, max_side):
    with Image.open(rainy_path) as file:
        rainy = ImageOps.exif_transpose(file).convert("RGB")

    with Image.open(clean_path) as file:
        clean = ImageOps.exif_transpose(file).convert("RGB")

    if rainy.size != clean.size:
        raise ValueError(
            f"Source dimensions do not match: "
            f"{rainy.size} vs {clean.size}"
        )

    source_size = rainy.size

    rainy.thumbnail(
        (max_side, max_side),
        Image.Resampling.LANCZOS,
    )

    clean = clean.resize(
        rainy.size,
        Image.Resampling.LANCZOS,
    )

    return rainy, clean, source_size


def to_tensor(image, device):
    array = np.asarray(
        image, dtype=np.float32
    ) / 255.0

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )


def tensor_to_image(tensor):
    array = (
        tensor[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    array = (
        np.clip(array * 255.0, 0, 255)
        .round()
        .astype(np.uint8)
    )

    return Image.fromarray(array)


def metrics_against(prediction, clean):
    prediction_array = np.asarray(
        prediction, dtype=np.float64
    )

    clean_array = np.asarray(
        clean, dtype=np.float64
    )

    mse = float(
        np.mean(
            (prediction_array - clean_array) ** 2
        )
    )

    psnr = (
        10.0 * math.log10((255.0 ** 2) / mse)
        if mse > 0
        else float("inf")
    )

    ssim = float(
        structural_similarity(
            np.asarray(clean),
            np.asarray(prediction),
            data_range=255,
            channel_axis=-1,
        )
    )

    return psnr, ssim


def save_comparison(rainy, restored, clean, path):
    images = [
        ("Rainy", rainy),
        ("PReNet restored", restored),
        ("Clean target", clean),
    ]

    width, height = rainy.size
    padding = 12
    heading = 35

    canvas = Image.new(
        "RGB",
        (
            width * 3 + padding * 4,
            height + heading + padding * 2,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, (label, image) in enumerate(images):
        x = padding + index * (width + padding)

        draw.text(
            (x, padding + 5),
            label,
            fill="black",
        )

        canvas.paste(image, (x, padding + heading))

    canvas.save(path)


def evaluate_one(row, model, device, max_side, output_dir):
    rainy_path = local_file(row["rainy_path"])
    clean_path = local_file(row["clean_path"])

    rainy, clean, source_size = read_aligned_images(
        rainy_path,
        clean_path,
        max_side,
    )

    tensor = to_tensor(rainy, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        restored_tensor, stages = model(tensor)
        restored_tensor = restored_tensor.clamp(0, 1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if restored_tensor.shape != tensor.shape:
        raise RuntimeError(
            "PReNet returned an unexpected output shape."
        )

    if not torch.isfinite(restored_tensor).all().item():
        raise RuntimeError(
            "PReNet output contains non-finite values."
        )

    restored = tensor_to_image(restored_tensor)

    rainy_psnr, rainy_ssim = metrics_against(
        rainy, clean
    )

    restored_psnr, restored_ssim = metrics_against(
        restored, clean
    )

    image_dir = (
        output_dir / f"image_{row['image_id']}"
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    rainy.save(image_dir / "rainy.png")
    restored.save(image_dir / "restored.png")
    clean.save(image_dir / "clean.png")

    save_comparison(
        rainy,
        restored,
        clean,
        image_dir / "comparison.png",
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    return {
        "image_id": row["image_id"],
        "split": row["split"],
        "severity": row["severity"],
        "source_width": source_size[0],
        "source_height": source_size[1],
        "eval_width": rainy.width,
        "eval_height": rainy.height,
        "rainy_psnr": rainy_psnr,
        "restored_psnr": restored_psnr,
        "psnr_change": restored_psnr - rainy_psnr,
        "rainy_ssim": rainy_ssim,
        "restored_ssim": restored_ssim,
        "ssim_change": restored_ssim - rainy_ssim,
        "inference_seconds": elapsed,
        "peak_gpu_allocated_mib": peak_memory,
        "recurrent_stages": len(stages),
    }


def average(rows, field):
    values = [float(row[field]) for row in rows]
    return sum(values) / len(values)


def make_summary(results, checkpoint):
    return {
        "dataset": "RealRain-1k",
        "split": "validation",
        "severity": "Light",
        "model": "PReNet6",
        "pretraining_dataset": "Rain100L",
        "checkpoint": str(checkpoint),
        "evaluation_note": (
            "Paired rainy and clean images evaluated at "
            "a common resized resolution, max side 256."
        ),
        "count": len(results),
        "mean_rainy_psnr": average(
            results, "rainy_psnr"
        ),
        "mean_restored_psnr": average(
            results, "restored_psnr"
        ),
        "mean_psnr_change": average(
            results, "psnr_change"
        ),
        "mean_rainy_ssim": average(
            results, "rainy_ssim"
        ),
        "mean_restored_ssim": average(
            results, "restored_ssim"
        ),
        "mean_ssim_change": average(
            results, "ssim_change"
        ),
        "psnr_improved_count": sum(
            row["psnr_change"] > 0 for row in results
        ),
        "ssim_improved_count": sum(
            row["ssim_change"] > 0 for row in results
        ),
        "mean_inference_seconds": average(
            results, "inference_seconds"
        ),
        "peak_gpu_allocated_mib": max(
            (
                row["peak_gpu_allocated_mib"]
                for row in results
                if row["peak_gpu_allocated_mib"] is not None
            ),
            default=None,
        ),
        "test_split_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count", type=int, default=10
    )

    parser.add_argument(
        "--max-side", type=int, default=256
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--download-missing-clean",
        action="store_true",
    )

    parser.add_argument(
        "--revision",
        default=None,
        help="Pinned Hugging Face dataset commit SHA.",
    )

    args = parser.parse_args()

    if args.max_side < 16:
        parser.error("--max-side must be at least 16.")

    rows = select_pairs(args.count)

    print("=" * 65)
    print("PRENET LIGHT-RAIN VALIDATION")
    print("=" * 65)

    print("Selected image IDs:")
    print(", ".join(row["image_id"] for row in rows))

    missing_rainy, missing_clean = check_inputs(rows)

    print("Missing rainy images:", len(missing_rainy))
    print("Missing clean images:", len(missing_clean))

    if missing_clean and args.download_missing_clean:
        download_clean_targets(
            missing_clean, args.revision
        )
        missing_rainy, missing_clean = check_inputs(rows)

    if args.prepare_only:
        if missing_rainy:
            print("\nMissing rainy inputs:")
            for item in missing_rainy:
                print(" ", item)

        if missing_clean:
            print("\nMissing clean targets:")
            for item in missing_clean:
                print(" ", item)

        print("\nPreparation only. No model inference.")
        return

    if missing_rainy or missing_clean:
        raise FileNotFoundError(
            "Dataset pairs are incomplete. "
            "Run --prepare-only and download the missing "
            "clean targets using the pinned revision."
        )

    model, device = load_prenet(device=args.device)

    print("Model: PReNet6 (Rain100L)")
    print("Device:", device)
    print("Evaluation images:", len(rows))
    print("Test split: NOT ACCESSED")

    results = []

    for index, row in enumerate(rows, 1):
        result = evaluate_one(
            row, model, device,
            args.max_side, args.output_dir,
        )

        if not all(
            math.isfinite(result[key])
            for key in (
                "rainy_psnr",
                "restored_psnr",
                "rainy_ssim",
                "restored_ssim",
            )
        ):
            raise ValueError(
                "A quality metric is not finite. "
                "Inspect the corresponding image pair."
            )

        results.append(result)

        print(
            f"[{index}/{len(rows)}] "
            f"Image {row['image_id']} "
            f"| PSNR {result['psnr_change']:+.3f} dB "
            f"| SSIM {result['ssim_change']:+.4f} "
            f"| {result['inference_seconds']:.3f}s",
            flush=True,
        )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    csv_path = args.output_dir / "results.csv"

    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    checkpoint = (
        ROOT / "external_models/PReNet/logs/"
        "Rain100L/PReNet6/net_latest.pth"
    )

    summary = make_summary(results, checkpoint)

    summary_path = args.output_dir / "summary.json"

    with summary_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, indent=2,
            allow_nan=False,
        )

    print("\n" + "=" * 65)
    print("MULTI-IMAGE VALIDATION SUMMARY")
    print("=" * 65)

    print("Images evaluated:", summary["count"])
    print(
        "Mean rainy PSNR:",
        f"{summary['mean_rainy_psnr']:.4f} dB",
    )
    print(
        "Mean restored PSNR:",
        f"{summary['mean_restored_psnr']:.4f} dB",
    )
    print(
        "Mean PSNR change:",
        f"{summary['mean_psnr_change']:+.4f} dB",
    )
    print(
        "Mean rainy SSIM:",
        f"{summary['mean_rainy_ssim']:.4f}",
    )
    print(
        "Mean restored SSIM:",
        f"{summary['mean_restored_ssim']:.4f}",
    )
    print(
        "Mean SSIM change:",
        f"{summary['mean_ssim_change']:+.4f}",
    )
    print(
        "PSNR improved:",
        f"{summary['psnr_improved_count']}/{len(results)}",
    )
    print(
        "SSIM improved:",
        f"{summary['ssim_improved_count']}/{len(results)}",
    )
    print(
        "Average inference time:",
        f"{summary['mean_inference_seconds']:.4f}s",
    )
    print(
        "Maximum peak PyTorch GPU allocation:",
        summary["peak_gpu_allocated_mib"],
        "MiB",
    )

    print("\nResults:", csv_path)
    print("Summary:", summary_path)
    print("Test split: NOT ACCESSED")


if __name__ == "__main__":
    main()
