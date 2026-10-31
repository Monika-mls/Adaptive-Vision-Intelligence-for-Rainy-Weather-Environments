"""Read-only checkpoint quality audit on Heavy validation center crops."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import ExperimentalVDMambaImageModel
from vision.deraining.vdmamba.validation_metrics import evaluate_batch, summarize


def comparison(rainy, restored, clean, path):
    images = [to_pil_image(t.detach().float().cpu().clamp(0, 1)).convert("RGB")
              for t in (rainy, restored, clean)]
    w, h = images[0].size
    canvas = Image.new("RGB", (3*w, h+22), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (im, label) in enumerate(zip(images, ("Rainy", "Restored", "Clean"))):
        canvas.paste(im, (j*w, 22))
        draw.text((j*w + 3, 4), label, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--count", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=1/255)
    parser.add_argument("--save-comparisons", type=int, default=15)
    args = parser.parse_args()
    if not (1 <= args.count <= 112) or args.save_comparisons < 0:
        parser.error("count must be 1..112 and save-comparisons must be >=0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        parser.error("Checkpoint not found")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = ExperimentalVDMambaImageModel(
        channels=int(cfg["channels"]), state_size=int(cfg["state_size"])
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    data = HeavyRainDataset(split="validation", patch_size=int(cfg["patch_size"]),
                            augment=False, limit=args.count)
    out = Path("outputs/vdmamba/quality_validation") / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    tensors_for_comparisons = []
    with torch.inference_mode():
        for sample in data:
            rainy = sample["rainy"].unsqueeze(0).to(args.device)
            clean = sample["clean"].unsqueeze(0).to(args.device)
            pred = model(rainy)
            measurements = evaluate_batch(rainy, pred, clean, clean_threshold=args.threshold)[0]
            measurements["image_id"] = str(sample["image_id"])
            rows.append(measurements)
            # Store only the worst comparisons by MSE delta, bounded to avoid VRAM growth.
            if args.save_comparisons:
                score = measurements["restored_mse"] - measurements["rainy_mse"]
                tensors_for_comparisons.append((score, str(sample["image_id"]),
                   rainy[0].cpu(), pred[0].cpu(), clean[0].cpu()))
                tensors_for_comparisons.sort(key=lambda x: x[0], reverse=True)
                if len(tensors_for_comparisons) > args.save_comparisons:
                    tensors_for_comparisons.pop()
    summary = summarize(rows)
    summary.update({
        "experiment": "experimental_vdmamba_inspired_validation",
        "checkpoint": str(checkpoint_path),
        "split": "validation",
        "crop": "deterministic center crop",
        "patch_size": int(cfg["patch_size"]),
        "clean_like_threshold": args.threshold,
        "not_full_resolution": True,
        "test_split_used": False,
    })
    with (out / "per_image.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for _, image_id, rainy, restored, clean in tensors_for_comparisons:
        comparison(rainy, restored, clean, out / f"worst_{image_id}.png")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("Results saved:", out)


if __name__ == "__main__":
    main()
