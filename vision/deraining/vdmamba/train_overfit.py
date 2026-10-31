
"""
Step 16: Controlled single-patch overfitting experiment.

Trains the experimental VDMamba-inspired network
on ONE fixed rainy/clean training pair.

Purpose:
- Verify optimizer updates
- Verify reconstruction loss can decrease
- Compare rainy, initial and trained PSNR
- Save visual evidence and an isolated checkpoint

Does not use validation or test data.
Does not modify existing checkpoints.
Not representative of generalization.
"""

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


def charbonnier(prediction, target, epsilon=1e-3):
    return torch.sqrt(
        (prediction - target).square() + epsilon**2
    ).mean()


@torch.no_grad()
def psnr(prediction, target):
    prediction = prediction.clamp(0, 1)
    mse = torch.mean(
        (prediction - target).square()
    ).item()

    if mse <= 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


def save_image(tensor, path):
    image = to_pil_image(
        tensor.detach().float().cpu().clamp(0, 1)
    )
    image.save(path)


def create_comparison(images, path):
    pil_images = [
        to_pil_image(
            image.detach().float().cpu().clamp(0, 1)
        ).convert("RGB")
        for image in images
    ]

    width, height = pil_images[0].size

    canvas = Image.new(
        "RGB",
        (width * len(pil_images), height),
    )

    for index, image in enumerate(pil_images):
        canvas.paste(image, (index * width, 0))

    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--state-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    args = parser.parse_args()

    if args.steps < 1 or args.patch_size < 2:
        parser.error("steps and patch-size must be positive.")

    if args.channels < 1 or args.state_size < 1:
        parser.error("Invalid model dimensions.")

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be positive and finite.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    dataset = HeavyRainDataset(
        split="train",
        patch_size=args.patch_size,
        augment=False,
        limit=1,
    )

    # Read one sample ONCE. The dataset's training
    # crop is random even when augment=False.
    # Keeping this batch fixed makes overfitting measurable.
    sample = dataset[0]

    rainy = sample["rainy"].unsqueeze(0).to(device)
    clean = sample["clean"].unsqueeze(0).to(device)

    if rainy.shape != clean.shape:
        raise RuntimeError("Rainy/clean shapes differ.")

    if not (torch.isfinite(rainy).all() and torch.isfinite(clean).all()):
        raise RuntimeError("Non-finite training data.")

    model = ExperimentalVDMambaImageModel(
        channels=args.channels,
        state_size=args.state_size,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        Path("outputs/vdmamba/overfit")
        / f"run_{timestamp}"
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    save_image(rainy[0], output_dir / "rainy.png")
    save_image(clean[0], output_dir / "clean.png")

    model.eval()

    with torch.no_grad():
        initial = model(rainy)

        if not torch.isfinite(initial).all():
            raise RuntimeError("Initial model output is non-finite.")

        initial_loss = charbonnier(initial, clean).item()
        initial_psnr = psnr(initial, clean)
        rainy_psnr = psnr(rainy, clean)

        save_image(initial[0], output_dir / "initial.png")

    print("=" * 65)
    print("VDMAMBA EXPERIMENTAL SINGLE-PATCH TRAINING")
    print("=" * 65)

    print("Device:", device)
    print("Image ID:", sample["image_id"])
    print("Input shape:", tuple(rainy.shape))
    print("Training steps:", args.steps)
    print("Learning rate:", args.lr)
    print("Rainy input PSNR:", round(rainy_psnr, 4))
    print("Initial model PSNR:", round(initial_psnr, 4))
    print("Initial loss:", round(initial_loss, 6))

    history = []

    model.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(rainy)

        if not torch.isfinite(prediction).all():
            raise RuntimeError(
                f"Non-finite output at step {step}"
            )

        loss = charbonnier(prediction, clean)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        record = {
            "step": step,
            "training_loss": float(loss.item()),
        }

        history.append(record)

        if step == 1 or step % 5 == 0 or step == args.steps:
            print(
                f"Step {step:03d}/{args.steps} | "
                f"Loss: {loss.item():.6f}",
                flush=True,
            )

    model.eval()

    with torch.no_grad():
        restored = model(rainy)

        if not torch.isfinite(restored).all():
            raise RuntimeError("Final model output is non-finite.")

        final_loss = charbonnier(restored, clean).item()
        final_psnr = psnr(restored, clean)

        save_image(restored[0], output_dir / "restored.png")

        create_comparison(
            [
                rainy[0],
                initial[0],
                restored[0],
                clean[0],
            ],
            output_dir / "comparison.png",
        )

    report = {
        "experiment": "single_patch_overfit",
        "model": "ExperimentalVDMambaImageModel",
        "model_status": "experimental_vdmamba_inspired",
        "generalization_evaluated": False,
        "image_id": str(sample["image_id"]),
        "device": str(device),
        "patch_size": args.patch_size,
        "channels": args.channels,
        "state_size": args.state_size,
        "seed": args.seed,
        "learning_rate": args.lr,
        "steps": args.steps,
        "rainy_psnr": rainy_psnr,
        "initial_model_psnr": initial_psnr,
        "trained_model_psnr": final_psnr,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "history": history,
    }

    report_path = output_dir / "report.json"

    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    checkpoint_path = output_dir / "overfit_checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "channels": args.channels,
                "state_size": args.state_size,
                "patch_size": args.patch_size,
            },
            "experiment": "single_patch_overfit",
            "steps": args.steps,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    print("\n" + "=" * 65)
    print("TRAINING EXPERIMENT RESULTS")
    print("=" * 65)

    print(f"Rainy input PSNR:    {rainy_psnr:.4f} dB")
    print(f"Initial model PSNR:  {initial_psnr:.4f} dB")
    print(f"Trained model PSNR:  {final_psnr:.4f} dB")
    print(f"Initial loss:        {initial_loss:.6f}")
    print(f"Final loss:          {final_loss:.6f}")

    print(
        "PSNR gain vs rainy:",
        f"{final_psnr - rainy_psnr:+.4f} dB",
    )

    print(
        "PSNR gain vs initial:",
        f"{final_psnr - initial_psnr:+.4f} dB",
    )

    print("Results saved:", output_dir)

    if final_loss < initial_loss:
        print("RESULT: Reconstruction loss decreased.")
    else:
        print("RESULT: Loss did not improve; investigate training.")

    print(
        "NOTE: This is training-set memorization, "
        "not validation performance."
    )


if __name__ == "__main__":
    main()
