
"""
Compare a trained VDMamba-inspired model with a
simple brightness-correction baseline.

Uses:
- 784-pair training-derived brightness offset
- 112 Heavy-rain validation center crops
- Existing trained epoch-4 checkpoint

No training or checkpoint modifications.
No held-out test data.
"""

import math

import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


CHECKPOINT = (
    "outputs/vdmamba/multi_train/"
    "run_20261028_093414_860421/best.pt"
)

# Estimated using the 784 training pairs only.
BRIGHTNESS_OFFSET = 12.9815 / 255.0


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
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

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=64,
        augment=False,
        limit=112,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    methods = ["rainy", "brightness_only", "vdmamba"]

    squared_errors = {name: 0.0 for name in methods}
    improved = {name: 0 for name in methods}
    signed_changes = {
        "brightness_only": 0.0,
        "vdmamba": 0.0,
    }

    total_pixels = 0
    count = 0

    with torch.inference_mode():
        for batch in loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            brightness_only = (
                rainy - BRIGHTNESS_OFFSET
            ).clamp(0, 1)

            restored = model(rainy).clamp(0, 1)

            predictions = {
                "rainy": rainy,
                "brightness_only": brightness_only,
                "vdmamba": restored,
            }

            baseline_mse = torch.mean(
                (rainy - clean).square()
            ).item()

            pixels = clean.numel()

            for name, prediction in predictions.items():
                if prediction.shape != clean.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}"
                    )

                if not torch.isfinite(prediction).all().item():
                    raise RuntimeError(
                        f"Non-finite output for {name}"
                    )

                mse = torch.mean(
                    (prediction - clean).square()
                ).item()

                squared_errors[name] += mse * pixels

                if name != "rainy" and mse < baseline_mse:
                    improved[name] += 1

            for name in signed_changes:
                signed_changes[name] += (
                    (predictions[name] - rainy)
                    .sum()
                    .item()
                )

            total_pixels += pixels
            count += 1

    def psnr(name):
        mse = squared_errors[name] / total_pixels

        if mse == 0:
            return float("inf")

        return -10 * math.log10(mse)

    rainy_psnr = psnr("rainy")

    print("=" * 65)
    print("HEAVY-RAIN BRIGHTNESS BASELINE COMPARISON")
    print("=" * 65)

    print("Validation pairs:", count)
    print(
        "Training-derived brightness offset:",
        round(BRIGHTNESS_OFFSET * 255, 4),
    )

    for name in methods:
        restored_psnr = psnr(name)

        print(f"\n{name.upper()}")
        print("Pooled PSNR:", round(restored_psnr, 4))
        print(
            "PSNR gain:",
            round(restored_psnr - rainy_psnr, 4),
        )

        if name != "rainy":
            print(
                "Improved patches:",
                f"{improved[name]}/{count}",
            )
            print(
                "Mean signed change (0-255):",
                round(
                    signed_changes[name]
                    / total_pixels * 255,
                    4,
                ),
            )

    print("\nComparison completed.")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
