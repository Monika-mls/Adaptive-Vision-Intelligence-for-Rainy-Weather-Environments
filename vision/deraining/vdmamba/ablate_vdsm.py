
"""
VDSM feature-subtraction ablation.

Compares:
1. Normal trained VDMamba-inspired model
2. Same trained model with VDSM subtraction bypassed

Validation split only.
No retraining.
No checkpoint changes.

This is an inference-time diagnostic, not a fair
retrained architecture comparison.
"""

import math

import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import (
    HeavyRainDataset,
)
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


CHECKPOINT = (
    "outputs/vdmamba/multi_train/"
    "run_20261028_093414_860421/best.pt"
)


def psnr_from_mse(mse):
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


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

    methods = [
        "rainy",
        "normal_vdsm",
        "vdsm_bypassed",
    ]

    squared_errors = {
        name: 0.0 for name in methods
    }

    improved = {
        name: 0 for name in methods
    }

    total_pixels = 0

    with torch.inference_mode():
        for batch in loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            features, _ = model.fem(rainy)

            decomposition = model.vdsm(features)

            normal_output = model.reconstruction(
                decomposition["background"],
                rainy,
            ).clamp(0, 1)

            # Bypass feature subtraction:
            # reconstruct from FEM features directly.
            bypass_output = model.reconstruction(
                features,
                rainy,
            ).clamp(0, 1)

            predictions = {
                "rainy": rainy,
                "normal_vdsm": normal_output,
                "vdsm_bypassed": bypass_output,
            }

            rainy_mse = (
                (rainy - clean).square().mean().item()
            )

            pixels = clean.numel()
            total_pixels += pixels

            for name, prediction in predictions.items():
                if prediction.shape != clean.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}"
                    )

                if not torch.isfinite(
                    prediction
                ).all().item():
                    raise RuntimeError(
                        f"Non-finite output for {name}"
                    )

                mse = (
                    (prediction - clean)
                    .square()
                    .mean()
                    .item()
                )

                squared_errors[name] += mse * pixels

                if (
                    name != "rainy"
                    and mse < rainy_mse
                ):
                    improved[name] += 1

    print("=" * 65)
    print("VDSM ABLATION DIAGNOSTIC")
    print("=" * 65)

    rainy_psnr = psnr_from_mse(
        squared_errors["rainy"] / total_pixels
    )

    for name in methods:
        pooled_mse = (
            squared_errors[name] / total_pixels
        )

        restored_psnr = psnr_from_mse(
            pooled_mse
        )

        print(f"\n{name.upper()}")
        print(
            "Pooled PSNR:",
            round(restored_psnr, 4),
        )

        print(
            "PSNR gain:",
            round(
                restored_psnr - rainy_psnr, 4
            ),
        )

        if name != "rainy":
            print(
                "Improved patches:",
                f"{improved[name]}/{len(dataset)}",
            )

    print("\nValidation samples:", len(dataset))
    print("Checkpoint modified: No")
    print("Test split used: No")


if __name__ == "__main__":
    main()
