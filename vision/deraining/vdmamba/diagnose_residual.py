
"""
Residual analysis for the experimental VDMamba-inspired model.

Compares predicted image corrections with the paired
rainy-to-clean target corrections.

Uses the validation split only.
Does not modify the checkpoint.
"""

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


def spatial_correlation(predicted, target):
    """
    Calculate correlation after subtracting each
    correction map's per-channel spatial mean.

    Returns None if either map has zero variation.
    """
    predicted = predicted.float()
    target = target.float()

    pred_centered = predicted - predicted.mean(
        dim=(-2, -1), keepdim=True
    )

    target_centered = target - target.mean(
        dim=(-2, -1), keepdim=True
    )

    numerator = (
        pred_centered * target_centered
    ).sum()

    pred_energy = pred_centered.square().sum()
    target_energy = target_centered.square().sum()

    denominator = torch.sqrt(
        pred_energy * target_energy
    )

    if denominator.item() < 1e-12:
        return None

    return (
        numerator / denominator
    ).item()


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

    results = {
        "target_mean": [],
        "predicted_mean": [],
        "target_std": [],
        "predicted_std": [],
        "correlation": [],
        "centered_mae": [],
        "brightness_bias": [],
    }

    with torch.inference_mode():
        for batch in loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            restored = model(rainy).clamp(0, 1)

            target_correction = clean - rainy
            predicted_correction = restored - rainy

            target_centered = target_correction - (
                target_correction.mean(
                    dim=(-2, -1), keepdim=True
                )
            )

            predicted_centered = predicted_correction - (
                predicted_correction.mean(
                    dim=(-2, -1), keepdim=True
                )
            )

            results["target_mean"].append(
                target_correction.mean().item() * 255
            )

            results["predicted_mean"].append(
                predicted_correction.mean().item() * 255
            )

            results["target_std"].append(
                target_centered.std(
                    unbiased=False
                ).item() * 255
            )

            results["predicted_std"].append(
                predicted_centered.std(
                    unbiased=False
                ).item() * 255
            )

            correlation = spatial_correlation(
                predicted_correction,
                target_correction,
            )

            if correlation is not None:
                results["correlation"].append(
                    correlation
                )

            results["centered_mae"].append(
                (
                    predicted_centered - target_centered
                ).abs().mean().item() * 255
            )

            results["brightness_bias"].append(
                abs(
                    predicted_correction.mean().item()
                    - target_correction.mean().item()
                ) * 255
            )

    print("=" * 65)
    print("VDMAMBA RESIDUAL ANALYSIS")
    print("=" * 65)

    print("Validation samples:", len(dataset))

    for key, values in results.items():
        if not values:
            print(f"{key}: unavailable")
            continue

        average = sum(values) / len(values)

        print(
            f"{key}: {average:.6f}"
        )

    print("=" * 65)

    print(
        "Correlation is calculated after removing "
        "per-channel spatial brightness offsets."
    )

    print(
        "Higher correlation suggests better alignment "
        "with target correction patterns."
    )

    print(
        "A correlation near zero suggests little "
        "linear alignment, not necessarily no restoration."
    )

    print(
        "Centered MAE and correction variation "
        "are measured on the 0-255 pixel scale."
    )

    print("Test split not used.")


if __name__ == "__main__":
    main()
