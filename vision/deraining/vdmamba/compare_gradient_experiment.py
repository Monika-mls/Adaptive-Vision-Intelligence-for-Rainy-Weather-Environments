
"""
Controlled comparison of 64-image restoration experiments.

Model A: Charbonnier + SSIM
Model B: Charbonnier + SSIM + Gradient

Evaluates the same 112 validation crops.
No retraining or checkpoint modification.
No test split usage.
"""

import math
import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


CHECKPOINTS = {
    "SSIM only": (
        "outputs/vdmamba/multi_train/"
        "run_20261028_071905_525138/best.pt"
    ),
    "SSIM + Gradient": (
        "outputs/vdmamba/multi_train/"
        "run_20261029_075740_376085/best.pt"
    ),
}


def load_model(path, device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]

    model = ExperimentalVDMambaImageModel(
        channels=int(config["channels"]),
        state_size=int(config["state_size"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    model.eval()

    return model


def gradient_mae(pred, target):
    dx1 = pred[..., :, 1:] - pred[..., :, :-1]
    dx2 = target[..., :, 1:] - target[..., :, :-1]

    dy1 = pred[..., 1:, :] - pred[..., :-1, :]
    dy2 = target[..., 1:, :] - target[..., :-1, :]

    return (
        (dx1 - dx2).abs().mean()
        + (dy1 - dy2).abs().mean()
    ).item() / 2


def spatial_correlation(pred_correction, target_correction):
    a = pred_correction - pred_correction.mean(
        dim=(-2, -1), keepdim=True
    )
    b = target_correction - target_correction.mean(
        dim=(-2, -1), keepdim=True
    )

    numerator = (a * b).sum()
    denominator = torch.sqrt(
        a.square().sum() * b.square().sum()
    )

    if denominator.item() < 1e-12:
        return None

    return (numerator / denominator).item()


def evaluate(model, loader, device):
    results = {
        "gradient_mae": [],
        "correlation": [],
        "centered_mae": [],
        "predicted_std": [],
        "target_std": [],
    }

    total_squared_error = 0.0
    total_pixels = 0

    with torch.inference_mode():
        for batch in loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            restored = model(rainy).clamp(0, 1)

            if restored.shape != clean.shape:
                raise RuntimeError("Output shape mismatch")

            if not torch.isfinite(restored).all().item():
                raise RuntimeError("Non-finite prediction")

            error = (restored - clean).square()
            total_squared_error += error.sum().item()
            total_pixels += clean.numel()

            results["gradient_mae"].append(
                gradient_mae(restored, clean)
            )

            predicted = restored - rainy
            target = clean - rainy

            correlation = spatial_correlation(
                predicted, target
            )

            if correlation is not None:
                results["correlation"].append(correlation)

            predicted_centered = predicted - predicted.mean(
                dim=(-2, -1), keepdim=True
            )
            target_centered = target - target.mean(
                dim=(-2, -1), keepdim=True
            )

            results["centered_mae"].append(
                (predicted_centered - target_centered)
                .abs().mean().item() * 255
            )

            results["predicted_std"].append(
                predicted_centered.std(
                    unbiased=False
                ).item() * 255
            )

            results["target_std"].append(
                target_centered.std(
                    unbiased=False
                ).item() * 255
            )

    summary = {}

    for key, values in results.items():
        summary[key] = (
            sum(values) / len(values)
            if values else None
        )

    mse = total_squared_error / total_pixels
    summary["pooled_psnr"] = (
        -10 * math.log10(mse)
        if mse > 0 else float("inf")
    )

    summary["valid_correlations"] = len(
        results["correlation"]
    )

    return summary


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    print("=" * 65)
    print("CONTROLLED GRADIENT-LOSS COMPARISON")
    print("=" * 65)
    print("Device:", device)
    print("Validation samples:", len(dataset))

    for name, checkpoint_path in CHECKPOINTS.items():
        model = load_model(checkpoint_path, device)
        metrics = evaluate(model, loader, device)

        print(f"\n{name.upper()}")
        print("-" * 45)

        for key, value in metrics.items():
            if value is None:
                print(f"{key}: unavailable")
            elif isinstance(value, int):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.6f}")

        del model

    print("\nComparison completed.")
    print("All metrics use the same validation samples.")
    print("Test split not used.")


if __name__ == "__main__":
    main()
