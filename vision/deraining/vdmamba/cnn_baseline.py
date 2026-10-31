
"""
Small residual CNN baseline for Heavy-rain restoration.

Trains on 64 paired images, evaluates on 112
validation crops, and compares against the existing
64-image VDMamba-inspired checkpoint.

Does not modify existing models or checkpoints.
Does not use the test split.
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)
from vision.deraining.vdmamba.restoration_loss import RestorationLoss


VDMAMBA_CHECKPOINT = (
    "outputs/vdmamba/multi_train/"
    "run_20261028_071905_525138/best.pt"
)


class SimpleCNN(nn.Module):
    def __init__(self, channels=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image):
        return image + self.net(image)


def load_vdmamba(device):
    checkpoint = torch.load(
        VDMAMBA_CHECKPOINT,
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

    return model.eval()


def gradient_mae(prediction, target):
    dx = (
        prediction[..., :, 1:] - prediction[..., :, :-1]
        - target[..., :, 1:] + target[..., :, :-1]
    ).abs().mean()

    dy = (
        prediction[..., 1:, :] - prediction[..., :-1, :]
        - target[..., 1:, :] + target[..., :-1, :]
    ).abs().mean()

    return ((dx + dy) / 2).item()


def correlation(prediction, target, rainy):
    a = prediction - rainy
    b = target - rainy

    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)

    denominator = torch.sqrt(
        a.square().sum() * b.square().sum()
    )

    if denominator.item() < 1e-12:
        return None

    return (a * b).sum().div(denominator).item()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    total_error = 0.0
    baseline_error = 0.0
    total_pixels = 0
    improved = 0
    gradients = []
    correlations = []
    centered_errors = []

    for batch in loader:
        rainy = batch["rainy"].to(device)
        clean = batch["clean"].to(device)
        restored = model(rainy).clamp(0, 1)

        if restored.shape != clean.shape:
            raise RuntimeError("Output shape mismatch")

        if not torch.isfinite(restored).all().item():
            raise RuntimeError("Non-finite prediction")

        error = (restored - clean).square()
        original_error = (rainy - clean).square()

        total_error += error.sum().item()
        baseline_error += original_error.sum().item()
        total_pixels += clean.numel()

        improved += int(
            error.mean().item() < original_error.mean().item()
        )

        gradients.append(gradient_mae(restored, clean))

        corr = correlation(restored, clean, rainy)
        if corr is not None:
            correlations.append(corr)

        predicted = restored - rainy
        target = clean - rainy

        predicted -= predicted.mean(
            dim=(-2, -1), keepdim=True
        )
        target -= target.mean(
            dim=(-2, -1), keepdim=True
        )

        centered_errors.append(
            (predicted - target).abs().mean().item() * 255
        )

    psnr = lambda mse: (
        -10 * math.log10(mse)
        if mse > 0 else float("inf")
    )

    restored_psnr = psnr(total_error / total_pixels)
    rainy_psnr = psnr(baseline_error / total_pixels)

    return {
        "restored_psnr": restored_psnr,
        "psnr_gain": restored_psnr - rainy_psnr,
        "gradient_mae": sum(gradients) / len(gradients),
        "spatial_correlation": (
            sum(correlations) / len(correlations)
            if correlations else None
        ),
        "centered_mae": (
            sum(centered_errors) / len(centered_errors)
        ),
        "improved_patches": improved,
        "validation_count": len(loader.dataset),
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    train_data = HeavyRainDataset(
        split="train",
        patch_size=64,
        augment=True,
        limit=64,
    )

    val_data = HeavyRainDataset(
        split="validation",
        patch_size=64,
        augment=False,
        limit=112,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    cnn = SimpleCNN(channels=8).to(device)

    criterion = RestorationLoss(
        ssim_weight=0.02,
        gradient_weight=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        cnn.parameters(),
        lr=0.0001,
    )

    output_dir = (
        Path("outputs/vdmamba/cnn_baseline")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cnn_params = sum(
        p.numel() for p in cnn.parameters()
    )

    print("=" * 60)
    print("CNN BASELINE EXPERIMENT")
    print("=" * 60)
    print("Device:", device)
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("CNN parameters:", cnn_params)

    best_psnr = float("-inf")
    best_metrics = None

    for epoch in range(1, 6):
        cnn.train()
        losses = []

        for batch in train_loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = cnn(rainy)
            loss = criterion(prediction, clean)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                cnn.parameters(), 1.0
            )

            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(cnn, val_loader, device)

        print(
            f"Epoch {epoch}/5 | "
            f"Loss: {sum(losses)/len(losses):.6f} | "
            f"PSNR: {metrics['restored_psnr']:.4f} | "
            f"Gain: {metrics['psnr_gain']:+.4f} dB"
        )

        if metrics["restored_psnr"] > best_psnr:
            best_psnr = metrics["restored_psnr"]
            best_metrics = metrics.copy()

            torch.save(
                {
                    "model_state_dict": cnn.state_dict(),
                    "epoch": epoch,
                    "channels": 8,
                    "metrics": metrics,
                    "model_type": "simple_cnn_baseline",
                },
                output_dir / "best.pt",
            )

    vdmamba = load_vdmamba(device)
    vdmamba_metrics = evaluate(
        vdmamba, val_loader, device
    )

    vdmamba_params = sum(
        p.numel() for p in vdmamba.parameters()
    )

    report = {
        "cnn": {
            "parameter_count": cnn_params,
            **best_metrics,
        },
        "vdmamba": {
            "parameter_count": vdmamba_params,
            **vdmamba_metrics,
        },
        "comparison_note": (
            "Same 64-image training count and validation "
            "split. Architecture and parameter counts differ."
        ),
    }

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue

        print(f"\n{name.upper()}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nResults:", output_dir)
    print("Test split used: No")
    print("Existing checkpoints modified: No")


if __name__ == "__main__":
    main()
