
"""
Multi-image training for the experimental VDMamba-inspired model.

Features:
- RealRain-1k Heavy-rain training and validation
- Paired random training crops
- Deterministic validation crops
- Charbonnier + optional SSIM + optional Gradient loss
- Adam optimizer
- Gradient clipping
- Pooled PSNR validation
- Best-checkpoint selection using validation PSNR
- Unique output directory for every training run
- Training history and configuration saving

This is an experimental VDMamba-inspired implementation,
not the official pretrained VDMamba model.

The test split is never used.
Existing PReNet and severity checkpoints are not modified.
"""

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)
from vision.deraining.vdmamba.restoration_loss import (
    RestorationLoss,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.inference_mode()
def validate(model, loader, device):
    """
    Evaluate model performance using pooled pixel MSE.

    Handles validation patches where rainy and clean
    images are identical without producing infinite
    per-image PSNR averages.

    Returns:
        rainy_psnr
        restored_psnr
        psnr_gain
        improved
        count
    """

    model.eval()

    rainy_squared_error = 0.0
    restored_squared_error = 0.0
    total_pixels = 0

    improved = 0
    count = 0

    for batch in loader:

        rainy = batch["rainy"].to(device)
        clean = batch["clean"].to(device)

        prediction = model(rainy)

        if prediction.shape != clean.shape:
            raise RuntimeError(
                "Validation output shape mismatch."
            )

        if not torch.isfinite(prediction).all().item():
            raise RuntimeError(
                "Non-finite validation prediction."
            )

        rainy_mse = (
            (rainy.float() - clean.float())
            .square()
            .mean()
            .item()
        )

        restored_mse = (
            (
                prediction.float().clamp(0, 1)
                - clean.float()
            )
            .square()
            .mean()
            .item()
        )

        if not (
            math.isfinite(rainy_mse)
            and math.isfinite(restored_mse)
        ):
            raise RuntimeError(
                "Non-finite validation MSE."
            )

        pixels = clean.numel()

        rainy_squared_error += rainy_mse * pixels
        restored_squared_error += restored_mse * pixels

        total_pixels += pixels

        if restored_mse < rainy_mse:
            improved += 1

        count += 1

    if count == 0 or total_pixels == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    pooled_rainy_mse = (
        rainy_squared_error / total_pixels
    )

    pooled_restored_mse = (
        restored_squared_error / total_pixels
    )

    if pooled_rainy_mse == 0:
        raise RuntimeError(
            "Entire validation baseline has zero MSE. "
            "Cannot calculate a finite PSNR gain."
        )

    def mse_to_psnr(mse):
        if mse == 0:
            return float("inf")

        return -10.0 * math.log10(mse)

    rainy_psnr = mse_to_psnr(
        pooled_rainy_mse
    )

    restored_psnr = mse_to_psnr(
        pooled_restored_mse
    )

    return {
        "rainy_psnr": rainy_psnr,
        "restored_psnr": restored_psnr,
        "psnr_gain": restored_psnr - rainy_psnr,
        "improved": improved,
        "count": count,
    }


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the experimental VDMamba-inspired "
            "Heavy-rain restoration model."
        )
    )

    # -----------------------------------------------------
    # Training configuration
    # -----------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--train-count",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--val-count",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
    )

    # -----------------------------------------------------
    # Model configuration
    # -----------------------------------------------------

    parser.add_argument(
        "--channels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--state-size",
        type=int,
        default=4,
    )

    # -----------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    # -----------------------------------------------------
    # Restoration loss configuration
    # -----------------------------------------------------

    # 0.0 = Charbonnier only
    # 0.02 = Charbonnier + SSIM
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=0.0,
    )

    # NEW:
    # 0.0 = Gradient loss disabled
    # 0.05 = Initial gradient-aware experiment
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.0,
    )

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # -----------------------------------------------------
    # Hardware configuration
    # -----------------------------------------------------

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate arguments
    # -----------------------------------------------------

    if args.epochs < 1:
        parser.error(
            "--epochs must be >= 1"
        )

    if not 1 <= args.train_count <= 784:
        parser.error(
            "--train-count must be between 1 and 784"
        )

    if not 1 <= args.val_count <= 112:
        parser.error(
            "--val-count must be between 1 and 112"
        )

    if args.patch_size < 7:
        parser.error(
            "--patch-size must be >= 7 "
            "for the current SSIM window."
        )

    if args.channels < 1:
        parser.error(
            "--channels must be positive"
        )

    if args.state_size < 1:
        parser.error(
            "--state-size must be positive"
        )

    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error(
            "--lr must be positive and finite"
        )

    if not (
        math.isfinite(args.ssim_weight)
        and 0.0 <= args.ssim_weight <= 1.0
    ):
        parser.error(
            "--ssim-weight must be between 0 and 1"
        )

    # NEW: Validate gradient weight
    if not (
        math.isfinite(args.gradient_weight)
        and 0.0 <= args.gradient_weight <= 1.0
    ):
        parser.error(
            "--gradient-weight must be between 0 and 1"
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA is unavailable on this system."
        )

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # -----------------------------------------------------
    # Load paired Heavy-rain data
    # -----------------------------------------------------

    train_data = HeavyRainDataset(
        split="train",
        patch_size=args.patch_size,
        augment=True,
        limit=args.train_count,
    )

    val_data = HeavyRainDataset(
        split="validation",
        patch_size=args.patch_size,
        augment=False,
        limit=args.val_count,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------
    # Initialize model
    # -----------------------------------------------------

    model = ExperimentalVDMambaImageModel(
        channels=args.channels,
        state_size=args.state_size,
    ).to(device)

    # -----------------------------------------------------
    # Initialize optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------------------------------------
    # Initialize restoration loss
    # -----------------------------------------------------

    # NEW:
    # Includes the optional gradient reconstruction loss.
    criterion = RestorationLoss(
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight,
    ).to(device)

    # -----------------------------------------------------
    # Create experiment directory
    # -----------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        Path("outputs/vdmamba/multi_train")
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = vars(args).copy()

    config["model_status"] = (
        "experimental_vdmamba_inspired"
    )

    # NEW:
    # Identify which loss components were enabled.
    config["loss_function"] = (
        "charbonnier"
        + (
            "_plus_ssim"
            if args.ssim_weight > 0
            else ""
        )
        + (
            "_plus_gradient"
            if args.gradient_weight > 0
            else ""
        )
    )

    config["validation_metric"] = (
        "pooled_psnr_from_pixel_mse"
    )

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Training setup
    # -----------------------------------------------------

    history = []

    best_val_psnr = float("-inf")
    best_epoch = None

    print("=" * 65)
    print(
        "EXPERIMENTAL VDMAMBA MULTI-IMAGE TRAINING"
    )
    print("=" * 65)

    print("Device:", device)
    print("Training pairs:", len(train_data))
    print("Validation pairs:", len(val_data))
    print("Patch size:", args.patch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Random seed:", args.seed)

    print(
        "Loss:",
        config["loss_function"],
    )

    print(
        "SSIM weight:",
        args.ssim_weight,
    )

    print(
        "Gradient weight:",
        args.gradient_weight,
    )

    print(
        "Output directory:",
        output_dir,
    )

    print("=" * 65)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        model.train()

        losses = []

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(rainy)

            if not torch.isfinite(
                prediction
            ).all().item():
                raise RuntimeError(
                    "Non-finite prediction at "
                    f"epoch {epoch}, step {step}"
                )

            # Includes Charbonnier, SSIM,
            # and gradient loss when enabled.
            loss = criterion(
                prediction,
                clean,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite training loss at "
                    f"epoch {epoch}, step {step}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        metrics = validate(
            model,
            val_loader,
            device,
        )

        train_loss = (
            sum(losses) / len(losses)
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }

        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.6f} | "
            f"Rainy PSNR: "
            f"{metrics['rainy_psnr']:.4f} | "
            f"Restored PSNR: "
            f"{metrics['restored_psnr']:.4f} | "
            f"Gain: "
            f"{metrics['psnr_gain']:+.4f} dB | "
            f"Improved: "
            f"{metrics['improved']}/"
            f"{metrics['count']}",
            flush=True,
        )

        # -------------------------------------------------
        # Save best checkpoint
        # -------------------------------------------------

        if (
            math.isfinite(
                metrics["restored_psnr"]
            )
            and metrics["restored_psnr"]
            > best_val_psnr
        ):

            best_val_psnr = metrics[
                "restored_psnr"
            ]

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "config": config,
                    "validation": metrics,
                    "model_status": (
                        "experimental_vdmamba_inspired"
                    ),
                },
                output_dir / "best.pt",
            )

        # -------------------------------------------------
        # Save training history
        # -------------------------------------------------

        history_path = (
            output_dir / "history.csv"
        )

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "rainy_psnr",
                    "restored_psnr",
                    "psnr_gain",
                    "improved",
                    "count",
                ],
            )

            writer.writeheader()
            writer.writerows(history)

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(
        "Best validation PSNR:",
        round(best_val_psnr, 4),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Results:",
        output_dir,
    )

    print(
        "Checkpoint:",
        output_dir / "best.pt",
    )

    print(
        "\nNOTE: This is an experimental VDMamba-inspired "
        "model, not the original published checkpoint."
    )

    print(
        "Validation uses fixed center crops, "
        "not complete rainy images."
    )


if __name__ == "__main__":
    main()
