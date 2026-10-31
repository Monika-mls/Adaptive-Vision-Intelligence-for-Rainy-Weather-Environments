
"""
Memory-safe Restormer fine-tuning experiment.

Uses pretrained official Restormer weights.
Trains only the output convolution.
Uses Heavy-rain training and validation data.

No test data, no modification to original checkpoint.
"""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vision.deraining.restormer_adapter import load_model
from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    restored_error = 0.0
    rainy_error = 0.0
    pixels = 0
    improved = 0

    for batch in loader:
        rainy = batch["rainy"].to(device)
        clean = batch["clean"].to(device)

        prediction = model(rainy).clamp(0, 1)

        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("Non-finite validation prediction")

        restored_mse = (prediction - clean).square()
        rainy_mse = (rainy - clean).square()

        restored_error += restored_mse.sum().item()
        rainy_error += rainy_mse.sum().item()
        pixels += clean.numel()

        improved += int(
            restored_mse.mean().item()
            < rainy_mse.mean().item()
        )

    def psnr(error):
        mse = error / pixels
        return (
            float("inf")
            if mse == 0
            else -10 * math.log10(mse)
        )

    rainy_psnr = psnr(rainy_error)
    restored_psnr = psnr(restored_error)

    return {
        "rainy_psnr": rainy_psnr,
        "restored_psnr": restored_psnr,
        "psnr_gain": restored_psnr - rainy_psnr,
        "improved_patches": improved,
        "validation_count": len(loader.dataset),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    if not 1 <= args.train_count <= 784:
        parser.error("--train-count must be between 1 and 784")

    if not 1 <= args.val_count <= 112:
        parser.error("--val-count must be between 1 and 112")

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.patch_size < 32 or args.patch_size % 8:
        parser.error(
            "--patch-size must be >=32 and divisible by 8"
        )

    if not 0 < args.lr <= 1e-3:
        parser.error("--lr must be in (0, 0.001]")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto"
        and torch.cuda.is_available()
        else (
            "cpu" if args.device == "auto"
            else args.device
        )
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print("=" * 60, flush=True)
    print("RESTORMER HEAVY-RAIN FINE-TUNING", flush=True)
    print("=" * 60, flush=True)
    print("Device:", device, flush=True)
    print("Training images:", args.train_count, flush=True)
    print("Validation images:", args.val_count, flush=True)
    print("Patch size:", args.patch_size, flush=True)

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
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Loading official pretrained Restormer...", flush=True)

    model = load_model(device)

    # Freeze the entire pretrained network.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Fine-tune only the final image-restoration convolution.
    for parameter in model.output.parameters():
        parameter.requires_grad_(True)

    model.eval()

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total parameters:", total_params, flush=True)
    print("Trainable parameters:", trainable, flush=True)

    optimizer = torch.optim.AdamW(
        model.output.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    output_dir = (
        Path("outputs/restormer/finetune")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    print("\nEvaluating ORIGINAL checkpoint...", flush=True)
    original = evaluate(model, val_loader, device)

    print(
        "Original PSNR:",
        f"{original['restored_psnr']:.4f}",
        flush=True,
    )
    print(
        "Original gain:",
        f"{original['psnr_gain']:+.4f} dB",
        flush=True,
    )

    best_psnr = original["restored_psnr"]
    best_epoch = 0
    history = []

    # Only record a fine-tuned checkpoint if validation improves.
    for epoch in range(1, args.epochs + 1):
        model.eval()
        losses = []

        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        for step, batch in enumerate(train_loader, start=1):
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # The frozen backbone has no parameter gradients.
            # Only model.output receives gradient updates.
            prediction = model(rainy)

            loss = F.l1_loss(prediction, clean)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.output.parameters(), 1.0
            )

            optimizer.step()
            losses.append(loss.item())

            if step == 1 or step % 16 == 0:
                print(
                    f"  Batch {step}/{len(train_loader)} "
                    f"Loss: {loss.item():.6f}",
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "training_loss": sum(losses) / len(losses),
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch} | "
            f"Loss: {row['training_loss']:.6f} | "
            f"PSNR: {metrics['restored_psnr']:.4f} | "
            f"Gain: {metrics['psnr_gain']:+.4f} dB | "
            f"Improved: {metrics['improved_patches']}/{args.val_count}",
            flush=True,
        )

        if metrics["restored_psnr"] > best_psnr:
            best_psnr = metrics["restored_psnr"]
            best_epoch = epoch

            torch.save(
                {
                    "params": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "config": vars(args),
                    "validation": metrics,
                    "model_status": "restormer_output_layer_finetuned",
                },
                output_dir / "best.pt",
            )

    report = {
        "experiment": "restormer_output_layer_finetuning",
        "original": original,
        "best_psnr": best_psnr,
        "best_epoch": best_epoch,
        "trainable_parameters": trainable,
        "total_parameters": total_params,
        "history": history,
        "test_split_used": False,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"Original PSNR: {original['restored_psnr']:.4f} dB"
    )
    print(f"Best PSNR:     {best_psnr:.4f} dB")
    print(f"Best epoch:    {best_epoch}")
    print(f"Results:       {output_dir}")

    if best_epoch == 0:
        print(
            "Fine-tuning did not improve validation PSNR. "
            "Original pretrained weights remain preferred."
        )
    else:
        print(
            "Fine-tuned checkpoint saved as best.pt. "
            "Visual rain removal still needs verification."
        )

    if device.type == "cuda":
        print(
            "Peak allocated GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**2,
                2,
            ),
            "MiB",
        )

    print("Original checkpoint modified: No")
    print("Test split used: No")


if __name__ == "__main__":
    main()
