
"""
Adaptive Vision Intelligence for Rainy Weather Environments

Phase 3: Binary Rain Severity Classification Training

Architecture: MobileNetV3-Small
Classes:
    0 - Light
    1 - Heavy

Features:
    - Dataset readiness checks
    - ImageNet transfer learning
    - Frozen backbone training
    - CrossEntropyLoss
    - Adam optimizer
    - Accuracy, precision, recall, F1-score
    - Confusion matrix
    - Early stopping
    - Best and latest checkpoint saving
    - Training history
    - CUDA memory monitoring

Safety:
    Running without --execute-training performs readiness
    checks only.

    Training requires both:
        --execute-training
        --pretrained
"""

import argparse
import csv
import json
import os
import random
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torch import nn

from vision.severity.dataset import create_severity_dataloader
from vision.severity.model import (
    CLASS_NAMES,
    create_severity_model,
)


# ============================================================
# 1. TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainConfig:
    manifest: Path = Path("docs/severity_labels.csv")
    data_root: Path = Path("datasets")

    checkpoint_dir: Path = Path("checkpoints/severity")

    epochs: int = 20
    batch_size: int = 4

    learning_rate: float = 0.0001

    patience: int = 5
    min_delta: float = 0.0001

    seed: int = 42
    device: str = "auto"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DEVICE SELECTION
# ============================================================

def get_device(device_name: str) -> torch.device:
    """Select CPU or CUDA device."""

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name not in ("cpu", "cuda"):
        raise ValueError(
            "Device must be one of: auto, cpu, cuda."
        )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return torch.device(device_name)


# ============================================================
# 4. DATASET READINESS CHECK
# ============================================================

def check_data_readiness(config: TrainConfig) -> dict:
    """
    Verify that all training and validation images exist.

    The official RealRain-1k split distribution is preserved.
    Test images are not accessed.
    """

    expected_counts = {
        ("train", "Light"): 784,
        ("train", "Heavy"): 784,
        ("validation", "Light"): 112,
        ("validation", "Heavy"): 112,
    }

    if not config.manifest.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {config.manifest}"
        )

    if not config.data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {config.data_root}"
        )

    counts = {
        key: 0
        for key in expected_counts
    }

    missing_files = []

    seen_paths = set()
    scene_splits = {}

    with config.manifest.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "split",
            "severity",
            "rainy_path",
            "scene_group_id",
        }

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Severity manifest is missing required columns."
            )

        for row in reader:

            split = row["split"]
            severity = row["severity"]

            # Test split must not participate in training.
            if split == "test":
                continue

            key = (split, severity)

            if key not in expected_counts:
                raise ValueError(
                    f"Unexpected split/severity combination: {key}"
                )

            relative_path = Path(row["rainy_path"])

            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise ValueError(
                    f"Unsafe dataset path: {relative_path}"
                )

            if str(relative_path) in seen_paths:
                raise ValueError(
                    f"Duplicate image path: {relative_path}"
                )

            seen_paths.add(str(relative_path))

            counts[key] += 1

            scene_group_id = row["scene_group_id"]

            if scene_group_id in scene_splits:

                if scene_splits[scene_group_id] != split:
                    raise ValueError(
                        "Scene leakage detected across splits."
                    )

            else:
                scene_splits[scene_group_id] = split

            image_path = (
                config.data_root / relative_path
            )

            if not image_path.is_file():
                missing_files.append(
                    str(relative_path)
                )

    counts_correct = (
        counts == expected_counts
    )

    ready = (
        counts_correct
        and len(missing_files) == 0
    )

    return {
        "ready": ready,
        "counts": {
            f"{split}/{severity}": counts[
                (split, severity)
            ]
            for split, severity in expected_counts
        },
        "missing_count": len(missing_files),
        "first_missing": missing_files[:5],
        "expected_images": sum(
            expected_counts.values()
        ),
    }


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix.

    Rows represent actual classes.
    Columns represent predicted classes.
    """

    if (
        logits.ndim != 2
        or logits.shape[1] != 2
    ):
        raise ValueError(
            "Expected logits of shape (batch_size, 2)."
        )

    if (
        targets.ndim != 1
        or len(targets) != len(logits)
    ):
        raise ValueError(
            "Target tensor dimensions do not match logits."
        )

    if torch.any(
        (targets < 0) | (targets > 1)
    ).item():
        raise ValueError(
            "Targets must contain only 0 or 1."
        )

    predictions = (
        logits.argmax(dim=1)
        .detach()
        .cpu()
    )

    targets = (
        targets.detach()
        .cpu()
        .long()
    )

    encoded = targets * 2 + predictions.long()

    matrix = torch.bincount(
        encoded,
        minlength=4,
    ).reshape(2, 2)

    return matrix


# ============================================================
# 6. CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    matrix: torch.Tensor,
) -> dict:
    """Calculate binary classification evaluation metrics."""

    if matrix.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    values = matrix.float()

    def safe_divide(numerator, denominator):
        if denominator.item() == 0:
            return 0.0

        return float(
            (numerator / denominator).item()
        )

    per_class = {}

    for index, class_name in enumerate(CLASS_NAMES):

        true_positive = values[index, index]

        total_predicted = values[:, index].sum()
        total_actual = values[index, :].sum()

        precision = safe_divide(
            true_positive,
            total_predicted,
        )

        recall = safe_divide(
            true_positive,
            total_actual,
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(total_actual.item()),
        }

    accuracy = safe_divide(
        values.diag().sum(),
        values.sum(),
    )

    macro_f1 = sum(
        class_metrics["f1"]
        for class_metrics in per_class.values()
    ) / len(CLASS_NAMES)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


# ============================================================
# 7. EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Stop training when validation macro-F1 stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0001,
    ):

        if patience < 1:
            raise ValueError(
                "Patience must be at least 1."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        self.patience = patience
        self.min_delta = min_delta

        self.best = float("-inf")
        self.counter = 0

    def step(self, score: float):
        """Update early stopping state."""

        if not np.isfinite(score):
            raise ValueError(
                "Validation score must be finite."
            )

        improved = (
            score > self.best + self.min_delta
        )

        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1

        should_stop = (
            self.counter >= self.patience
        )

        return improved, should_stop


# ============================================================
# 8. CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
):
    """
    Save model and optimizer states atomically.

    Prevents partially written checkpoint files.
    """

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_f1": float(score),
        "classes": list(CLASS_NAMES),
        "architecture": "mobilenet_v3_small",
        "preprocessing": "IMAGENET1K_V1",
        "backbone_frozen": getattr(
            model,
            "backbone_frozen",
            True,
        ),
    }

    file_descriptor, temporary_path = (
        tempfile.mkstemp(
            dir=checkpoint_path.parent,
            suffix=".tmp",
        )
    )

    os.close(file_descriptor)

    try:
        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            checkpoint_path,
        )

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


# ============================================================
# 9. TRAINING / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> dict:
    """
    Run one training or validation epoch.

    If optimizer is None, performs validation only.
    """

    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0.0
    total_images = 0

    total_confusion_matrix = torch.zeros(
        (2, 2),
        dtype=torch.long,
    )

    with torch.set_grad_enabled(is_training):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = len(labels)

            total_loss += (
                loss.detach().item()
                * batch_size
            )

            total_images += batch_size

            batch_matrix = confusion_matrix(
                logits.detach(),
                labels.detach(),
            )

            total_confusion_matrix += batch_matrix

    if total_images == 0:
        raise RuntimeError(
            "DataLoader contains no images."
        )

    average_loss = (
        total_loss / total_images
    )

    metrics = calculate_metrics(
        total_confusion_matrix
    )

    return {
        "loss": average_loss,
        **metrics,
    }


# ============================================================
# 10. MAIN TRAINING FUNCTION
# ============================================================

def train(
    config: TrainConfig,
    pretrained: bool,
):
    """Train MobileNetV3-Small using original Light/Heavy labels."""

    if not pretrained:
        raise RuntimeError(
            "Training requires --pretrained."
        )

    if (
        config.epochs < 1
        or config.batch_size < 1
        or config.learning_rate <= 0
    ):
        raise ValueError(
            "Invalid training configuration."
        )

    readiness = check_data_readiness(config)

    if not readiness["ready"]:
        raise RuntimeError(
            "Dataset is incomplete. "
            f"Readiness report: {readiness}"
        )

    history_path = (
        config.checkpoint_dir / "history.jsonl"
    )

    if history_path.exists():
        raise FileExistsError(
            "Existing training history detected. "
            "Archive the previous experiment before "
            "starting another."
        )

    set_seed(config.seed)

    device = get_device(config.device)

    print("=" * 65)
    print("ADAPTIVE VISION INTELLIGENCE")
    print("BINARY RAIN SEVERITY MODEL TRAINING")
    print("=" * 65)

    print("Device:", device)
    print("Classes:", CLASS_NAMES)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print("Maximum epochs:", config.epochs)

    print("\nLoading ImageNet pretrained MobileNetV3-Small...")

    # Pretrained weights are requested only when
    # training is explicitly started.
    model = create_severity_model(
        pretrained=True,
        freeze_backbone=True,
        device=str(device),
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    train_loader = create_severity_dataloader(
        manifest_path=config.manifest,
        dataset_root=config.data_root,
        split="train",
        batch_size=config.batch_size,
        allow_partial=False,
        shuffle=True,
    )

    validation_loader = create_severity_dataloader(
        manifest_path=config.manifest,
        dataset_root=config.data_root,
        split="validation",
        batch_size=config.batch_size,
        allow_partial=False,
        shuffle=False,
    )

    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
    )

    config.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        training_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        validation_macro_f1 = (
            validation_metrics["macro_f1"]
        )

        improved, should_stop = (
            early_stopping.step(
                validation_macro_f1
            )
        )

        record = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
        }

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as history_file:

            history_file.write(
                json.dumps(record) + "\n"
            )

        # Most recent checkpoint
        save_checkpoint(
            path=config.checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            score=early_stopping.best,
        )

        # Best validation checkpoint
        if improved:
            save_checkpoint(
                path=config.checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                score=early_stopping.best,
            )

        print(
            f"\nEpoch {epoch}/{config.epochs}"
        )

        print(
            f"Training Loss: "
            f"{training_metrics['loss']:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Macro-F1: "
            f"{validation_macro_f1:.4f}"
        )

        if device.type == "cuda":

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

            print(
                f"Peak allocated GPU memory: "
                f"{peak_memory:.2f} MiB"
            )

        if improved:
            print(
                "Best model checkpoint updated."
            )

        if should_stop:
            print(
                "\nEarly stopping triggered."
            )
            break

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(
        "Best checkpoint:",
        config.checkpoint_dir / "best.pt",
    )

    print(
        "Last checkpoint:",
        config.checkpoint_dir / "last.pt",
    )

    print(
        "Training history:",
        history_path,
    )

    print(
        "\nTest dataset was not accessed."
    )


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Binary Light/Heavy rain severity training."
        )
    )

    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Explicitly start model training.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Safe default: do not train automatically.
    if not args.execute_training:

        print("=" * 65)
        print("TRAINING READINESS CHECK ONLY")
        print("=" * 65)

        print(
            "No model training or pretrained weight downloads."
        )

        readiness = check_data_readiness(
            config
        )

        print(
            json.dumps(
                readiness,
                indent=2,
            )
        )

        return

    train(
        config=config,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
