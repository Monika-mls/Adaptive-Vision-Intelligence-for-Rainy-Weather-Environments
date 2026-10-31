
"""
Adaptive Vision Intelligence for Rainy Weather Environments
Phase 3 - Step 18: Fine-tuning with safe checkpoint resumption.

DEFAULT:
    Read-only preflight.

NEW TRAINING:
    --execute-finetuning

RESUME EXISTING EXPERIMENT:
    --execute-finetuning --resume

Safeguards:
    - Never overwrites the original baseline checkpoint.
    - Validates checkpoint, history, and experiment settings.
    - Restores model, Adam optimizer and early stopping.
    - Continues at last completed epoch + 1.
    - Preserves the existing best checkpoint.
    - Displays batch-level progress.
    - Uses only train and validation data.
    - Never downloads model weights or test images.
"""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vision.severity.dataset import create_severity_dataloader
from vision.severity.model import create_severity_model
from vision.severity.predict import (
    CLASS_NAMES,
    validate_checkpoint,
)
from vision.severity.train import (
    EarlyStopping,
    TrainConfig,
    calculate_metrics,
    check_data_readiness,
    confusion_matrix,
    get_device,
    set_seed,
)


BASELINE_SHA256 = (
    "9f563076d8cdc5e0fa602e16787388991430867af2c98a325dfd58678fcda62b"
)

EXPERIMENT_FIELDS = (
    "manifest",
    "data_root",
    "source_checkpoint",
    "output_dir",
    "batch_size",
    "backbone_lr",
    "head_lr",
    "unfreeze_last_blocks",
    "patience",
    "min_delta",
    "horizontal_flip_probability",
    "brightness_strength",
    "use_augmentation",
    "seed",
)


@dataclass
class FineTuneConfig:
    manifest: Path = Path("docs/severity_labels.csv")
    data_root: Path = Path("datasets")

    source_checkpoint: Path = Path(
        "checkpoints/severity/best.pt"
    )

    output_dir: Path = Path(
        "checkpoints/severity/finetune_run_01"
    )

    epochs: int = 12
    batch_size: int = 4

    backbone_lr: float = 1e-5
    head_lr: float = 1e-4

    unfreeze_last_blocks: int = 2

    patience: int = 4
    min_delta: float = 1e-4

    horizontal_flip_probability: float = 0.5
    brightness_strength: float = 0.05

    use_augmentation: bool = True

    seed: int = 42
    device: str = "cuda"

    log_every: int = 25


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_config(config, resume=False):
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("Invalid epochs or batch size.")

    if config.backbone_lr <= 0 or config.head_lr <= 0:
        raise ValueError("Learning rates must be positive.")

    if config.unfreeze_last_blocks < 1:
        raise ValueError("Invalid number of unfrozen blocks.")

    if config.patience < 1 or config.min_delta < 0:
        raise ValueError("Invalid early stopping settings.")

    if config.log_every < 1:
        raise ValueError("log_every must be positive.")

    if not 0 <= config.horizontal_flip_probability <= 1:
        raise ValueError("Invalid horizontal flip probability.")

    if not 0 <= config.brightness_strength <= 0.1:
        raise ValueError("Invalid brightness strength.")

    source = config.source_checkpoint.resolve()
    output = config.output_dir.resolve()

    if source == output or source in output.parents:
        raise ValueError(
            "Experiment output cannot replace source checkpoint."
        )

    if output == source.parent:
        raise ValueError(
            "Fine-tuning requires a separate output directory."
        )

    if resume:
        if not output.is_dir():
            raise FileNotFoundError(
                f"Resume directory does not exist: {output}"
            )
    else:
        if output.exists():
            raise FileExistsError(
                f"Experiment output already exists: {output}"
            )

    return True


def read_history(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("Training history is empty.")

    for expected_epoch, record in enumerate(records, 1):
        if record.get("epoch") != expected_epoch:
            raise ValueError(
                "History contains missing or duplicate epochs."
            )

        validation = record.get("validation", {})
        score = validation.get("macro_f1")

        if not isinstance(score, (int, float)):
            raise ValueError("Invalid validation macro-F1.")

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Non-finite validation macro-F1.")

    return records


def reconstruct_early_stopping(records, patience, min_delta):
    stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
    )

    best_epoch = None
    stopped = False

    for record in records:
        improved, should_stop = stopper.step(
            record["validation"]["macro_f1"]
        )

        if improved:
            best_epoch = record["epoch"]

        if should_stop:
            stopped = True

            if record != records[-1]:
                raise ValueError(
                    "History continues after early stopping."
                )

    return stopper, best_epoch, stopped


def validate_saved_config(config):
    config_path = config.output_dir / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open(encoding="utf-8") as file:
        saved = json.load(file)

    current = asdict(config)

    for field in EXPERIMENT_FIELDS:
        if field not in saved:
            raise ValueError(
                f"Saved experiment config missing: {field}"
            )

        old = saved[field]
        new = current[field]

        if isinstance(new, Path):
            old = str(Path(old).resolve())
            new = str(new.resolve())

        if old != new:
            raise ValueError(
                f"Resume setting mismatch for '{field}': "
                f"saved={old!r}, requested={new!r}"
            )

    if saved.get("source_checkpoint_sha256") != BASELINE_SHA256:
        raise ValueError(
            "Saved experiment baseline fingerprint mismatch."
        )

    # Epoch limit may be increased, but never reduced
    # below the original intended maximum.
    if config.epochs < saved.get("epochs", 0):
        raise ValueError(
            "Cannot reduce original experiment epoch limit."
        )

    return saved


def inspect_resume_state(config):
    """
    Read and validate existing experiment files.

    Does not update checkpoints or history.
    """

    root = config.output_dir

    saved_config = validate_saved_config(config)

    last_path = root / "last.pt"
    best_path = root / "best.pt"
    history_path = root / "history.jsonl"

    for path in (last_path, best_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = read_history(history_path)

    last_checkpoint = torch.load(
        last_path,
        map_location="cpu",
        weights_only=True,
    )

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    validate_checkpoint(last_checkpoint)
    validate_checkpoint(best_checkpoint)

    for name, checkpoint in (
        ("last", last_checkpoint),
        ("best", best_checkpoint),
    ):
        if checkpoint.get("experiment_type") != "partial_finetuning":
            raise ValueError(
                f"Unexpected {name} checkpoint experiment type."
            )

        if checkpoint.get("baseline_sha256") != BASELINE_SHA256:
            raise ValueError(
                f"{name} checkpoint baseline hash mismatch."
            )

        if checkpoint.get(
            "unfrozen_last_feature_blocks"
        ) != config.unfreeze_last_blocks:
            raise ValueError(
                f"{name} checkpoint unfreezing mismatch."
            )

    last_epoch = last_checkpoint.get("epoch")

    if last_epoch != records[-1]["epoch"]:
        raise ValueError(
            "last.pt epoch does not match history.jsonl."
        )

    optimizer_state = last_checkpoint.get(
        "optimizer_state_dict"
    )

    if not isinstance(optimizer_state, dict):
        raise ValueError("Optimizer state is missing.")

    if len(optimizer_state.get("param_groups", [])) != 2:
        raise ValueError(
            "Expected two Adam optimizer parameter groups."
        )

    stopper, best_epoch, already_stopped = (
        reconstruct_early_stopping(
            records,
            config.patience,
            config.min_delta,
        )
    )

    if best_checkpoint.get("epoch") != best_epoch:
        raise ValueError(
            "Best checkpoint epoch does not match history."
        )

    for name, checkpoint in (
        ("last", last_checkpoint),
        ("best", best_checkpoint),
    ):
        if not math.isclose(
            float(checkpoint["best_macro_f1"]),
            float(stopper.best),
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"{name} checkpoint best score mismatch."
            )

    if last_epoch >= config.epochs:
        raise ValueError(
            "Experiment has already reached max epochs."
        )

    if already_stopped:
        raise ValueError(
            "Early stopping was already triggered. "
            "Do not resume this completed experiment."
        )

    return {
        "saved_config": saved_config,
        "history": records,
        "last_checkpoint": last_checkpoint,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "next_epoch": last_epoch + 1,
        "early_stopping": stopper,
    }


def check_finetune_readiness(config, resume=False):
    validate_config(config, resume=resume)

    data = check_data_readiness(
        TrainConfig(
            manifest=config.manifest,
            data_root=config.data_root,
        )
    )

    if not data["ready"]:
        raise RuntimeError(
            f"Dataset is incomplete: {data}"
        )

    if not config.source_checkpoint.is_file():
        raise FileNotFoundError(
            config.source_checkpoint
        )

    actual_hash = sha256_file(
        config.source_checkpoint
    )

    if actual_hash != BASELINE_SHA256:
        raise ValueError(
            "Original baseline checkpoint SHA256 mismatch."
        )

    baseline = torch.load(
        config.source_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    validate_checkpoint(baseline)

    if baseline.get("epoch") != 8:
        raise ValueError(
            "Expected original epoch-8 baseline."
        )

    report = {
        "ready": True,
        "resume_requested": resume,
        "source_epoch": baseline["epoch"],
        "source_macro_f1": baseline.get(
            "best_macro_f1"
        ),
        "baseline_sha256": actual_hash,
        "dataset": data,
        "output_dir": str(
            config.output_dir.resolve()
        ),
        "training_started": False,
        "model_downloads": False,
        "test_split_used": False,
    }

    if resume:
        state = inspect_resume_state(config)

        report["last_completed_epoch"] = state[
            "last_epoch"
        ]
        report["next_epoch"] = state["next_epoch"]
        report["best_finetune_epoch"] = state[
            "best_epoch"
        ]
        report["best_finetune_macro_f1"] = state[
            "early_stopping"
        ].best
        report["early_stop_counter"] = state[
            "early_stopping"
        ].counter

    return report


def prepare_finetuning_model(
    model,
    unfreeze_last_blocks=2,
):
    if not hasattr(model, "model"):
        raise TypeError(
            "Expected model wrapper exposing .model."
        )

    backbone = model.model

    if (
        not hasattr(backbone, "features")
        or not hasattr(backbone, "classifier")
    ):
        raise TypeError(
            "Expected MobileNetV3 architecture."
        )

    blocks = list(backbone.features.children())

    if not 1 <= unfreeze_last_blocks <= len(blocks):
        raise ValueError(
            "Invalid number of blocks to unfreeze."
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for block in blocks[-unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)

    for parameter in backbone.classifier.parameters():
        parameter.requires_grad_(True)

    backbone_params = [
        p for p in backbone.features.parameters()
        if p.requires_grad
    ]

    head_params = list(
        backbone.classifier.parameters()
    )

    if not backbone_params or not head_params:
        raise RuntimeError(
            "Required trainable parameters are missing."
        )

    if hasattr(model, "backbone_frozen"):
        model.backbone_frozen = False

    return backbone_params, head_params


def set_finetune_train_mode(
    model,
    unfreeze_last_blocks,
):
    model.train()

    blocks = list(model.model.features.children())

    for block in blocks[:-unfreeze_last_blocks]:
        block.eval()

    for block in blocks[-unfreeze_last_blocks:]:
        block.train()

    model.model.classifier.train()


def augment_training_batch(
    images,
    flip_probability=0.5,
    brightness_strength=0.05,
):
    """
    Train-only augmentation of ImageNet-normalized tensors.

    Avoids Python-side GPU synchronization for flip masks.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "Expected normalized [N, 3, H, W] tensors."
        )

    if not 0 <= flip_probability <= 1:
        raise ValueError("Invalid flip probability.")

    if not 0 <= brightness_strength <= 0.1:
        raise ValueError("Invalid brightness strength.")

    result = images.clone()
    n = result.shape[0]

    if flip_probability > 0:
        flip_mask = (
            torch.rand(n, device=result.device)
            < flip_probability
        ).view(n, 1, 1, 1)

        # No mask.any().item() synchronization.
        result = torch.where(
            flip_mask,
            torch.flip(result, dims=[-1]),
            result,
        )

    if brightness_strength > 0:
        mean = result.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = result.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        rgb = result * std + mean

        scale = 1.0 + (
            2 * torch.rand(
                n, 1, 1, 1,
                device=result.device,
            ) - 1.0
        ) * brightness_strength

        rgb = (rgb * scale).clamp(0, 1)
        result = (rgb - mean) / std

    return result


def run_finetune_epoch(
    model,
    loader,
    criterion,
    device,
    unfreeze_last_blocks,
    optimizer=None,
    augment=False,
    flip_probability=0.5,
    brightness_strength=0.05,
    epoch_number=None,
    log_every=25,
):
    training = optimizer is not None

    if training:
        set_finetune_train_mode(
            model, unfreeze_last_blocks
        )
    else:
        model.eval()

    total_loss = 0.0
    total_images = 0
    matrix = torch.zeros(
        (2, 2), dtype=torch.long
    )

    total_batches = len(loader)
    start_time = time.perf_counter()

    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(
            loader, 1
        ):
            images = images.to(device)
            labels = labels.to(device)

            if training and augment:
                images = augment_training_batch(
                    images,
                    flip_probability,
                    brightness_strength,
                )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)

            if logits.shape != (len(labels), 2):
                raise RuntimeError(
                    "Expected binary classification logits."
                )

            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Encountered non-finite loss."
                )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = len(labels)

            total_images += batch_size
            total_loss += (
                loss.detach().item() * batch_size
            )

            matrix += confusion_matrix(
                logits.detach(),
                labels.detach(),
            )

            if (
                batch_index % log_every == 0
                or batch_index == total_batches
            ):
                stage = (
                    "TRAIN" if training else "VALIDATION"
                )

                elapsed = (
                    time.perf_counter() - start_time
                )

                print(
                    f"Epoch {epoch_number} | {stage} "
                    f"batch {batch_index}/{total_batches} "
                    f"| loss {total_loss / total_images:.4f} "
                    f"| elapsed {elapsed:.1f}s",
                    flush=True,
                )

    if not total_images:
        raise RuntimeError(
            "DataLoader contains no images."
        )

    return {
        "loss": total_loss / total_images,
        **calculate_metrics(matrix),
    }


def save_finetune_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_macro_f1,
    config,
    baseline_sha256,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_f1": float(best_macro_f1),
        "classes": list(CLASS_NAMES),
        "architecture": "mobilenet_v3_small",
        "preprocessing": "IMAGENET1K_V1",
        "backbone_frozen": False,
        "experiment_type": "partial_finetuning",
        "unfrozen_last_feature_blocks":
            config.unfreeze_last_blocks,
        "baseline_sha256": baseline_sha256,
        "seed": config.seed,
    }

    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )
    os.close(fd)

    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                default=str,
                allow_nan=False,
            )

        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execute_finetuning(config, resume=False):
    """
    This function performs real optimizer updates.
    Call only after explicit authorization.
    """

    readiness = check_finetune_readiness(
        config, resume=resume
    )

    baseline_hash = readiness["baseline_sha256"]

    set_seed(config.seed)
    device = get_device(config.device)

    state = (
        inspect_resume_state(config)
        if resume else None
    )

    model = create_severity_model(
        pretrained=False,
        freeze_backbone=True,
        device="cpu",
    )

    if resume:
        source_state = state[
            "last_checkpoint"
        ]["model_state_dict"]
    else:
        baseline = torch.load(
            config.source_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        source_state = baseline[
            "model_state_dict"
        ]

    model.load_state_dict(
        source_state,
        strict=True,
    )

    backbone_params, head_params = (
        prepare_finetuning_model(
            model,
            config.unfreeze_last_blocks,
        )
    )

    model = model.to(device)

    optimizer = torch.optim.Adam([
        {
            "params": backbone_params,
            "lr": config.backbone_lr,
        },
        {
            "params": head_params,
            "lr": config.head_lr,
        },
    ])

    criterion = nn.CrossEntropyLoss()

    if resume:
        optimizer.load_state_dict(
            state["last_checkpoint"][
                "optimizer_state_dict"
            ]
        )

        stopper = state["early_stopping"]
        start_epoch = state["next_epoch"]

    else:
        stopper = EarlyStopping(
            config.patience,
            config.min_delta,
        )
        start_epoch = 1

    train_loader = create_severity_dataloader(
        manifest_path=config.manifest,
        dataset_root=config.data_root,
        split="train",
        batch_size=config.batch_size,
        allow_partial=False,
        shuffle=True,
    )

    val_loader = create_severity_dataloader(
        manifest_path=config.manifest,
        dataset_root=config.data_root,
        split="validation",
        batch_size=config.batch_size,
        allow_partial=False,
        shuffle=False,
    )

    if not resume:
        config.output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        write_json_atomic(
            config.output_dir / "config.json",
            {
                **asdict(config),
                "source_checkpoint_sha256":
                    baseline_hash,
                "started_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

    history_path = (
        config.output_dir / "history.jsonl"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=" * 65)
    print("MOBILENETV3 CONTROLLED FINE-TUNING")
    print("=" * 65)
    print("Resume:", resume)
    print("Device:", device)
    print("Starting epoch:", start_epoch)
    print("Maximum epochs:", config.epochs)
    print("Batch size:", config.batch_size)
    print("Baseline preserved:", baseline_hash)
    print("Test split: EXCLUDED")

    if resume:
        print(
            "Previous best epoch:",
            state["best_epoch"],
        )
        print(
            "Previous best macro-F1:",
            stopper.best,
        )
        print(
            "Early stopping counter:",
            stopper.counter,
        )

    try:
        for epoch in range(
            start_epoch,
            config.epochs + 1,
        ):
            train_metrics = run_finetune_epoch(
                model,
                train_loader,
                criterion,
                device,
                config.unfreeze_last_blocks,
                optimizer=optimizer,
                augment=config.use_augmentation,
                flip_probability=(
                    config.horizontal_flip_probability
                ),
                brightness_strength=(
                    config.brightness_strength
                ),
                epoch_number=epoch,
                log_every=config.log_every,
            )

            val_metrics = run_finetune_epoch(
                model,
                val_loader,
                criterion,
                device,
                config.unfreeze_last_blocks,
                optimizer=None,
                augment=False,
                epoch_number=epoch,
                log_every=config.log_every,
            )

            score = val_metrics["macro_f1"]

            improved, should_stop = stopper.step(
                score
            )

            record = {
                "epoch": epoch,
                "training": train_metrics,
                "validation": val_metrics,
            }

            # Preserve existing history by appending only
            # after an entire epoch has completed.
            with history_path.open(
                "a", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(record) + "\n"
                )

            save_finetune_checkpoint(
                config.output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                stopper.best,
                config,
                baseline_hash,
            )

            if improved:
                save_finetune_checkpoint(
                    config.output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    stopper.best,
                    config,
                    baseline_hash,
                )

            print(
                f"\nEpoch {epoch}/{config.epochs} COMPLETE "
                f"| Train loss: {train_metrics['loss']:.4f} "
                f"| Val loss: {val_metrics['loss']:.4f} "
                f"| Accuracy: "
                f"{val_metrics['accuracy'] * 100:.2f}% "
                f"| Macro-F1: {score:.4f}",
                flush=True,
            )

            if device.type == "cuda":
                print(
                    "Peak PyTorch allocated GPU memory: "
                    f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB",
                    flush=True,
                )

            if improved:
                print(
                    "Best checkpoint updated.",
                    flush=True,
                )

            if should_stop:
                print(
                    "Early stopping triggered.",
                    flush=True,
                )
                break

    finally:
        if sha256_file(
            config.source_checkpoint
        ) != baseline_hash:
            raise RuntimeError(
                "Baseline checkpoint changed unexpectedly!"
            )

        print(
            "Baseline SHA256 unchanged.",
            flush=True,
        )

    print("\nFine-tuning completed.")
    print("Best:", config.output_dir / "best.pt")
    print("Last:", config.output_dir / "last.pt")
    print("History:", history_path)
    print("No test images accessed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute-finetuning",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume saved model, optimizer, and history.",
    )

    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/severity/best.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "checkpoints/severity/finetune_run_01"
        ),
    )

    parser.add_argument(
        "--epochs", type=int, default=12
    )

    parser.add_argument(
        "--batch-size", type=int, default=4
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )

    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--no-augmentation",
        action="store_true",
    )

    args = parser.parse_args()

    config = FineTuneConfig(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        unfreeze_last_blocks=(
            args.unfreeze_last_blocks
        ),
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        use_augmentation=not args.no_augmentation,
        log_every=args.log_every,
    )

    if not args.execute_finetuning:
        print("=" * 65)
        print("FINE-TUNING READ-ONLY PREFLIGHT")
        print("=" * 65)

        report = check_finetune_readiness(
            config,
            resume=args.resume,
        )

        print(json.dumps(report, indent=2))
        print("No training performed.")
        return

    execute_finetuning(
        config,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
