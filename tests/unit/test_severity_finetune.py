
"""
Unit tests for the controlled fine-tuning implementation.

No ImageNet downloads.
No real dataset access.
No actual fine-tuning experiment.
No GPU required.
"""

import hashlib

import pytest
import torch
from torch import nn

from vision.severity.fine_tune import (
    FineTuneConfig,
    augment_training_batch,
    prepare_finetuning_model,
    set_finetune_train_mode,
    sha256_file,
    save_finetune_checkpoint,
    validate_config,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(
                nn.Linear(4, 4),
                nn.BatchNorm1d(4),
            ),
            nn.Sequential(
                nn.Linear(4, 4),
                nn.BatchNorm1d(4),
            ),
            nn.Sequential(
                nn.Linear(4, 4),
                nn.BatchNorm1d(4),
            ),
            nn.Sequential(
                nn.Linear(4, 4),
                nn.BatchNorm1d(4),
            ),
        )

        self.classifier = nn.Linear(4, 2)


class TinySeverityWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.backbone_frozen = True


def test_default_config():
    config = FineTuneConfig()

    assert config.batch_size == 4
    assert config.epochs == 12
    assert config.unfreeze_last_blocks == 2
    assert config.backbone_lr == 1e-5
    assert config.head_lr == 1e-4


def test_invalid_learning_rate(tmp_path):
    config = FineTuneConfig(
        output_dir=tmp_path / "new_experiment",
        head_lr=0,
    )

    with pytest.raises(ValueError):
        validate_config(config)


def test_existing_experiment_rejected(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    config = FineTuneConfig(output_dir=output)

    with pytest.raises(FileExistsError):
        validate_config(config)


def test_partial_unfreezing():
    model = TinySeverityWrapper()

    backbone_params, head_params = (
        prepare_finetuning_model(
            model,
            unfreeze_last_blocks=2,
        )
    )

    early_blocks = list(
        model.model.features.children()
    )[:2]

    later_blocks = list(
        model.model.features.children()
    )[2:]

    assert all(
        not p.requires_grad
        for block in early_blocks
        for p in block.parameters()
    )

    assert all(
        p.requires_grad
        for block in later_blocks
        for p in block.parameters()
    )

    assert all(
        p.requires_grad
        for p in model.model.classifier.parameters()
    )

    assert len(backbone_params) > 0
    assert len(head_params) > 0
    assert model.backbone_frozen is False


def test_frozen_batchnorm_uses_eval_mode():
    model = TinySeverityWrapper()

    prepare_finetuning_model(
        model,
        unfreeze_last_blocks=2,
    )

    set_finetune_train_mode(
        model,
        unfreeze_last_blocks=2,
    )

    blocks = list(model.model.features.children())

    assert not blocks[0].training
    assert not blocks[1].training

    assert blocks[2].training
    assert blocks[3].training

    assert model.model.classifier.training


def test_augmentation_disabled_is_identity():
    images = torch.randn(2, 3, 224, 224)

    result = augment_training_batch(
        images,
        flip_probability=0.0,
        brightness_strength=0.0,
    )

    assert torch.equal(result, images)

    # Original tensors are not modified.
    assert result.data_ptr() != images.data_ptr()


def test_horizontal_flip():
    images = torch.arange(
        2 * 3 * 4 * 5,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5)

    result = augment_training_batch(
        images,
        flip_probability=1.0,
        brightness_strength=0.0,
    )

    expected = torch.flip(images, dims=[-1])

    assert torch.equal(result, expected)


def test_augmentation_finite():
    torch.manual_seed(42)

    images = torch.randn(4, 3, 224, 224)

    result = augment_training_batch(
        images,
        flip_probability=0.5,
        brightness_strength=0.05,
    )

    assert result.shape == images.shape
    assert torch.isfinite(result).all().item()


def test_invalid_augmentation_strength():
    images = torch.randn(1, 3, 32, 32)

    with pytest.raises(ValueError):
        augment_training_batch(
            images,
            brightness_strength=0.5,
        )


def test_checkpoint_isolation(tmp_path):
    model = TinySeverityWrapper()

    backbone_params, head_params = (
        prepare_finetuning_model(model, 2)
    )

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-4},
    ])

    baseline = tmp_path / "baseline.pt"

    # Test-only baseline content.
    baseline.write_bytes(b"unchanged-baseline")

    initial_digest = sha256_file(baseline)

    config = FineTuneConfig(
        source_checkpoint=baseline,
        output_dir=tmp_path / "fine_tuned",
    )

    output = config.output_dir / "best.pt"

    save_finetune_checkpoint(
        path=output,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_macro_f1=0.78,
        config=config,
        baseline_sha256=initial_digest,
    )

    checkpoint = torch.load(
        output,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_macro_f1"] == pytest.approx(
        0.78
    )
    assert checkpoint["classes"] == ["Light", "Heavy"]
    assert checkpoint["backbone_frozen"] is False

    assert checkpoint[
        "unfrozen_last_feature_blocks"
    ] == 2

    assert checkpoint[
        "baseline_sha256"
    ] == initial_digest

    assert sha256_file(baseline) == initial_digest


def test_checkpoint_hash(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello")

    expected = hashlib.sha256(b"hello").hexdigest()

    assert sha256_file(path) == expected
