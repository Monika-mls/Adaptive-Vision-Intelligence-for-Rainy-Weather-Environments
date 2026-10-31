
"""
Phase 3 Step 18: Fine-tuning resume tests.

All checkpoints and histories are temporary.
No training, GPU or downloads required.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from vision.severity.fine_tune import (
    BASELINE_SHA256,
    FineTuneConfig,
    augment_training_batch,
    inspect_resume_state,
    read_history,
    reconstruct_early_stopping,
    validate_config,
)


def make_record(epoch, score):
    return {
        "epoch": epoch,
        "training": {"loss": 0.6},
        "validation": {
            "loss": 0.6,
            "macro_f1": score,
            "accuracy": 0.7,
        },
    }


def prepare_fake_run(tmp_path):
    """Create a realistic three-epoch resume fixture."""

    root = tmp_path / "run"
    root.mkdir()

    config = FineTuneConfig(
        source_checkpoint=tmp_path / "baseline.pt",
        output_dir=root,
    )

    (tmp_path / "baseline.pt").write_bytes(
        b"test baseline"
    )

    settings = {
        **asdict(config),
        "source_checkpoint_sha256": BASELINE_SHA256,
    }

    (root / "config.json").write_text(
        json.dumps(settings, default=str),
        encoding="utf-8",
    )

    records = [
        make_record(1, 0.6874333036977085),
        make_record(2, 0.6733),
        make_record(3, 0.6783),
    ]

    (root / "history.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )

    model = nn.Linear(3, 2)

    optimizer = torch.optim.Adam([
        {
            "params": [model.weight],
            "lr": 1e-5,
        },
        {
            "params": [model.bias],
            "lr": 1e-4,
        },
    ])

    payload = {
        "epoch": 3,
        "architecture": "mobilenet_v3_small",
        "preprocessing": "IMAGENET1K_V1",
        "classes": ["Light", "Heavy"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_f1": 0.6874333036977085,
        "experiment_type": "partial_finetuning",
        "unfrozen_last_feature_blocks": 2,
        "baseline_sha256": BASELINE_SHA256,
    }

    torch.save(payload, root / "last.pt")

    best_payload = dict(payload)
    best_payload["epoch"] = 1

    torch.save(best_payload, root / "best.pt")

    return config, root


def test_three_epochs_reconstruct_correctly():
    records = [
        make_record(1, 0.6874333036977085),
        make_record(2, 0.6733),
        make_record(3, 0.6783),
    ]

    stopper, best_epoch, stopped = (
        reconstruct_early_stopping(
            records,
            patience=4,
            min_delta=1e-4,
        )
    )

    assert best_epoch == 1
    assert stopper.counter == 2
    assert stopper.best == pytest.approx(
        0.6874333036977085
    )
    assert not stopped


def test_resume_starts_at_epoch_four(tmp_path):
    config, _ = prepare_fake_run(tmp_path)

    state = inspect_resume_state(config)

    assert state["last_epoch"] == 3
    assert state["next_epoch"] == 4
    assert state["best_epoch"] == 1
    assert state["early_stopping"].counter == 2


def test_resume_rejects_epoch_mismatch(tmp_path):
    config, root = prepare_fake_run(tmp_path)

    path = root / "last.pt"

    checkpoint = torch.load(
        path,
        weights_only=True,
        map_location="cpu",
    )

    checkpoint["epoch"] = 2
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="does not match"):
        inspect_resume_state(config)


def test_resume_rejects_config_changes(tmp_path):
    config, _ = prepare_fake_run(tmp_path)

    config.batch_size = 2

    with pytest.raises(ValueError, match="batch_size"):
        inspect_resume_state(config)


def test_resume_rejects_existing_completed_run(tmp_path):
    config, root = prepare_fake_run(tmp_path)

    config.epochs = 3

    # Original experiment's epoch limit was 12,
    # so reducing it is rejected.
    with pytest.raises(ValueError):
        inspect_resume_state(config)


def test_existing_directory_requires_resume(tmp_path):
    config, _ = prepare_fake_run(tmp_path)

    with pytest.raises(FileExistsError):
        validate_config(config, resume=False)

    assert validate_config(config, resume=True)


def test_history_rejects_duplicate_epoch(tmp_path):
    history = tmp_path / "history.jsonl"

    history.write_text(
        json.dumps(make_record(1, 0.7)) + "\n"
        + json.dumps(make_record(1, 0.6)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_history(history)


def test_horizontal_flip_without_sync():
    images = torch.arange(
        2 * 3 * 4 * 5,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5)

    result = augment_training_batch(
        images,
        flip_probability=1.0,
        brightness_strength=0.0,
    )

    assert torch.equal(
        result,
        torch.flip(images, dims=[-1]),
    )
