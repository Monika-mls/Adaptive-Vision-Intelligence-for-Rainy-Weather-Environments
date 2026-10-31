
"""
Training infrastructure tests.

No real model training.
No ImageNet weight downloads.
No GPU allocation required.
"""

import csv

import pytest
import torch
from torch import nn

from vision.severity.train import (
    TrainConfig,
    EarlyStopping,
    confusion_matrix,
    calculate_metrics,
    save_checkpoint,
    check_data_readiness,
    get_device,
)


def test_confusion_matrix():
    logits = torch.tensor([
        [4.0, 0.0],
        [0.0, 4.0],
        [0.0, 4.0],
        [4.0, 0.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    result = confusion_matrix(logits, labels)

    assert result.tolist() == [
        [1, 1],
        [1, 1],
    ]


def test_perfect_classification_metrics():
    matrix = torch.tensor([
        [3, 0],
        [0, 2],
    ])

    metrics = calculate_metrics(matrix)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["Light"]["recall"] == 1.0
    assert metrics["per_class"]["Heavy"]["precision"] == 1.0


def test_incorrect_classification_metrics():
    matrix = torch.tensor([
        [1, 1],
        [1, 1],
    ])

    metrics = calculate_metrics(matrix)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5


def test_missing_class_metrics():
    matrix = torch.tensor([
        [4, 0],
        [0, 0],
    ])

    metrics = calculate_metrics(matrix)

    assert metrics["per_class"]["Heavy"]["f1"] == 0.0
    assert metrics["macro_f1"] == 0.5


def test_early_stopping():
    stopper = EarlyStopping(
        patience=2, min_delta=0.001
    )

    assert stopper.step(0.5) == (True, False)
    assert stopper.step(0.5005) == (False, False)
    assert stopper.step(0.49) == (False, True)


def test_early_stopping_resets():
    stopper = EarlyStopping(patience=2)

    stopper.step(0.5)
    stopper.step(0.4)

    improved, should_stop = stopper.step(0.7)

    assert improved
    assert not should_stop
    assert stopper.counter == 0


def test_checkpoint_saving(tmp_path):
    model = nn.Linear(3, 2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    path = tmp_path / "best.pt"

    save_checkpoint(
        path, model, optimizer,
        epoch=3, score=0.85,
    )

    assert path.exists()

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_macro_f1"] == 0.85
    assert checkpoint["classes"] == ["Light", "Heavy"]

    restored = nn.Linear(3, 2)
    restored.load_state_dict(
        checkpoint["model_state_dict"]
    )

    for a, b in zip(
        model.parameters(),
        restored.parameters(),
    ):
        assert torch.equal(a, b)


def test_cpu_device():
    device = get_device("cpu")

    assert device.type == "cpu"


def test_invalid_device():
    with pytest.raises(ValueError):
        get_device("invalid")


def test_readiness_rejects_incomplete_data(tmp_path):
    data_root = tmp_path / "datasets"
    data_root.mkdir()

    manifest = tmp_path / "severity.csv"

    fields = [
        "split",
        "severity",
        "rainy_path",
        "scene_group_id",
    ]

    with manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerow({
            "split": "train",
            "severity": "Light",
            "rainy_path": "train/example.png",
            "scene_group_id": "100",
        })

    config = TrainConfig(
        manifest=manifest,
        data_root=data_root,
    )

    result = check_data_readiness(config)

    assert result["ready"] is False
    assert result["missing_count"] == 1
