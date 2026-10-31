
"""
Tests for binary severity Dataset and DataLoader.

Uses temporary synthetic test images.
No dataset downloads or training.
"""

import csv

import pytest
import torch
from PIL import Image

from vision.severity.dataset import (
    RainSeverityDataset,
    SeverityDatasetError,
    create_severity_dataloader,
)


@pytest.fixture
def sample_dataset(tmp_path):
    root = tmp_path / "datasets"
    root.mkdir()

    rows = []

    examples = [
        ("Light", "train", "light_train.png"),
        ("Heavy", "train", "heavy_train.png"),
        ("Light", "validation", "light_val.png"),
        ("Heavy", "validation", "heavy_val.png"),
    ]

    for index, (severity, split, filename) in enumerate(
        examples
    ):
        folder = root / split
        folder.mkdir(exist_ok=True)

        image_path = folder / filename

        Image.new(
            "RGB",
            (300, 280),
            color=(50 + index * 20, 80, 120),
        ).save(image_path)

        rows.append({
            "severity": severity,
            "split": split,
            "rainy_path": f"{split}/{filename}",
            "scene_group_id": str(index),
        })

    manifest = tmp_path / "manifest.csv"

    with manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "severity",
                "split",
                "rainy_path",
                "scene_group_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest, root


def test_training_dataset_length(sample_dataset):
    manifest, root = sample_dataset

    dataset = RainSeverityDataset(
        manifest, root, split="train"
    )

    assert len(dataset) == 2


def test_validation_dataset_length(sample_dataset):
    manifest, root = sample_dataset

    dataset = RainSeverityDataset(
        manifest, root, split="validation"
    )

    assert len(dataset) == 2


def test_image_tensor_shape(sample_dataset):
    manifest, root = sample_dataset

    dataset = RainSeverityDataset(
        manifest, root, split="train"
    )

    image, _ = dataset[0]

    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32


def test_binary_labels(sample_dataset):
    manifest, root = sample_dataset

    dataset = RainSeverityDataset(
        manifest, root, split="train"
    )

    labels = [dataset[i][1].item() for i in range(2)]

    assert labels == [0, 1]


def test_dataloader_batch_shape(sample_dataset):
    manifest, root = sample_dataset

    loader = create_severity_dataloader(
        manifest,
        root,
        split="train",
        batch_size=2,
        shuffle=False,
    )

    images, labels = next(iter(loader))

    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)


def test_reject_invalid_split(sample_dataset):
    manifest, root = sample_dataset

    with pytest.raises(SeverityDatasetError):
        RainSeverityDataset(
            manifest, root, split="unknown"
        )


def test_reject_missing_manifest(sample_dataset):
    _, root = sample_dataset

    with pytest.raises(FileNotFoundError):
        RainSeverityDataset(
            "missing.csv", root, split="train"
        )


def test_reject_missing_image(sample_dataset):
    manifest, root = sample_dataset

    (root / "train/light_train.png").unlink()

    with pytest.raises(SeverityDatasetError):
        RainSeverityDataset(
            manifest, root, split="train"
        )


def test_partial_dataset_allowed(sample_dataset):
    manifest, root = sample_dataset

    (root / "train/light_train.png").unlink()

    dataset = RainSeverityDataset(
        manifest,
        root,
        split="train",
        allow_partial=True,
    )

    assert len(dataset) == 1
    assert len(dataset.missing_files) == 1


def test_reject_zero_batch_size(sample_dataset):
    manifest, root = sample_dataset

    with pytest.raises(SeverityDatasetError):
        create_severity_dataloader(
            manifest,
            root,
            split="train",
            batch_size=0,
        )
