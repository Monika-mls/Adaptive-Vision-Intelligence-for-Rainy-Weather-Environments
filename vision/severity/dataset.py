
"""
RealRain-1k binary rain severity dataset.

Class mapping:
    Light -> 0
    Heavy -> 1

Uses the original severity label manifest and loads
only rainy input images.

Supports:
- Train / validation / test splits
- Official MobileNetV3-Small preprocessing
- Explicit checking of local files
- PyTorch Dataset and DataLoader

No images are downloaded automatically.
"""

import csv
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader

from vision.preprocessing.image_loader import load_image
from vision.preprocessing.inference_preprocessor import (
    InferenceConfig,
    prepare_inference_image,
)


CLASS_TO_INDEX = {
    "Light": 0,
    "Heavy": 1,
}

INDEX_TO_CLASS = {
    0: "Light",
    1: "Heavy",
}

Split = Literal["train", "validation", "test"]


class SeverityDatasetError(ValueError):
    """Raised when severity dataset inputs are invalid."""


class RainSeverityDataset(Dataset):
    """Dataset for Light/Heavy rainy image classification."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: Split = "train",
        allow_partial: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split

        if split not in ("train", "validation", "test"):
            raise SeverityDatasetError(
                f"Unsupported split: {split}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest missing: {self.manifest_path}"
            )

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory missing: {self.dataset_root}"
            )

        with self.manifest_path.open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            required = {
                "severity",
                "split",
                "rainy_path",
                "scene_group_id",
            }

            if not required.issubset(
                set(reader.fieldnames or [])
            ):
                raise SeverityDatasetError(
                    "Manifest is missing required columns."
                )

            rows = list(reader)

        self.samples = []
        self.missing_files = []

        for row in rows:
            if row["split"] != split:
                continue

            severity = row["severity"]

            if severity not in CLASS_TO_INDEX:
                raise SeverityDatasetError(
                    f"Unknown severity label: {severity}"
                )

            relative_path = Path(row["rainy_path"])

            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise SeverityDatasetError(
                    f"Unsafe dataset path: {relative_path}"
                )

            image_path = self.dataset_root / relative_path

            if not image_path.is_file():
                self.missing_files.append(str(image_path))
                continue

            self.samples.append({
                "path": image_path,
                "label": CLASS_TO_INDEX[severity],
                "severity": severity,
                "scene_group_id": row["scene_group_id"],
            })

        if self.missing_files and not allow_partial:
            raise SeverityDatasetError(
                f"{len(self.missing_files)} images are missing "
                f"from the {split} split. "
                "Set allow_partial=True only for smoke tests."
            )

        if not self.samples:
            raise SeverityDatasetError(
                f"No available images found for split: {split}"
            )

        print(
            f"[{split}] Loaded {len(self.samples)} images; "
            f"missing: {len(self.missing_files)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = load_image(sample["path"])

        # The inference preprocessing profile is also
        # our deterministic MobileNet classification baseline.
        batch = prepare_inference_image(
            image,
            InferenceConfig(
                task="severity",
                device="cpu",
            ),
        )

        tensor = batch.squeeze(0)

        label = torch.tensor(
            sample["label"], dtype=torch.long
        )

        return tensor, label


def create_severity_dataloader(
    manifest_path: str | Path,
    dataset_root: str | Path,
    split: Split,
    batch_size: int = 4,
    allow_partial: bool = False,
    shuffle: bool | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader for a dataset split."""

    if batch_size <= 0:
        raise SeverityDatasetError(
            "batch_size must be positive."
        )

    dataset = RainSeverityDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split=split,
        allow_partial=allow_partial,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
