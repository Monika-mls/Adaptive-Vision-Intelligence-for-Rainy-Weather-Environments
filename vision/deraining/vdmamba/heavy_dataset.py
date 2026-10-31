
"""
Paired RealRain-1k Heavy-rain dataset.

Uses the existing severity_labels.csv manifest.

Training:
- Random aligned crop
- Paired flips and rotations

Validation:
- Deterministic center crop
- No random augmentation

Test:
- Explicitly excluded

No images or checkpoint files are modified.
"""

import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class HeavyRainDataset(Dataset):
    def __init__(
        self,
        root=".",
        split="train",
        patch_size=64,
        augment=True,
        limit=None,
    ):
        self.root = Path(root).resolve()
        self.split = split.lower()
        self.patch_size = patch_size
        self.augment = augment and self.split == "train"

        if self.split not in ("train", "validation"):
            raise ValueError(
                "Only train and validation splits are allowed."
            )

        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer.")

        if limit is not None and (
            not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be positive.")

        manifest = self.root / "docs/severity_labels.csv"

        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        self.samples = []

        with manifest.open(
            newline="",
            encoding="utf-8-sig",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["severity"].strip().lower() != "heavy":
                    continue

                if row["split"].strip().lower() != self.split:
                    continue

                rainy = self.resolve_path(row["rainy_path"])
                clean = self.resolve_path(row["clean_path"])

                if rainy is None or clean is None:
                    raise FileNotFoundError(
                        f"Missing paired image for "
                        f"{row['image_id']}: "
                        f"{row['rainy_path']} / {row['clean_path']}"
                    )

                self.samples.append({
                    "image_id": row["image_id"],
                    "rainy": rainy,
                    "clean": clean,
                })

        self.samples.sort(key=lambda row: row["image_id"])

        if limit is not None:
            self.samples = self.samples[:limit]

        if not self.samples:
            raise RuntimeError(
                f"No Heavy-rain pairs found for {self.split}."
            )

    def resolve_path(self, value):
        relative = Path(value.strip().replace("\\", "/"))

        if not value.strip() or ".." in relative.parts:
            raise ValueError(f"Unsafe dataset path: {value}")

        candidates = (
            [relative]
            if relative.is_absolute()
            else [
                self.root / relative,
                self.root / "datasets" / relative,
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["rainy"]) as image:
            rainy = image.convert("RGB")

        with Image.open(sample["clean"]) as image:
            clean = image.convert("RGB")

        if rainy.size != clean.size:
            raise ValueError(
                f"Pair size mismatch for {sample['image_id']}: "
                f"{rainy.size} vs {clean.size}"
            )

        width, height = rainy.size
        size = self.patch_size

        if width < size or height < size:
            raise ValueError(
                f"Image {sample['image_id']} is smaller "
                f"than the requested {size}x{size} patch."
            )

        if self.split == "train":
            left = random.randint(0, width - size)
            top = random.randint(0, height - size)
        else:
            left = (width - size) // 2
            top = (height - size) // 2

        box = (left, top, left + size, top + size)

        rainy = rainy.crop(box)
        clean = clean.crop(box)

        if self.augment:
            if random.random() < 0.5:
                rainy = ImageOps.mirror(rainy)
                clean = ImageOps.mirror(clean)

            if random.random() < 0.5:
                rainy = ImageOps.flip(rainy)
                clean = ImageOps.flip(clean)

            rotation = random.choice((0, 90, 180, 270))

            if rotation:
                rainy = rainy.rotate(rotation)
                clean = clean.rotate(rotation)

        return {
            "rainy": to_tensor(rainy),
            "clean": to_tensor(clean),
            "image_id": sample["image_id"],
        }
