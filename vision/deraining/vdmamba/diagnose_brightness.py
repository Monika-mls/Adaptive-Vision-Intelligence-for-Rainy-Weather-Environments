
"""
Diagnose brightness behavior in paired Heavy-rain data.

Uses the existing HeavyRainDataset.

Compares:
1. Rainy image brightness
2. Clean target brightness
3. Target minus rainy brightness
4. Fraction of pixels where the clean target
   is darker than the rainy input

Only analyzes the training split.
Does not modify checkpoints or images.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import (
    HeavyRainDataset,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    dataset = HeavyRainDataset(
        split="train",
        patch_size=args.patch_size,
        augment=False,
        limit=args.count,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rainy_means = []
    clean_means = []
    signed_changes = []
    darker_fractions = []
    brighter_fractions = []

    for batch in loader:
        rainy = batch["rainy"].float()
        clean = batch["clean"].float()

        difference = clean - rainy

        rainy_means.append(
            float(rainy.mean().item())
        )

        clean_means.append(
            float(clean.mean().item())
        )

        signed_changes.append(
            float(difference.mean().item())
        )

        darker_fractions.append(
            float((difference < 0).float().mean().item())
        )

        brighter_fractions.append(
            float((difference > 0).float().mean().item())
        )

    def average(values):
        return sum(values) / len(values)

    print("=" * 60)
    print("HEAVY-RAIN DATASET BRIGHTNESS DIAGNOSTICS")
    print("=" * 60)

    print("Pairs analyzed:", len(dataset))

    print(
        "Mean rainy brightness:",
        round(average(rainy_means) * 255, 4),
    )

    print(
        "Mean clean brightness:",
        round(average(clean_means) * 255, 4),
    )

    print(
        "Mean target minus rainy:",
        round(average(signed_changes) * 255, 4),
    )

    print(
        "Fraction clean darker:",
        round(average(darker_fractions), 4),
    )

    print(
        "Fraction clean brighter:",
        round(average(brighter_fractions), 4),
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
