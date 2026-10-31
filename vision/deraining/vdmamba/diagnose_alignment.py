
"""
Diagnostic for paired rainy/clean image alignment.

Uses HeavyRainDataset with deterministic validation crops.

Compares gradient-map agreement at zero shift versus
small translated shifts.

Does not modify images or checkpoints.
Does not use the held-out test split.

IMPORTANT:
This is a heuristic alignment diagnostic.
A shift preference is not definitive proof of misregistration.
"""

import argparse

import torch
import torch.nn.functional as F

from vision.deraining.vdmamba.heavy_dataset import (
    HeavyRainDataset,
)


def gradient_magnitude(image):
    gray = (
        0.299 * image[0]
        + 0.587 * image[1]
        + 0.114 * image[2]
    )

    dx = F.pad(
        (gray[:, 1:] - gray[:, :-1]).abs(),
        (0, 1, 0, 0),
    )

    dy = F.pad(
        (gray[1:, :] - gray[:-1, :]).abs(),
        (0, 0, 0, 1),
    )

    return dx + dy


def shifted_overlap(a, b, dy, dx):
    height, width = a.shape

    top_a = max(0, dy)
    top_b = max(0, -dy)

    left_a = max(0, dx)
    left_b = max(0, -dx)

    overlap_h = height - abs(dy)
    overlap_w = width - abs(dx)

    if overlap_h <= 0 or overlap_w <= 0:
        raise ValueError("Shift exceeds image dimensions")

    a_crop = a[
        top_a:top_a + overlap_h,
        left_a:left_a + overlap_w,
    ]

    b_crop = b[
        top_b:top_b + overlap_h,
        left_b:left_b + overlap_w,
    ]

    return a_crop, b_crop


def normalized_correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()

    denominator = torch.sqrt(
        a.square().sum() * b.square().sum()
    )

    if denominator.item() < 1e-10:
        return None

    return float(
        (a * b).sum().item() / denominator.item()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count", type=int, default=112
    )

    parser.add_argument(
        "--patch-size", type=int, default=64
    )

    parser.add_argument(
        "--max-shift", type=int, default=3
    )

    args = parser.parse_args()

    if args.max_shift < 0 or args.max_shift >= args.patch_size:
        parser.error("Invalid --max-shift")

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=args.patch_size,
        augment=False,
        limit=args.count,
    )

    shift_counts = {}
    total_improvement = 0.0
    valid_samples = 0
    nonzero_preferred = 0

    for index in range(len(dataset)):
        sample = dataset[index]

        rainy = gradient_magnitude(
            sample["rainy"]
        )

        clean = gradient_magnitude(
            sample["clean"]
        )

        # Use the same central evaluation region
        # for every shift to avoid overlap-size bias.
        margin = args.max_shift

        if 2 * margin >= args.patch_size:
            raise ValueError("Shift margin too large")

        reference = rainy[
            margin:-margin if margin else None,
            margin:-margin if margin else None,
        ]

        best_score = float("-inf")
        best_shift = (0, 0)
        zero_score = None

        for dy in range(
            -args.max_shift,
            args.max_shift + 1,
        ):
            for dx in range(
                -args.max_shift,
                args.max_shift + 1,
            ):
                shifted = clean[
                    margin - dy:
                    args.patch_size - margin - dy,
                    margin - dx:
                    args.patch_size - margin - dx,
                ]

                score = normalized_correlation(
                    reference, shifted
                )

                if score is None:
                    continue

                if dy == 0 and dx == 0:
                    zero_score = score

                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)

        if zero_score is None:
            continue

        valid_samples += 1

        shift_counts[best_shift] = (
            shift_counts.get(best_shift, 0) + 1
        )

        total_improvement += best_score - zero_score

        if best_shift != (0, 0):
            nonzero_preferred += 1

    print("=" * 65)
    print("PAIR ALIGNMENT DIAGNOSTIC")
    print("=" * 65)

    print("Samples evaluated:", valid_samples)

    print(
        "Nonzero shift preferred:",
        nonzero_preferred,
    )

    if valid_samples:
        print(
            "Fraction preferring nonzero shift:",
            round(nonzero_preferred / valid_samples, 4),
        )

        print(
            "Mean correlation improvement:",
            round(total_improvement / valid_samples, 6),
        )

    print("\nMost common preferred shifts:")

    for shift, count in sorted(
        shift_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        print(shift, ":", count)

    print("\nThis is a heuristic, not a definitive alignment test.")
    print("Validation split only. Test split unused.")


if __name__ == "__main__":
    main()
