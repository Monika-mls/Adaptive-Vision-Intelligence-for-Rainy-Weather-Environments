
"""
Adaptive Vision Intelligence for Rainy Weather Environments

Phase 3 - Step 14
Full Light/Heavy Validation and Routing Evaluation

Uses:
    - Original RealRain-1k validation split
    - Trained MobileNetV3-Small best.pt
    - The existing predict.py inference pipeline

Reports:
    - Accuracy and macro-F1
    - Class-wise precision, recall, F1
    - Confusion matrix
    - Uncertainty coverage
    - Accepted-only accuracy
    - Heavy-to-Light routing errors
    - Confidence statistics
    - Per-image predictions

SAFETY:
    No training.
    No optimizer.
    No weight downloads.
    No test split access.
    No image restoration.
"""

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch

from vision.severity.predict import (
    CLASS_NAMES,
    load_trained_model,
    predict_image,
)


DEFAULT_MANIFEST = Path("docs/severity_labels.csv")
DEFAULT_DATA_ROOT = Path("datasets")
DEFAULT_CHECKPOINT = Path("checkpoints/severity/best.pt")

DEFAULT_JSON = Path("docs/severity_validation_report.json")
DEFAULT_CSV = Path("docs/severity_validation_predictions.csv")

EXPECTED_PER_CLASS = 112

ROUTES = {
    "Light": "PReNet",
    "Heavy": "VDMamba",
}


# ---------------------------------------------------------
# 1. LOAD VALIDATION MANIFEST
# ---------------------------------------------------------

def load_validation_samples(manifest_path, data_root):
    """
    Read only validation image records.

    Enforces:
        112 Light images
        112 Heavy images
        No missing files
        No duplicated image paths
        No unexpected folder or class
    """

    manifest_path = Path(manifest_path)
    data_root = Path(data_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest missing: {manifest_path}"
        )

    samples = []
    seen = set()
    counts = Counter()

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "split",
            "severity",
            "rainy_path",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Manifest is missing required columns."
            )

        for row in reader:
            # Critically, no test image is read or opened.
            if row["split"] != "validation":
                continue

            label = row["severity"]

            if label not in CLASS_NAMES:
                raise ValueError(
                    f"Unexpected validation class: {label}"
                )

            remote = PurePosixPath(
                row["rainy_path"].replace("\\", "/")
            )

            expected_track = (
                "RealRain-1k-L"
                if label == "Light"
                else "RealRain-1k-H"
            )

            if (
                len(remote.parts) != 4
                or remote.parts[:3] != (
                    expected_track,
                    "validation",
                    "input",
                )
                or remote.suffix.lower() != ".png"
                or remote.is_absolute()
                or any(p in (".", "..") for p in remote.parts)
                or ":" in str(remote)
            ):
                raise ValueError(
                    f"Invalid validation path: {remote}"
                )

            relative = remote.as_posix()

            if relative in seen:
                raise ValueError(
                    f"Duplicate image: {relative}"
                )

            seen.add(relative)

            image_path = data_root.joinpath(*remote.parts)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing validation image: {image_path}"
                )

            samples.append({
                "image_path": image_path,
                "relative_path": relative,
                "actual_class": label,
            })

            counts[label] += 1

    if counts != Counter({
        "Light": EXPECTED_PER_CLASS,
        "Heavy": EXPECTED_PER_CLASS,
    }):
        raise ValueError(
            "Unexpected validation class distribution: "
            f"{dict(counts)}"
        )

    return sorted(
        samples,
        key=lambda s: s["relative_path"],
    )


# ---------------------------------------------------------
# 2. METRIC HELPERS
# ---------------------------------------------------------

def safe_divide(numerator, denominator):
    return (
        float(numerator / denominator)
        if denominator
        else 0.0
    )


def mean_or_none(values):
    return (
        sum(values) / len(values)
        if values
        else None
    )


def summarize_predictions(rows):
    """
    Compute full-set classification and routing metrics.

    rows contain:
        actual_class
        predicted_class
        confidence
        uncertain
        automatic_routing_allowed

    This function can also be independently unit-tested.
    """

    if not rows:
        raise ValueError(
            "Cannot evaluate an empty prediction list."
        )

    confusion = [
        [0, 0],
        [0, 0],
    ]

    accepted_confusion = [
        [0, 0],
        [0, 0],
    ]

    accepted_count = 0
    accepted_correct = 0
    blocked_incorrect = 0
    blocked_correct = 0

    heavy_to_light_total = 0
    heavy_to_light_accepted = 0

    correct_confidences = []
    incorrect_confidences = []

    class_to_index = {
        "Light": 0,
        "Heavy": 1,
    }

    for row in rows:
        actual = row["actual_class"]
        predicted = row["predicted_class"]

        if actual not in class_to_index:
            raise ValueError(
                f"Unexpected actual class: {actual}"
            )

        if predicted not in class_to_index:
            raise ValueError(
                f"Unexpected predicted class: {predicted}"
            )

        confidence = float(row["confidence"])

        if not math.isfinite(confidence) or not (
            0.5 <= confidence <= 1.0
        ):
            raise ValueError(
                "Invalid binary prediction confidence."
            )

        accepted = row["automatic_routing_allowed"]
        uncertain = row["uncertain"]

        if (
            not isinstance(accepted, bool)
            or not isinstance(uncertain, bool)
            or accepted == uncertain
        ):
            raise ValueError(
                "Inconsistent routing/uncertainty flags."
            )

        actual_index = class_to_index[actual]
        predicted_index = class_to_index[predicted]

        confusion[actual_index][predicted_index] += 1

        correct = actual == predicted

        if correct:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

        if actual == "Heavy" and predicted == "Light":
            heavy_to_light_total += 1

            if accepted:
                heavy_to_light_accepted += 1

        if accepted:
            accepted_count += 1

            accepted_confusion[
                actual_index
            ][predicted_index] += 1

            if correct:
                accepted_correct += 1
        else:
            if correct:
                blocked_correct += 1
            else:
                blocked_incorrect += 1

    total = len(rows)
    correct_total = sum(
        confusion[index][index]
        for index in range(2)
    )

    per_class = {}

    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index][index]

        actual_total = sum(confusion[index])

        predicted_total = sum(
            confusion[row_index][index]
            for row_index in range(2)
        )

        precision = safe_divide(
            true_positive,
            predicted_total,
        )

        recall = safe_divide(
            true_positive,
            actual_total,
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
        }

    macro_f1 = sum(
        item["f1"]
        for item in per_class.values()
    ) / len(CLASS_NAMES)

    heavy_support = sum(confusion[1])

    return {
        "total_images": total,
        "correct_predictions": correct_total,
        "incorrect_predictions": total - correct_total,
        "accuracy": safe_divide(correct_total, total),
        "macro_f1": macro_f1,
        "class_order": list(CLASS_NAMES),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "routing": {
            "accepted": accepted_count,
            "blocked": total - accepted_count,
            "coverage": safe_divide(
                accepted_count, total
            ),
            "accepted_correct": accepted_correct,
            "accepted_incorrect": (
                accepted_count - accepted_correct
            ),
            "accepted_accuracy": (
                safe_divide(
                    accepted_correct,
                    accepted_count,
                )
                if accepted_count
                else None
            ),
            "blocked_correct": blocked_correct,
            "blocked_incorrect": blocked_incorrect,
            "heavy_to_light_total": heavy_to_light_total,
            "heavy_to_light_accepted": (
                heavy_to_light_accepted
            ),
            "heavy_to_light_blocked": (
                heavy_to_light_total
                - heavy_to_light_accepted
            ),
            "heavy_to_light_accepted_rate": (
                safe_divide(
                    heavy_to_light_accepted,
                    heavy_support,
                )
            ),
            "accepted_confusion_matrix": (
                accepted_confusion
            ),
        },
        "confidence": {
            "mean_correct": mean_or_none(
                correct_confidences
            ),
            "mean_incorrect": mean_or_none(
                incorrect_confidences
            ),
        },
    }


# ---------------------------------------------------------
# 3. WRITE REPORTS ATOMICALLY
# ---------------------------------------------------------

def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8", newline=""
        ) as file:
            file.write(content)

        os.replace(temp_path, path)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_reports(
    report,
    rows,
    json_path=DEFAULT_JSON,
    csv_path=DEFAULT_CSV,
):
    json_content = json.dumps(
        report,
        indent=2,
        allow_nan=False,
    )

    atomic_write_text(
        json_path,
        json_content + "\n",
    )

    from io import StringIO

    buffer = StringIO(newline="")

    fields = [
        "relative_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "light_probability",
        "heavy_probability",
        "uncertain",
        "automatic_routing_allowed",
        "suggested_restoration_model",
        "correct",
    ]

    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            key: row[key]
            for key in fields
        })

    atomic_write_text(
        csv_path,
        buffer.getvalue(),
    )


# ---------------------------------------------------------
# 4. RUN FULL VALIDATION
# ---------------------------------------------------------

def evaluate_validation(
    manifest_path=DEFAULT_MANIFEST,
    data_root=DEFAULT_DATA_ROOT,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device="auto",
    confidence_threshold=0.75,
    margin_threshold=0.20,
    json_path=DEFAULT_JSON,
    csv_path=DEFAULT_CSV,
):
    """
    Evaluate the entire validation split.

    Loads the trained checkpoint once.
    Uses the identical inference path as predict.py.
    """

    samples = load_validation_samples(
        manifest_path,
        data_root,
    )

    model, selected_device, metadata = (
        load_trained_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    )

    rows = []

    print("=" * 65)
    print("FULL RAIN SEVERITY VALIDATION")
    print("=" * 65)
    print("Checkpoint epoch:", metadata["checkpoint_epoch"])
    print("Device:", selected_device)
    print("Validation images:", len(samples))
    print("Test split: NOT ACCESSED")
    print()

    for index, sample in enumerate(samples, start=1):
        result = predict_image(
            image_path=sample["image_path"],
            model=model,
            device=selected_device,
            metadata=metadata,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )

        prediction = result["prediction"]

        predicted = prediction["predicted_class"]
        actual = sample["actual_class"]

        row = {
            "relative_path": sample["relative_path"],
            "actual_class": actual,
            "predicted_class": predicted,
            "confidence": prediction["confidence"],
            "light_probability": prediction[
                "probabilities"
            ]["Light"],
            "heavy_probability": prediction[
                "probabilities"
            ]["Heavy"],
            "uncertain": prediction["uncertain"],
            "automatic_routing_allowed": prediction[
                "automatic_routing_allowed"
            ],
            "suggested_restoration_model": prediction[
                "suggested_restoration_model"
            ],
            "correct": actual == predicted,
        }

        rows.append(row)

        if index % 25 == 0 or index == len(samples):
            print(
                f"Processed {index}/{len(samples)} images",
                flush=True,
            )

    metrics = summarize_predictions(rows)

    report = {
        "project": "Adaptive Vision Intelligence",
        "phase": "3 - Step 14",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_split": "validation",
        "test_split_accessed": False,
        "model": metadata,
        "thresholds": {
            "confidence": confidence_threshold,
            "margin": margin_threshold,
            "calibrated": False,
        },
        "metrics": metrics,
        "restoration_models_executed": False,
        "limitations": [
            "Validation set only, not held-out test.",
            "Confidence thresholds have not been calibrated.",
            "Routing decisions do not execute restoration.",
            "Validation performance does not establish "
            "out-of-distribution generalization.",
        ],
    }

    save_reports(
        report,
        rows,
        json_path,
        csv_path,
    )

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(
        f"Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1: {metrics['macro_f1']:.4f}"
    )

    routing = metrics["routing"]

    print(
        f"Routing coverage: "
        f"{routing['coverage'] * 100:.2f}%"
    )

    accepted_accuracy = routing["accepted_accuracy"]

    if accepted_accuracy is None:
        print("Accepted accuracy: N/A (no accepted images)")
    else:
        print(
            f"Accepted accuracy: "
            f"{accepted_accuracy * 100:.2f}%"
        )

    print(
        "Heavy -> Light errors:",
        routing["heavy_to_light_total"],
    )

    print(
        "Heavy -> Light errors accepted:",
        routing["heavy_to_light_accepted"],
    )

    print(
        "Heavy -> Light errors blocked:",
        routing["heavy_to_light_blocked"],
    )

    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    print("\nJSON report:", json_path)
    print("CSV predictions:", csv_path)

    print("\nNo training or restoration performed.")

    return report


# ---------------------------------------------------------
# 5. COMMAND-LINE ENTRYPOINT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate binary Light/Heavy severity "
            "and automatic routing on validation images."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV,
    )

    args = parser.parse_args()

    if not (
        0.5 <= args.confidence_threshold <= 1.0
    ):
        parser.error(
            "Confidence threshold must be in [0.5, 1.0]."
        )

    if not (
        0.0 <= args.margin_threshold <= 1.0
    ):
        parser.error(
            "Margin threshold must be in [0, 1]."
        )

    evaluate_validation(
        manifest_path=args.manifest,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        json_path=args.json_output,
        csv_path=args.csv_output,
    )


if __name__ == "__main__":
    main()
