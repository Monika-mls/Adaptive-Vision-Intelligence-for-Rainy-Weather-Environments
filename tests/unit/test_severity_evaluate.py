
"""
Unit tests for Light/Heavy severity evaluation.

Uses synthetic prediction records and temporary files.

No ImageNet downloads.
No model training.
No RealRain test images.
"""

import csv
import json

import pytest

from vision.severity.evaluate import (
    load_validation_samples,
    summarize_predictions,
    save_reports,
)


def prediction(
    actual,
    predicted,
    accepted=True,
    confidence=0.9,
):
    return {
        "actual_class": actual,
        "predicted_class": predicted,
        "confidence": confidence,
        "uncertain": not accepted,
        "automatic_routing_allowed": accepted,
    }


def test_perfect_predictions():
    rows = [
        prediction("Light", "Light"),
        prediction("Heavy", "Heavy"),
    ]

    result = summarize_predictions(rows)

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]


def test_confusion_matrix():
    rows = [
        prediction("Light", "Light"),
        prediction("Light", "Heavy"),
        prediction("Heavy", "Light"),
        prediction("Heavy", "Heavy"),
    ]

    result = summarize_predictions(rows)

    assert result["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]

    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == 0.5


def test_routing_coverage():
    rows = [
        prediction("Light", "Light", True),
        prediction("Heavy", "Heavy", True),
        prediction("Light", "Heavy", False),
        prediction("Heavy", "Light", False),
    ]

    routing = summarize_predictions(rows)["routing"]

    assert routing["accepted"] == 2
    assert routing["blocked"] == 2
    assert routing["coverage"] == 0.5
    assert routing["accepted_accuracy"] == 1.0
    assert routing["blocked_incorrect"] == 2


def test_heavy_to_light_errors():
    rows = [
        prediction("Heavy", "Light", True),
        prediction("Heavy", "Light", False),
        prediction("Heavy", "Heavy", True),
        prediction("Light", "Light", True),
    ]

    routing = summarize_predictions(rows)["routing"]

    assert routing["heavy_to_light_total"] == 2
    assert routing["heavy_to_light_accepted"] == 1
    assert routing["heavy_to_light_blocked"] == 1
    assert routing["heavy_to_light_accepted_rate"] == 1 / 3


def test_no_accepted_predictions():
    rows = [
        prediction("Light", "Light", False),
        prediction("Heavy", "Light", False),
    ]

    routing = summarize_predictions(rows)["routing"]

    assert routing["coverage"] == 0.0
    assert routing["accepted_accuracy"] is None
    assert routing["accepted"] == 0
    assert routing["blocked"] == 2


def test_empty_predictions_rejected():
    with pytest.raises(ValueError):
        summarize_predictions([])


def test_invalid_class_rejected():
    rows = [
        prediction("Medium", "Light")
    ]

    with pytest.raises(ValueError):
        summarize_predictions(rows)


def test_inconsistent_routing_flags_rejected():
    row = prediction("Light", "Light")
    row["uncertain"] = True

    with pytest.raises(ValueError):
        summarize_predictions([row])


def test_reports_saved(tmp_path):
    rows = [{
        "relative_path": (
            "RealRain-1k-L/validation/input/1007.png"
        ),
        "actual_class": "Light",
        "predicted_class": "Light",
        "confidence": 0.9,
        "light_probability": 0.9,
        "heavy_probability": 0.1,
        "uncertain": False,
        "automatic_routing_allowed": True,
        "suggested_restoration_model": "PReNet",
        "correct": True,
    }]

    report = {
        "metrics": summarize_predictions(rows)
    }

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "predictions.csv"

    save_reports(
        report,
        rows,
        json_path,
        csv_path,
    )

    assert json_path.is_file()
    assert csv_path.is_file()

    loaded = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    assert loaded["metrics"]["accuracy"] == 1.0

    with csv_path.open(
        newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    assert len(records) == 1
    assert records[0]["actual_class"] == "Light"


def test_manifest_ignores_test_split(tmp_path):
    manifest = tmp_path / "labels.csv"
    data_root = tmp_path / "datasets"
    data_root.mkdir()

    with manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "severity",
                "rainy_path",
            ],
        )

        writer.writeheader()

        # A test entry must never cause a file read.
        writer.writerow({
            "split": "test",
            "severity": "Light",
            "rainy_path": (
                "RealRain-1k-L/test/input/missing.png"
            ),
        })

        # No validation entries means the count audit
        # should reject the manifest cleanly.
        # It must not request the test image.

    with pytest.raises(ValueError):
        load_validation_samples(
            manifest,
            data_root,
        )
