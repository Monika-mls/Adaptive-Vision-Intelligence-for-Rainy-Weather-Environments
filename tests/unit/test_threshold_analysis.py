
"""
Unit tests for offline severity threshold analysis.

No model downloads.
No training.
No GPU requirements.
No RealRain image access.
"""

import csv
import json

import pytest

from vision.severity.threshold_analysis import (
    analyze_heavy_errors,
    audit_baseline,
    confusion_metrics,
    evaluate_threshold,
    load_predictions,
    parse_bool,
    save_reports,
    threshold_decision,
)


def make_row(
    actual,
    p_heavy,
    index=0,
    baseline_accepted=False,
):
    predicted = (
        "Heavy" if p_heavy > 0.5 else "Light"
    )

    track = (
        "RealRain-1k-L"
        if actual == "Light"
        else "RealRain-1k-H"
    )

    return {
        "relative_path": (
            f"{track}/validation/input/{index}.png"
        ),
        "actual_class": actual,
        "baseline_predicted_class": predicted,
        "baseline_confidence": max(
            p_heavy, 1 - p_heavy
        ),
        "baseline_accepted": baseline_accepted,
        "p_light": 1 - p_heavy,
        "p_heavy": p_heavy,
    }


def test_parse_bool():
    assert parse_bool("True") is True
    assert parse_bool("false") is False

    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_light_decision():
    result = threshold_decision(
        p_heavy=0.10
    )

    assert result["predicted_class"] == "Light"
    assert result["automatic_routing_allowed"]
    assert result["suggested_restoration_model"] == "PReNet"


def test_heavy_decision():
    result = threshold_decision(
        p_heavy=0.90
    )

    assert result["predicted_class"] == "Heavy"
    assert result["automatic_routing_allowed"]
    assert result["suggested_restoration_model"] == "VDMamba"


def test_lower_threshold_changes_class():
    baseline = threshold_decision(
        p_heavy=0.45,
        heavy_threshold=0.50,
    )

    alternative = threshold_decision(
        p_heavy=0.45,
        heavy_threshold=0.40,
    )

    assert baseline["predicted_class"] == "Light"
    assert alternative["predicted_class"] == "Heavy"

    # The selected Heavy probability is only 0.45,
    # so this new prediction is blocked by the gate.
    assert alternative["automatic_routing_allowed"] is False


def test_ambiguous_prediction_blocked():
    result = threshold_decision(
        p_heavy=0.52
    )

    assert result["uncertain"]
    assert not result["automatic_routing_allowed"]


def test_threshold_boundaries():
    with pytest.raises(ValueError):
        threshold_decision(
            p_heavy=0.5,
            heavy_threshold=1.2,
        )

    with pytest.raises(ValueError):
        threshold_decision(
            p_heavy=float("nan")
        )


def test_perfect_classification():
    metrics = confusion_metrics([
        [3, 0],
        [0, 2],
    ])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["Heavy"]["recall"] == 1.0


def test_heavy_recall_improves_with_lower_threshold():
    rows = [
        make_row("Light", 0.10, 1),
        make_row("Light", 0.20, 2),
        make_row("Heavy", 0.45, 3),
        make_row("Heavy", 0.90, 4),
    ]

    baseline, _ = evaluate_threshold(
        rows, heavy_threshold=0.50
    )

    alternative, _ = evaluate_threshold(
        rows, heavy_threshold=0.40
    )

    assert (
        baseline["classification"]["per_class"]["Heavy"]["recall"]
        == 0.5
    )

    assert (
        alternative["classification"]["per_class"]["Heavy"]["recall"]
        == 1.0
    )


def test_heavy_error_counts():
    rows = [
        make_row("Heavy", 0.20, 1, True),
        make_row("Heavy", 0.40, 2, False),
        make_row("Heavy", 0.80, 3, True),
        make_row("Light", 0.20, 4, True),
    ]

    result = analyze_heavy_errors(rows)

    assert result["total"] == 2
    assert result["baseline_accepted"] == 1

    assert (
        result["heavy_probability_bands"]["0.20_to_0.30"]
        == 1
    )

    assert (
        result["heavy_probability_bands"]["0.40_to_0.50"]
        == 1
    )


def test_baseline_audit_rejects_wrong_results():
    rows = [
        make_row("Light", 0.10, 1),
        make_row("Heavy", 0.90, 2),
    ]

    with pytest.raises(RuntimeError):
        audit_baseline(rows)


def test_invalid_manifest_counts(tmp_path):
    path = tmp_path / "predictions.csv"

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

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )

        writer.writeheader()

        writer.writerow({
            "relative_path": (
                "RealRain-1k-L/validation/input/1.png"
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
        })

    # One row is insufficient:
    # the validation set must have exactly 224.
    with pytest.raises(ValueError):
        load_predictions(path)


def test_report_writing(tmp_path):
    rows = [
        make_row("Light", 0.10, 1),
        make_row("Heavy", 0.90, 2),
    ]

    result, _ = evaluate_threshold(
        rows, heavy_threshold=0.50
    )

    analysis = {
        "threshold_comparisons": [result],
        "heavy_error_analysis": analyze_heavy_errors(rows),
    }

    output = tmp_path / "reports"

    save_reports(
        comparisons=[result],
        analysis=analysis,
        output_dir=output,
    )

    assert (
        output / "threshold_comparison.csv"
    ).is_file()

    assert (
        output / "threshold_analysis.json"
    ).is_file()

    assert (
        output / "heavy_to_light_errors.csv"
    ).is_file()

    loaded = json.loads(
        (output / "threshold_analysis.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(
        loaded["threshold_comparisons"]
    ) == 1
