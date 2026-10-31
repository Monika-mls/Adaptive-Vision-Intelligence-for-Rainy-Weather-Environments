
"""
Adaptive Vision Intelligence for Rainy Weather Environments

Phase 3 - Step 15:
Offline Binary Severity Threshold Analysis

Input:
    docs/severity_validation_predictions.csv

Outputs:
    docs/threshold_analysis/threshold_comparison.csv
    docs/threshold_analysis/threshold_analysis.json
    docs/threshold_analysis/heavy_to_light_errors.csv

No model loading, training, image access or test split access.

This is exploratory VALIDATION analysis.
Do not treat the selected threshold as final test performance.
"""

import argparse
import csv
import json
import math
import os
import tempfile

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


INPUT_CSV = Path(
    "docs/severity_validation_predictions.csv"
)

OUTPUT_DIR = Path("docs/threshold_analysis")

DEFAULT_THRESHOLDS = (
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
)

CLASS_NAMES = ("Light", "Heavy")

EXPECTED_COUNTS = {
    "Light": 112,
    "Heavy": 112,
}

BASELINE = {
    "correct": 163,
    "confusion_matrix": [[90, 22], [39, 73]],
    "accepted": 103,
    "accepted_correct": 84,
    "heavy_to_light_accepted": 13,
}


# ---------------------------------------------------------
# 1. DATA VALIDATION
# ---------------------------------------------------------

def parse_bool(value):
    """Read true/false values from the evaluation CSV."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "true":
            return True

        if normalized == "false":
            return False

    raise ValueError(f"Invalid boolean: {value!r}")


def load_predictions(path=INPUT_CSV):
    """
    Load the existing validation predictions.

    Never opens input images or model checkpoints.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {path}"
        )

    required = {
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
    }

    rows = []
    seen = set()
    counts = Counter()

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        if not required.issubset(reader.fieldnames or []):
            missing = required - set(
                reader.fieldnames or []
            )
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for record in reader:
            relative_path = record[
                "relative_path"
            ].replace("\\", "/")

            actual = record["actual_class"]
            predicted = record["predicted_class"]

            if actual not in CLASS_NAMES:
                raise ValueError(
                    f"Invalid actual label: {actual}"
                )

            if predicted not in CLASS_NAMES:
                raise ValueError(
                    f"Invalid predicted label: {predicted}"
                )

            expected_folder = (
                "RealRain-1k-L"
                if actual == "Light"
                else "RealRain-1k-H"
            )

            parts = relative_path.split("/")

            if (
                len(parts) != 4
                or parts[0] != expected_folder
                or parts[1] != "validation"
                or parts[2] != "input"
                or not parts[3].lower().endswith(".png")
                or any(p in ("", ".", "..") for p in parts)
                or ":" in relative_path
            ):
                raise ValueError(
                    f"Non-validation or invalid path: "
                    f"{relative_path}"
                )

            if relative_path in seen:
                raise ValueError(
                    f"Duplicate image: {relative_path}"
                )

            seen.add(relative_path)

            p_light = float(
                record["light_probability"]
            )

            p_heavy = float(
                record["heavy_probability"]
            )

            confidence = float(
                record["confidence"]
            )

            if not all(
                math.isfinite(v)
                for v in (p_light, p_heavy, confidence)
            ):
                raise ValueError(
                    "Non-finite probability or confidence."
                )

            if (
                not 0 <= p_light <= 1
                or not 0 <= p_heavy <= 1
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    "Probabilities must lie within [0, 1]."
                )

            # predict.py rounds probabilities to 6 decimals.
            if abs(p_light + p_heavy - 1) > 0.000003:
                raise ValueError(
                    "Class probabilities do not sum to one."
                )

            expected_prediction = (
                "Heavy" if p_heavy > p_light
                else "Light"
            )

            if predicted != expected_prediction:
                raise ValueError(
                    "CSV prediction disagrees with probabilities."
                )

            expected_confidence = max(
                p_light, p_heavy
            )

            if abs(
                confidence - expected_confidence
            ) > 0.000003:
                raise ValueError(
                    "CSV confidence mismatch."
                )

            uncertain = parse_bool(
                record["uncertain"]
            )

            accepted = parse_bool(
                record["automatic_routing_allowed"]
            )

            correct = parse_bool(
                record["correct"]
            )

            if accepted == uncertain:
                raise ValueError(
                    "Inconsistent uncertainty and routing flags."
                )

            if correct != (actual == predicted):
                raise ValueError(
                    "Incorrect correctness flag."
                )

            expected_route = (
                "PReNet"
                if predicted == "Light"
                else "VDMamba"
            )

            if (
                record["suggested_restoration_model"]
                != expected_route
            ):
                raise ValueError(
                    "Restoration route does not match class."
                )

            counts[actual] += 1

            rows.append({
                "relative_path": relative_path,
                "actual_class": actual,
                "baseline_predicted_class": predicted,
                "baseline_confidence": confidence,
                "baseline_accepted": accepted,
                "p_light": p_light,
                "p_heavy": p_heavy,
            })

    if counts != Counter(EXPECTED_COUNTS):
        raise ValueError(
            "Expected 112 Light and 112 Heavy "
            f"validation images; found {dict(counts)}"
        )

    return rows


# ---------------------------------------------------------
# 2. THRESHOLD-BASED DECISION
# ---------------------------------------------------------

def threshold_decision(
    p_heavy,
    heavy_threshold=0.50,
    confidence_threshold=0.75,
    margin_threshold=0.20,
):
    """
    Simulate a binary decision and uncertainty gate.

    Class threshold and acceptance threshold are
    deliberately kept separate.
    """

    values = (
        p_heavy,
        heavy_threshold,
        confidence_threshold,
        margin_threshold,
    )

    if not all(
        isinstance(x, (int, float))
        and math.isfinite(x)
        for x in values
    ):
        raise ValueError(
            "Thresholds and probabilities must be finite."
        )

    if not 0 <= p_heavy <= 1:
        raise ValueError(
            "Heavy probability must be in [0, 1]."
        )

    if not 0 < heavy_threshold < 1:
        raise ValueError(
            "Heavy threshold must be in (0, 1)."
        )

    if not 0.5 <= confidence_threshold <= 1:
        raise ValueError(
            "Confidence threshold must be in [0.5, 1]."
        )

    if not 0 <= margin_threshold <= 1:
        raise ValueError(
            "Margin threshold must be in [0, 1]."
        )

    p_light = 1.0 - p_heavy

    predicted = (
        "Heavy"
        if p_heavy >= heavy_threshold
        else "Light"
    )

    # Confidence belongs to the SELECTED class,
    # not necessarily the highest-probability class.
    selected_confidence = (
        p_heavy
        if predicted == "Heavy"
        else p_light
    )

    margin = abs(p_heavy - p_light)

    confidence_ok = (
        selected_confidence >= confidence_threshold
    )

    margin_ok = (
        margin >= margin_threshold
    )

    accepted = confidence_ok and margin_ok

    return {
        "predicted_class": predicted,
        "selected_confidence": selected_confidence,
        "margin": margin,
        "automatic_routing_allowed": accepted,
        "uncertain": not accepted,
        "suggested_restoration_model": (
            "VDMamba"
            if predicted == "Heavy"
            else "PReNet"
        ),
    }


# ---------------------------------------------------------
# 3. CLASSIFICATION METRICS
# ---------------------------------------------------------

def safe_divide(a, b):
    return float(a / b) if b else 0.0


def confusion_metrics(matrix):
    """Compute accuracy, precision, recall and F1."""

    if (
        len(matrix) != 2
        or any(len(row) != 2 for row in matrix)
    ):
        raise ValueError(
            "Confusion matrix must be 2x2."
        )

    total = sum(
        sum(row) for row in matrix
    )

    correct = matrix[0][0] + matrix[1][1]

    per_class = {}

    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index][index]

        support = sum(matrix[index])

        predicted_total = sum(
            matrix[row][index]
            for row in range(2)
        )

        precision = safe_divide(
            tp, predicted_total
        )

        recall = safe_divide(
            tp, support
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": safe_divide(correct, total),
        "macro_f1": sum(
            m["f1"] for m in per_class.values()
        ) / 2,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------
# 4. EVALUATE ONE THRESHOLD
# ---------------------------------------------------------

def evaluate_threshold(
    rows,
    heavy_threshold,
    confidence_threshold=0.75,
    margin_threshold=0.20,
):
    """Calculate classification and routing metrics."""

    if not rows:
        raise ValueError("Prediction rows are empty.")

    matrix = [[0, 0], [0, 0]]

    index_map = {
        "Light": 0,
        "Heavy": 1,
    }

    accepted = 0
    accepted_correct = 0
    blocked_correct = 0
    blocked_incorrect = 0

    heavy_to_light_accepted = 0

    details = []

    for row in rows:
        actual = row["actual_class"]

        if actual not in index_map:
            raise ValueError(
                f"Unexpected class: {actual}"
            )

        decision = threshold_decision(
            p_heavy=row["p_heavy"],
            heavy_threshold=heavy_threshold,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )

        predicted = decision["predicted_class"]

        matrix[
            index_map[actual]
        ][index_map[predicted]] += 1

        correct = actual == predicted
        is_accepted = decision[
            "automatic_routing_allowed"
        ]

        if is_accepted:
            accepted += 1

            if correct:
                accepted_correct += 1
            elif (
                actual == "Heavy"
                and predicted == "Light"
            ):
                heavy_to_light_accepted += 1
        else:
            if correct:
                blocked_correct += 1
            else:
                blocked_incorrect += 1

        details.append({
            "relative_path": row["relative_path"],
            "actual_class": actual,
            "predicted_class": predicted,
            "p_heavy": row["p_heavy"],
            "selected_confidence": decision[
                "selected_confidence"
            ],
            "correct": correct,
            "automatic_routing_allowed": is_accepted,
            "suggested_restoration_model": decision[
                "suggested_restoration_model"
            ],
        })

    classification = confusion_metrics(matrix)

    total = len(rows)

    heavy_to_light_total = matrix[1][0]
    heavy_support = sum(matrix[1])

    routing = {
        "accepted": accepted,
        "blocked": total - accepted,
        "coverage": safe_divide(accepted, total),
        "accepted_correct": accepted_correct,
        "accepted_incorrect": accepted - accepted_correct,
        "accepted_accuracy": (
            safe_divide(accepted_correct, accepted)
            if accepted else None
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
    }

    return {
        "heavy_threshold": heavy_threshold,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "classification": classification,
        "routing": routing,
    }, details


# ---------------------------------------------------------
# 5. BASELINE REPRODUCTION
# ---------------------------------------------------------

def audit_baseline(rows):
    """
    Ensure our threshold analysis reproduces the
    known epoch-8 validation and routing results.
    """

    result, _ = evaluate_threshold(
        rows,
        heavy_threshold=0.50,
        confidence_threshold=0.75,
        margin_threshold=0.20,
    )

    matrix = result[
        "classification"
    ]["confusion_matrix"]

    correct = matrix[0][0] + matrix[1][1]

    routing = result["routing"]

    checks = {
        "correct": correct == BASELINE["correct"],
        "confusion_matrix": (
            matrix == BASELINE["confusion_matrix"]
        ),
        "accepted": (
            routing["accepted"] == BASELINE["accepted"]
        ),
        "accepted_correct": (
            routing["accepted_correct"]
            == BASELINE["accepted_correct"]
        ),
        "heavy_to_light_accepted": (
            routing["heavy_to_light_accepted"]
            == BASELINE["heavy_to_light_accepted"]
        ),
    }

    if not all(checks.values()):
        raise RuntimeError(
            "Baseline audit failed. Do not compare "
            "thresholds until the existing validation "
            f"results are reproduced. Checks: {checks}"
        )

    return {
        "passed": True,
        "checks": checks,
    }


# ---------------------------------------------------------
# 6. ANALYZE HEAVY-TO-LIGHT MISTAKES
# ---------------------------------------------------------

def analyze_heavy_errors(rows):
    """Study existing baseline Heavy-to-Light errors."""

    errors = [
        row for row in rows
        if row["actual_class"] == "Heavy"
        and row["baseline_predicted_class"] == "Light"
    ]

    errors = sorted(
        errors,
        key=lambda row: row["p_heavy"],
        reverse=True,
    )

    def count_between(low, high):
        return sum(
            low <= row["p_heavy"] < high
            for row in errors
        )

    return {
        "total": len(errors),
        "baseline_accepted": sum(
            row["baseline_accepted"]
            for row in errors
        ),
        "heavy_probability_bands": {
            "0.00_to_0.20": count_between(0, 0.20),
            "0.20_to_0.30": count_between(0.20, 0.30),
            "0.30_to_0.40": count_between(0.30, 0.40),
            "0.40_to_0.50": count_between(0.40, 0.50),
        },
        "mean_heavy_probability": (
            sum(row["p_heavy"] for row in errors)
            / len(errors)
            if errors else None
        ),
        "errors": [
            {
                "relative_path": row["relative_path"],
                "p_heavy": row["p_heavy"],
                "p_light": row["p_light"],
                "baseline_confidence": row[
                    "baseline_confidence"
                ],
                "baseline_accepted": row[
                    "baseline_accepted"
                ],
            }
            for row in errors
        ],
    }


# ---------------------------------------------------------
# 7. REPORT WRITING
# ---------------------------------------------------------

def atomic_write(path, text):
    """Replace a report without leaving partial output."""

    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8",
            newline=""
        ) as file:
            file.write(text)

        os.replace(temporary, path)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_csv(rows, fields):
    from io import StringIO

    buffer = StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(rows)

    return buffer.getvalue()


def save_reports(
    comparisons,
    analysis,
    output_dir=OUTPUT_DIR,
):
    output_dir = Path(output_dir)

    comparison_rows = []

    for item in comparisons:
        classification = item["classification"]
        routing = item["routing"]

        comparison_rows.append({
            "heavy_threshold": item["heavy_threshold"],
            "accuracy": classification["accuracy"],
            "macro_f1": classification["macro_f1"],
            "light_recall": classification[
                "per_class"
            ]["Light"]["recall"],
            "heavy_recall": classification[
                "per_class"
            ]["Heavy"]["recall"],
            "heavy_to_light_total": routing[
                "heavy_to_light_total"
            ],
            "coverage": routing["coverage"],
            "accepted": routing["accepted"],
            "accepted_accuracy": routing[
                "accepted_accuracy"
            ],
            "accepted_incorrect": routing[
                "accepted_incorrect"
            ],
            "heavy_to_light_accepted": routing[
                "heavy_to_light_accepted"
            ],
            "heavy_to_light_blocked": routing[
                "heavy_to_light_blocked"
            ],
        })

    comparison_fields = list(
        comparison_rows[0].keys()
    )

    atomic_write(
        output_dir / "threshold_comparison.csv",
        make_csv(
            comparison_rows,
            comparison_fields,
        ),
    )

    atomic_write(
        output_dir / "threshold_analysis.json",
        json.dumps(
            analysis,
            indent=2,
            allow_nan=False,
        ) + "\n",
    )

    error_rows = analysis[
        "heavy_error_analysis"
    ]["errors"]

    error_fields = [
        "relative_path",
        "p_heavy",
        "p_light",
        "baseline_confidence",
        "baseline_accepted",
    ]

    atomic_write(
        output_dir / "heavy_to_light_errors.csv",
        make_csv(error_rows, error_fields),
    )


# ---------------------------------------------------------
# 8. COMPLETE OFFLINE ANALYSIS
# ---------------------------------------------------------

def run_analysis(
    input_path=INPUT_CSV,
    output_dir=OUTPUT_DIR,
    thresholds=DEFAULT_THRESHOLDS,
    confidence_threshold=0.75,
    margin_threshold=0.20,
):
    rows = load_predictions(input_path)

    baseline_audit = audit_baseline(rows)

    comparisons = []

    for threshold in thresholds:
        result, _ = evaluate_threshold(
            rows,
            heavy_threshold=threshold,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )

        comparisons.append(result)

    heavy_errors = analyze_heavy_errors(rows)

    analysis = {
        "project": "Adaptive Vision Intelligence",
        "phase": "3 - Step 15",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_csv": str(input_path),
        "dataset_split": "validation",
        "images_analyzed": len(rows),
        "test_split_accessed": False,
        "training_performed": False,
        "model_weights_modified": False,
        "baseline_audit": baseline_audit,
        "uncertainty_policy": {
            "minimum_selected_class_probability": (
                confidence_threshold
            ),
            "minimum_probability_margin": (
                margin_threshold
            ),
            "calibrated": False,
        },
        "threshold_comparisons": comparisons,
        "heavy_error_analysis": heavy_errors,
        "limitations": [
            "Threshold comparison is exploratory "
            "on the validation split.",
            "Thresholds are not probability calibration.",
            "Validation-selected thresholds may yield "
            "optimistic performance estimates.",
            "Accepted prediction accuracy is undefined "
            "when no predictions are accepted.",
            "Predictions are derived from rounded "
            "six-decimal probabilities.",
            "No restoration models were executed.",
        ],
    }

    save_reports(
        comparisons,
        analysis,
        output_dir,
    )

    print("\n" + "=" * 95)
    print("LIGHT/HEAVY THRESHOLD COMPARISON")
    print("=" * 95)

    print(
        f"{'Heavy T':<10}"
        f"{'Accuracy':<11}"
        f"{'Heavy Recall':<15}"
        f"{'Coverage':<12}"
        f"{'Accepted Acc':<16}"
        f"{'H->L Accepted':<14}"
    )

    for item in comparisons:
        c = item["classification"]
        r = item["routing"]

        accepted_acc = (
            f"{r['accepted_accuracy'] * 100:.2f}%"
            if r["accepted_accuracy"] is not None
            else "N/A"
        )

        print(
            f"{item['heavy_threshold']:<10.2f}"
            f"{c['accuracy'] * 100:<11.2f}"
            f"{c['per_class']['Heavy']['recall'] * 100:<15.2f}"
            f"{r['coverage'] * 100:<12.2f}"
            f"{accepted_acc:<16}"
            f"{r['heavy_to_light_accepted']:<14}"
        )

    print("\nBASELINE AUDIT:", "PASSED")

    print(
        "Baseline Heavy-to-Light mistakes:",
        heavy_errors["total"],
    )

    print(
        "Baseline accepted Heavy-to-Light mistakes:",
        heavy_errors["baseline_accepted"],
    )

    print("\nSaved reports to:", output_dir)
    print("No training, test access or restoration performed.")

    return analysis


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
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

    args = parser.parse_args()

    if (
        not args.thresholds
        or any(
            not math.isfinite(t) or not 0 < t < 1
            for t in args.thresholds
        )
    ):
        parser.error(
            "All Heavy thresholds must lie in (0, 1)."
        )

    # The baseline audit is defined for the original
    # 0.75 confidence / 0.20 margin policy.
    if (
        args.confidence_threshold != 0.75
        or args.margin_threshold != 0.20
    ):
        parser.error(
            "For this baseline-comparison experiment, "
            "keep confidence=0.75 and margin=0.20."
        )

    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )


if __name__ == "__main__":
    main()
