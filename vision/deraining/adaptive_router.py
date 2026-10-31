
"""
Adaptive Vision Intelligence for Rainy Weather Environments

Phase 4 - Step 9: Adaptive Restoration Router

Route policy:
    Accepted Light -> PReNet6
    Accepted Heavy -> VDMamba unavailable
    Uncertain      -> Block automatic restoration

Uses the original epoch-8 severity classifier and the
verified Rain100L PReNet6 restoration adapter.

No training.
No test-split access.
No checkpoint modification.
No silent fallback between restoration models.
"""

import argparse
import json
import time
from pathlib import Path

from vision.severity.predict import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    load_trained_model,
    predict_image,
    save_prediction_json,
)

from vision.deraining.prenet_adapter import (
    DEFAULT_CHECKPOINT as PRENET_CHECKPOINT,
    restore_image,
)


DEFAULT_SEVERITY_CHECKPOINT = Path(
    "checkpoints/severity/best.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/adaptive_router"
)


def decide_route(prediction):
    """
    Make a safe routing decision from the existing
    severity classifier's prediction dictionary.

    Does not execute any restoration model.
    """

    required = {
        "predicted_class",
        "uncertain",
        "automatic_routing_allowed",
        "suggested_restoration_model",
    }

    missing = required - set(prediction)

    if missing:
        raise ValueError(
            f"Prediction missing fields: {sorted(missing)}"
        )

    severity = prediction["predicted_class"]
    uncertain = prediction["uncertain"]
    allowed = prediction["automatic_routing_allowed"]
    suggested = prediction["suggested_restoration_model"]

    if type(uncertain) is not bool or type(allowed) is not bool:
        raise ValueError(
            "Routing gate values must be booleans."
        )

    expected_models = {
        "Light": "PReNet",
        "Heavy": "VDMamba",
    }

    if severity not in expected_models:
        raise ValueError(
            f"Unsupported severity class: {severity}"
        )

    if suggested != expected_models[severity]:
        raise ValueError(
            "Classifier severity and restoration route disagree."
        )

    # Fail closed on inconsistent uncertainty flags.
    if uncertain or not allowed:
        return {
            "status": "blocked_uncertain",
            "executed_model": None,
            "restoration_executed": False,
            "reason": (
                "The classifier did not authorize "
                "automatic restoration."
            ),
        }

    if severity == "Heavy":
        return {
            "status": "model_unavailable",
            "executed_model": None,
            "restoration_executed": False,
            "reason": (
                "The Heavy-rain route requires the exact "
                "VDMamba restoration model, which has not "
                "yet been integrated."
            ),
        }

    return {
        "status": "ready_prenet",
        "executed_model": None,
        "restoration_executed": False,
        "reason": None,
    }


def run_adaptive_restoration(
    image_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    severity_checkpoint=DEFAULT_SEVERITY_CHECKPOINT,
    prenet_checkpoint=PRENET_CHECKPOINT,
    device="auto",
    max_side=256,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold=DEFAULT_MARGIN_THRESHOLD,
):
    """
    Predict severity, gate the route, and execute
    PReNet only for an accepted Light prediction.

    Produces a JSON report for every valid decision.
    """

    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if max_side < 16:
        raise ValueError(
            "max_side must be at least 16."
        )

    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be in [0.5, 1.0]."
        )

    if not 0 <= margin_threshold <= 1.0:
        raise ValueError(
            "margin_threshold must be in [0, 1]."
        )

    report_path = output_dir / "routing_report.json"
    restored_path = output_dir / "restored.png"

    if report_path.exists() or restored_path.exists():
        raise FileExistsError(
            "Output already exists. Use a separate "
            f"output directory: {output_dir}"
        )

    if report_path == image_path or restored_path == image_path:
        raise ValueError(
            "The output must not overwrite the input image."
        )

    start = time.perf_counter()

    classifier, selected_device, metadata = (
        load_trained_model(
            checkpoint_path=severity_checkpoint,
            device=device,
        )
    )

    prediction_result = predict_image(
        image_path=image_path,
        model=classifier,
        device=selected_device,
        metadata=metadata,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    classification_seconds = time.perf_counter() - start

    prediction = prediction_result["prediction"]
    decision = decide_route(prediction)

    report = {
        "project": (
            "Adaptive Vision Intelligence for "
            "Rainy Weather Environments"
        ),
        "input_image": str(image_path),
        "severity_checkpoint": str(
            Path(severity_checkpoint).resolve()
        ),
        "prenet_checkpoint": str(
            Path(prenet_checkpoint).resolve()
        ),
        "classification": prediction_result,
        "routing": dict(decision),
        "restoration": None,
        "output_image": None,
        "timings": {
            "classification_seconds": round(
                classification_seconds, 4
            ),
            "restoration_seconds": None,
            "total_seconds": None,
        },
    }

    if decision["status"] == "ready_prenet":
        # This is the ONLY branch allowed to run PReNet.
        restoration = restore_image(
            input_path=image_path,
            output_path=restored_path,
            checkpoint_path=prenet_checkpoint,
            device=str(selected_device),
            max_side=max_side,
        )

        if not restored_path.is_file():
            raise RuntimeError(
                "PReNet did not create its output image."
            )

        report["routing"] = {
            "status": "restoration_completed",
            "executed_model": "PReNet",
            "restoration_executed": True,
            "reason": None,
        }

        report["restoration"] = restoration
        report["output_image"] = str(restored_path)

        report["timings"]["restoration_seconds"] = (
            restoration["inference_seconds"]
        )

    report["timings"]["total_seconds"] = round(
        time.perf_counter() - start,
        4,
    )

    # Existing atomic JSON writer can store any
    # JSON-serializable result, not just severity outputs.
    save_prediction_json(report, report_path)

    report["report_path"] = str(report_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Severity-aware PReNet routing with explicit "
            "Heavy and uncertain statuses."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--severity-checkpoint",
        type=Path,
        default=DEFAULT_SEVERITY_CHECKPOINT,
    )

    parser.add_argument(
        "--prenet-checkpoint",
        type=Path,
        default=PRENET_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
    )

    args = parser.parse_args()

    result = run_adaptive_restoration(
        image_path=args.image,
        output_dir=args.output_dir,
        severity_checkpoint=args.severity_checkpoint,
        prenet_checkpoint=args.prenet_checkpoint,
        device=args.device,
        max_side=args.max_side,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )

    prediction = result["classification"]["prediction"]
    routing = result["routing"]

    print("=" * 65)
    print("ADAPTIVE VISION INTELLIGENCE")
    print("SEVERITY-AWARE RESTORATION")
    print("=" * 65)

    print("Input image:", args.image)
    print(
        "Severity checkpoint epoch:",
        result["classification"]["model"][
            "checkpoint_epoch"
        ],
    )

    print("\nPredicted severity:", prediction["predicted_class"])
    print("Confidence:", prediction["confidence"])
    print("Margin:", prediction["probability_margin"])
    print("Uncertain:", prediction["uncertain"])

    print("\nClass probabilities:")
    for name, probability in prediction["probabilities"].items():
        print(f"  {name}: {probability:.4f}")

    print("\nSuggested model:", prediction[
        "suggested_restoration_model"
    ])

    print("Routing status:", routing["status"])
    print("Restoration executed:", routing["restoration_executed"])
    print("Executed model:", routing["executed_model"])

    if routing["reason"]:
        print("Reason:", routing["reason"])

    if result["output_image"]:
        print("\nRestored image:", result["output_image"])
        print(
            "Restoration inference time:",
            result["timings"]["restoration_seconds"],
        )

    print(
        "\nTotal processing time:",
        result["timings"]["total_seconds"],
    )
    print("JSON report:", result["report_path"])
    print("No training performed.")


if __name__ == "__main__":
    main()
