
"""
Adaptive Vision Intelligence for Rainy Weather Environments

Modes:
    auto      -> MobileNetV3-Small -> PReNet / Restormer
    prenet    -> Manual Light-rain restoration
    restormer -> Manual Heavy-rain restoration

Uncertain automatic predictions are blocked.

Run:
    python -m backend.app
"""

import json
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "outputs" / "web_restoration"

SEVERITY_CHECKPOINT = ROOT / "checkpoints/severity/best.pt"

RESTORMER_CHECKPOINT = (
    ROOT / "outputs/restormer/finetune/"
    "20261030_210003_966091/best.pt"
)

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_SIZES = {128, 256, 384, 512}
ALLOWED_MODELS = {"auto", "prenet", "restormer"}

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SOURCE_PIXELS = 20_000_000
MAX_HEAVY_PIXELS = 1_200_000

TILE_SIZE = 128
OVERLAP = 32


def run_prenet(input_path, output_path, max_side):
    from vision.deraining.prenet_adapter import restore_image

    return restore_image(
        input_path=input_path,
        output_path=output_path,
        device="auto",
        max_side=max_side,
    )


def run_restormer(input_path, output_path):
    from vision.deraining.restormer_tiled import restore_image

    return restore_image(
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=RESTORMER_CHECKPOINT,
        device_name="auto",
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )


def classify_rain(image_path):
    """
    Use the existing trained MobileNetV3-Small classifier
    and its original confidence/margin thresholds.
    """
    from vision.severity.predict import (
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_MARGIN_THRESHOLD,
        load_trained_model,
        predict_image,
    )

    start = time.perf_counter()

    classifier, device, metadata = load_trained_model(
        checkpoint_path=SEVERITY_CHECKPOINT,
        device="auto",
    )

    result = predict_image(
        image_path=image_path,
        model=classifier,
        device=device,
        metadata=metadata,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        margin_threshold=DEFAULT_MARGIN_THRESHOLD,
    )

    elapsed = time.perf_counter() - start

    return result["prediction"], round(elapsed, 4)


def choose_automatic_model(prediction):
    """
    Fail closed if confidence gates or predictions disagree.
    """
    severity = prediction["predicted_class"]

    expected = {
        "Light": ("PReNet", "prenet"),
        "Heavy": ("VDMamba", "restormer"),
    }

    if severity not in expected:
        raise ValueError("Unsupported severity prediction")

    expected_suggestion, selected_model = expected[severity]

    if (
        prediction["suggested_restoration_model"]
        != expected_suggestion
    ):
        raise ValueError(
            "Classifier prediction and suggested model disagree"
        )

    if (
        type(prediction["uncertain"]) is not bool
        or type(prediction["automatic_routing_allowed"]) is not bool
    ):
        raise ValueError("Invalid routing gate values")

    if (
        prediction["uncertain"]
        or not prediction["automatic_routing_allowed"]
    ):
        return None

    return selected_model


def create_app(test_config=None):
    app = Flask(__name__)

    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        UPLOAD_ROOT=str(UPLOAD_ROOT),
        JSON_SORT_KEYS=False,
    )

    if test_config:
        app.config.update(test_config)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        return jsonify(
            error="Maximum upload size is 12 MB."
        ), 413

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok",
            mode="adaptive_restoration",
            automatic_routing_enabled=True,
            models={
                "auto": "Automatic severity-aware routing",
                "prenet": "PReNet6 (Light rain)",
                "restormer": "Fine-tuned Restormer (Heavy rain)",
            },
            severity_checkpoint_available=(
                SEVERITY_CHECKPOINT.is_file()
            ),
            restormer_checkpoint_available=(
                RESTORMER_CHECKPOINT.is_file()
            ),
        )

    @app.post("/api/restore")
    def restore():
        uploaded = request.files.get("image")

        if uploaded is None or not uploaded.filename:
            return jsonify(
                error="Choose a rainy image to upload."
            ), 400

        suffix = Path(uploaded.filename).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            return jsonify(
                error="Upload a PNG, JPG, JPEG or WEBP image."
            ), 400

        selected_mode = request.form.get(
            "model", "auto"
        ).strip().lower()

        if selected_mode not in ALLOWED_MODELS:
            return jsonify(
                error="Choose auto, prenet or restormer."
            ), 400

        try:
            max_side = int(
                request.form.get("max_side", "256")
            )
        except (TypeError, ValueError):
            return jsonify(
                error="Invalid processing dimension."
            ), 400

        if max_side not in ALLOWED_SIZES:
            return jsonify(
                error="Choose 128, 256, 384 or 512 pixels."
            ), 400

        try:
            with Image.open(uploaded.stream) as opened:
                opened.verify()

            uploaded.stream.seek(0)

            with Image.open(uploaded.stream) as opened:
                if (
                    opened.width * opened.height
                    > MAX_SOURCE_PIXELS
                ):
                    return jsonify(
                        error="Image exceeds 20 megapixels."
                    ), 400

                rgb = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            return jsonify(
                error="Uploaded file is not a valid image."
            ), 400

        request_id = uuid.uuid4().hex
        job_dir = Path(app.config["UPLOAD_ROOT"]) / request_id
        job_dir.mkdir(parents=True, exist_ok=False)

        input_path = job_dir / "input.png"
        output_path = job_dir / "restored.png"

        rgb.save(input_path, format="PNG")
        width, height = rgb.size

        start = time.perf_counter()

        classification = None
        classification_seconds = None
        executed_model = selected_mode

        if selected_mode == "auto":
            if not SEVERITY_CHECKPOINT.is_file():
                return jsonify(
                    error="Severity classifier checkpoint not found."
                ), 503

            try:
                classification, classification_seconds = (
                    classify_rain(input_path)
                )

                executed_model = choose_automatic_model(
                    classification
                )

            except Exception:
                app.logger.exception(
                    "Severity prediction failed for %s",
                    request_id,
                )
                return jsonify(
                    error="Severity classification failed."
                ), 500

            if executed_model is None:
                blocked = {
                    "request_id": request_id,
                    "mode": "automatic",
                    "status": "blocked_uncertain",
                    "severity_classification_performed": True,
                    "classification": classification,
                    "classification_seconds": (
                        classification_seconds
                    ),
                    "selected_model": None,
                    "restoration_executed": False,
                    "restored_url": None,
                    "input_url": (
                        f"/api/results/{request_id}/input.png"
                    ),
                    "total_processing_seconds": round(
                        time.perf_counter() - start, 4
                    ),
                    "message": (
                        "Rain severity prediction is uncertain. "
                        "Choose Light or Heavy manually."
                    ),
                }

                (job_dir / "result.json").write_text(
                    json.dumps(blocked, indent=2),
                    encoding="utf-8",
                )

                return jsonify(blocked)

        if (
            executed_model == "restormer"
            and width * height > MAX_HEAVY_PIXELS
        ):
            return jsonify(
                error=(
                    "Heavy-rain restoration currently supports "
                    "images up to 1.2 megapixels."
                )
            ), 400

        if (
            executed_model == "restormer"
            and not RESTORMER_CHECKPOINT.is_file()
        ):
            return jsonify(
                error="Fine-tuned Restormer checkpoint not found."
            ), 503

        try:
            if executed_model == "prenet":
                result = run_prenet(
                    input_path,
                    output_path,
                    max_side,
                )
            else:
                result = run_restormer(
                    input_path,
                    output_path,
                )

            if not output_path.is_file():
                raise RuntimeError(
                    "Restoration did not create an output image."
                )

        except Exception:
            app.logger.exception(
                "Restoration failed for job %s",
                request_id,
            )
            return jsonify(
                error=(
                    "Restoration failed. "
                    "Check the Flask terminal."
                )
            ), 500

        total_seconds = round(
            time.perf_counter() - start,
            4,
        )

        if executed_model == "prenet":
            model_name = "PReNet6 (Rain100L pretrained)"
            processed_size = result["processed_size"]
            stages = result["recurrent_stages"]
            tiles = None
        else:
            model_name = "Fine-tuned Restormer"
            processed_size = result["original_size"]
            stages = None
            tiles = result["tiles_processed"]

        response = {
            "request_id": request_id,
            "mode": (
                "automatic"
                if selected_mode == "auto"
                else f"direct_{executed_model}"
            ),
            "status": "restoration_completed",
            "selected_model": executed_model,
            "model": model_name,
            "severity_classification_performed": (
                classification is not None
            ),
            "classification": classification,
            "classification_seconds": (
                classification_seconds
            ),
            "restoration_executed": True,
            "original_size": result["original_size"],
            "processed_size": processed_size,
            "output_size": result["output_size"],
            "device": result["device"],
            "recurrent_stages": stages,
            "tiles_processed": tiles,
            "inference_seconds": (
                result["inference_seconds"]
            ),
            "peak_gpu_allocated_mib": (
                result["peak_gpu_allocated_mib"]
            ),
            "total_processing_seconds": total_seconds,
            "input_url": (
                f"/api/results/{request_id}/input.png"
            ),
            "restored_url": (
                f"/api/results/{request_id}/restored.png"
            ),
            "quality_evaluated": False,
            "warning": (
                "Restoration effectiveness can vary across "
                "images, especially external photographs."
                if executed_model == "restormer"
                else None
            ),
        }

        (job_dir / "result.json").write_text(
            json.dumps(response, indent=2),
            encoding="utf-8",
        )

        return jsonify(response)

    @app.get("/api/results/<request_id>/<name>")
    def results(request_id, name):
        if (
            len(request_id) != 32
            or any(
                char not in "0123456789abcdef"
                for char in request_id
            )
        ):
            return jsonify(
                error="Invalid result identifier."
            ), 404

        if name not in {"input.png", "restored.png"}:
            return jsonify(
                error="File unavailable."
            ), 404

        job_dir = (
            Path(app.config["UPLOAD_ROOT"]) / request_id
        )

        if not (job_dir / name).is_file():
            return jsonify(
                error="Image not found."
            ), 404

        return send_from_directory(
            job_dir,
            name,
            mimetype="image/png",
            as_attachment=False,
        )

    return app


app = create_app()


if __name__ == "__main__":
    print(
        "Adaptive Vision API: "
        "http://127.0.0.1:5000/api/health"
    )
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
