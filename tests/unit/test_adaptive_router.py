
"""
Phase 4: Adaptive router unit tests.

Tests routing safety without running actual ML models.

No CUDA, downloads, training or checkpoint changes.
"""

from pathlib import Path

import pytest
from PIL import Image

import vision.deraining.adaptive_router as router


def make_prediction(
    severity="Light",
    uncertain=False,
    allowed=True,
):
    models = {
        "Light": "PReNet",
        "Heavy": "VDMamba",
    }

    return {
        "predicted_class": severity,
        "predicted_class_index": (
            0 if severity == "Light" else 1
        ),
        "confidence": 0.92,
        "probability_margin": 0.84,
        "probabilities": {
            "Light": 0.92 if severity == "Light" else 0.08,
            "Heavy": 0.08 if severity == "Light" else 0.92,
        },
        "uncertain": uncertain,
        "uncertainty_reasons": (
            ["confidence_below_threshold"]
            if uncertain else []
        ),
        "automatic_routing_allowed": allowed,
        "suggested_restoration_model": models[severity],
        "restoration_executed": False,
    }


def test_accepted_light_routes_to_prenet():
    decision = router.decide_route(
        make_prediction("Light")
    )

    assert decision["status"] == "ready_prenet"


def test_accepted_heavy_does_not_route_to_prenet():
    decision = router.decide_route(
        make_prediction("Heavy")
    )

    assert decision["status"] == "model_unavailable"
    assert decision["restoration_executed"] is False


def test_uncertain_light_is_blocked():
    decision = router.decide_route(
        make_prediction(
            "Light",
            uncertain=True,
            allowed=False,
        )
    )

    assert decision["status"] == "blocked_uncertain"


def test_uncertain_heavy_is_blocked():
    decision = router.decide_route(
        make_prediction(
            "Heavy",
            uncertain=True,
            allowed=False,
        )
    )

    assert decision["status"] == "blocked_uncertain"


def test_denied_light_is_blocked():
    decision = router.decide_route(
        make_prediction(
            "Light",
            uncertain=False,
            allowed=False,
        )
    )

    assert decision["status"] == "blocked_uncertain"


def test_route_mismatch_is_rejected():
    prediction = make_prediction("Heavy")
    prediction["suggested_restoration_model"] = "PReNet"

    with pytest.raises(ValueError, match="disagree"):
        router.decide_route(prediction)


def test_unknown_class_is_rejected():
    prediction = make_prediction("Light")
    prediction["predicted_class"] = "Medium"

    with pytest.raises(ValueError, match="Unsupported"):
        router.decide_route(prediction)


def setup_mock_pipeline(monkeypatch, prediction, calls):
    def fake_load(**kwargs):
        calls["load"] += 1
        return object(), "cpu", {
            "checkpoint_epoch": 8,
            "device": "cpu",
        }

    def fake_predict(**kwargs):
        calls["predict"] += 1
        return {
            "prediction": prediction,
            "model": {
                "checkpoint_epoch": 8,
                "device": "cpu",
            },
        }

    def fake_restore(**kwargs):
        calls["restore"] += 1

        output = Path(kwargs["output_path"])
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.new(
            "RGB",
            (32, 32),
            "gray",
        ).save(output)

        return {
            "restoration_completed": True,
            "inference_seconds": 0.01,
            "peak_gpu_allocated_mib": None,
        }

    monkeypatch.setattr(
        router,
        "load_trained_model",
        fake_load,
    )

    monkeypatch.setattr(
        router,
        "predict_image",
        fake_predict,
    )

    monkeypatch.setattr(
        router,
        "restore_image",
        fake_restore,
    )


@pytest.mark.parametrize(
    "severity,uncertain,allowed,expected_calls,expected_status",
    [
        ("Light", False, True, 1, "restoration_completed"),
        ("Heavy", False, True, 0, "model_unavailable"),
        ("Light", True, False, 0, "blocked_uncertain"),
        ("Heavy", True, False, 0, "blocked_uncertain"),
    ],
)
def test_pipeline_executes_only_accepted_light(
    tmp_path,
    monkeypatch,
    severity,
    uncertain,
    allowed,
    expected_calls,
    expected_status,
):
    calls = {
        "load": 0,
        "predict": 0,
        "restore": 0,
    }

    prediction = make_prediction(
        severity=severity,
        uncertain=uncertain,
        allowed=allowed,
    )

    setup_mock_pipeline(
        monkeypatch,
        prediction,
        calls,
    )

    input_path = tmp_path / "rainy.png"

    Image.new(
        "RGB",
        (32, 32),
        "gray",
    ).save(input_path)

    output_dir = tmp_path / "output"

    result = router.run_adaptive_restoration(
        image_path=input_path,
        output_dir=output_dir,
        device="cpu",
    )

    assert result["routing"]["status"] == expected_status
    assert calls["load"] == 1
    assert calls["predict"] == 1
    assert calls["restore"] == expected_calls

    assert (output_dir / "routing_report.json").is_file()

    if expected_calls:
        assert (output_dir / "restored.png").is_file()
        assert result["routing"]["executed_model"] == "PReNet"
    else:
        assert not (output_dir / "restored.png").exists()
        assert result["routing"]["restoration_executed"] is False


def test_existing_output_is_not_overwritten(tmp_path):
    input_path = tmp_path / "rainy.png"

    Image.new(
        "RGB",
        (32, 32),
        "gray",
    ).save(input_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    (output_dir / "routing_report.json").write_text(
        "previous experiment",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        router.run_adaptive_restoration(
            image_path=input_path,
            output_dir=output_dir,
        )

    assert (
        output_dir / "routing_report.json"
    ).read_text(encoding="utf-8") == "previous experiment"
