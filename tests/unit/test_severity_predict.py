
"""
Tests for binary severity prediction.

No dataset downloads.
No actual training.
No GPU required.
No modification of the trained best.pt checkpoint.
"""

import importlib
import json

import pytest
import torch
from PIL import Image
from torch import nn


predict = importlib.import_module(
    "vision.severity.predict"
)


class TinyTestModel(nn.Module):
    """Small stand-in model for checkpoint tests."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, images):
        features = images.mean(dim=(2, 3))
        return self.classifier(features)


def make_image(tmp_path, filename="rain.png"):
    path = tmp_path / filename

    Image.new(
        "RGB",
        (400, 300),
        color=(110, 130, 150),
    ).save(path)

    return path


def make_checkpoint(tmp_path, classes=None):
    model = TinyTestModel()

    path = tmp_path / "best.pt"

    torch.save({
        "epoch": 8,
        "architecture": "mobilenet_v3_small",
        "preprocessing": "IMAGENET1K_V1",
        "classes": classes or ["Light", "Heavy"],
        "best_macro_f1": 0.7261,
        "model_state_dict": model.state_dict(),
    }, path)

    return path


def test_class_mapping():
    assert predict.CLASS_NAMES == ("Light", "Heavy")

    assert predict.RESTORATION_ROUTES == {
        "Light": "PReNet",
        "Heavy": "VDMamba",
    }


def test_light_prediction():
    logits = torch.tensor([[4.0, 0.0]])

    result = predict.interpret_logits(logits)

    assert result["predicted_class"] == "Light"
    assert result["predicted_class_index"] == 0
    assert result["suggested_restoration_model"] == "PReNet"
    assert result["uncertain"] is False
    assert result["automatic_routing_allowed"] is True
    assert result["restoration_executed"] is False


def test_heavy_prediction():
    logits = torch.tensor([[0.0, 4.0]])

    result = predict.interpret_logits(logits)

    assert result["predicted_class"] == "Heavy"
    assert result["predicted_class_index"] == 1
    assert result["suggested_restoration_model"] == "VDMamba"
    assert result["uncertain"] is False


def test_probabilities_sum_to_one():
    result = predict.interpret_logits(
        torch.tensor([[0.5, 1.5]])
    )

    total = sum(result["probabilities"].values())

    assert total == pytest.approx(1.0, abs=1e-5)


def test_ambiguous_prediction_is_blocked():
    result = predict.interpret_logits(
        torch.tensor([[0.0, 0.0]])
    )

    assert result["uncertain"] is True
    assert result["automatic_routing_allowed"] is False

    assert "confidence_below_threshold" in (
        result["uncertainty_reasons"]
    )

    assert "probability_margin_below_threshold" in (
        result["uncertainty_reasons"]
    )


def test_low_margin_threshold():
    logits = torch.log(
        torch.tensor([[0.56, 0.44]])
    )

    result = predict.interpret_logits(
        logits,
        confidence_threshold=0.5,
        margin_threshold=0.2,
    )

    assert result["uncertain"] is True
    assert result["uncertainty_reasons"] == [
        "probability_margin_below_threshold"
    ]


def test_invalid_logits():
    with pytest.raises(ValueError):
        predict.interpret_logits(
            torch.tensor([[1.0, 2.0, 3.0]])
        )


def test_nonfinite_logits():
    with pytest.raises(ValueError):
        predict.interpret_logits(
            torch.tensor([[float("nan"), 1.0]])
        )


def test_invalid_threshold():
    with pytest.raises(ValueError):
        predict.interpret_logits(
            torch.tensor([[1.0, 2.0]]),
            confidence_threshold=1.2,
        )


def test_image_preprocessing(tmp_path):
    path = make_image(tmp_path)

    tensor, size = predict.preprocess_image(path)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all().item()

    assert size == {
        "width": 400,
        "height": 300,
    }


def test_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.preprocess_image(
            tmp_path / "missing.png"
        )


def test_invalid_extension(tmp_path):
    path = tmp_path / "image.txt"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        predict.preprocess_image(path)


def test_checkpoint_class_order_rejected(tmp_path):
    path = make_checkpoint(
        tmp_path,
        classes=["Heavy", "Light"],
    )

    checkpoint = torch.load(
        path,
        weights_only=True,
        map_location="cpu",
    )

    with pytest.raises(ValueError):
        predict.validate_checkpoint(checkpoint)


def test_checkpoint_loading_without_download(
    tmp_path, monkeypatch
):
    checkpoint_path = make_checkpoint(tmp_path)

    factory_calls = []

    def fake_factory(
        pretrained,
        freeze_backbone,
        device,
    ):
        factory_calls.append({
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "device": device,
        })
        return TinyTestModel()

    monkeypatch.setattr(
        predict,
        "create_severity_model",
        fake_factory,
    )

    model, device, metadata = (
        predict.load_trained_model(
            checkpoint_path,
            device="cpu",
        )
    )

    assert isinstance(model, TinyTestModel)
    assert device.type == "cpu"
    assert metadata["checkpoint_epoch"] == 8
    assert metadata["checkpoint_best_macro_f1"] == 0.7261

    assert factory_calls == [{
        "pretrained": False,
        "freeze_backbone": True,
        "device": "cpu",
    }]

    assert model.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )


def test_prediction_with_test_model(
    tmp_path, monkeypatch
):
    image_path = make_image(tmp_path)
    checkpoint_path = make_checkpoint(tmp_path)

    monkeypatch.setattr(
        predict,
        "create_severity_model",
        lambda pretrained, freeze_backbone, device:
            TinyTestModel(),
    )

    model, device, metadata = (
        predict.load_trained_model(
            checkpoint_path,
            device="cpu",
        )
    )

    result = predict.predict_image(
        image_path=image_path,
        model=model,
        device=device,
        metadata=metadata,
    )

    assert result["image"]["original_size"] == {
        "width": 400,
        "height": 300,
    }

    assert result["prediction"]["predicted_class"] in (
        "Light",
        "Heavy",
    )

    assert result["prediction"][
        "restoration_executed"
    ] is False


def test_json_output(tmp_path):
    result = {
        "prediction": {
            "predicted_class": "Heavy",
            "confidence": 0.9,
        }
    }

    output = tmp_path / "reports" / "prediction.json"

    saved_path = predict.save_prediction_json(
        result,
        output,
    )

    assert saved_path == output
    assert output.exists()

    loaded = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert loaded == result
