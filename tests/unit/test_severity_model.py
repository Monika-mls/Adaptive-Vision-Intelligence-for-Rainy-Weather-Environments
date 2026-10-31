
"""
Unit tests for MobileNetV3-Small binary severity classifier.

Uses random model initialization.
No pretrained weights are downloaded.
"""

import torch
import torch.nn as nn

from vision.severity.model import (
    CLASS_NAMES,
    NUM_CLASSES,
    RainSeverityClassifier,
    create_severity_model,
)


def test_binary_classes():
    assert CLASS_NAMES == ("Light", "Heavy")
    assert NUM_CLASSES == 2


def test_model_creation():
    model = create_severity_model(
        pretrained=False,
        freeze_backbone=True,
    )

    assert isinstance(model, RainSeverityClassifier)


def test_classification_head():
    model = RainSeverityClassifier(
        pretrained=False
    )

    assert isinstance(
        model.model.classifier[3],
        nn.Linear,
    )

    assert model.model.classifier[3].out_features == 2


def test_forward_output_shape():
    model = RainSeverityClassifier(
        pretrained=False
    )

    model.eval()

    images = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all().item()


def test_backbone_frozen():
    model = RainSeverityClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.model.features.parameters()
    )


def test_backbone_unfreeze():
    model = RainSeverityClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    model.unfreeze_backbone()

    assert all(
        parameter.requires_grad
        for parameter in model.model.features.parameters()
    )


def test_classifier_trainable():
    model = RainSeverityClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.model.classifier.parameters()
    )


def test_frozen_backbone_stays_eval():
    model = RainSeverityClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    model.train()

    assert model.model.features.training is False
    assert model.model.classifier.training is True


def test_prediction_probabilities():
    model = RainSeverityClassifier(
        pretrained=False
    )

    images = torch.randn(1, 3, 224, 224)

    probabilities = model.predict_proba(images)

    assert probabilities.shape == (1, 2)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(1),
        atol=1e-5,
    )


def test_prediction_format():
    model = RainSeverityClassifier(
        pretrained=False
    )

    images = torch.randn(1, 3, 224, 224)

    results = model.predict(images)

    assert len(results) == 1

    result = results[0]

    assert result["predicted_class"] in CLASS_NAMES
    assert 0 <= result["confidence"] <= 1

    assert set(result["probabilities"]) == {
        "Light",
        "Heavy",
    }


def test_cross_entropy_compatibility():
    model = RainSeverityClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    model.train()

    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    logits = model(images)

    loss = nn.CrossEntropyLoss()(logits, labels)

    assert torch.isfinite(loss).item()
