
"""
Binary rain severity classification using MobileNetV3-Small.

Classes:
    0 - Light
    1 - Heavy

Supports ImageNet transfer learning, backbone freezing,
logit prediction and class probability conversion.
"""

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


CLASS_NAMES = ("Light", "Heavy")
NUM_CLASSES = len(CLASS_NAMES)


class RainSeverityClassifier(nn.Module):
    """MobileNetV3-Small for binary rain classification."""

    def __init__(
        self,
        pretrained: bool = False,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        self.model = mobilenet_v3_small(weights=weights)

        # Replace ImageNet's 1000-class classification layer
        # with our two-class rain severity output.
        input_features = self.model.classifier[3].in_features

        self.model.classifier[3] = nn.Linear(
            input_features,
            NUM_CLASSES,
        )

        self.backbone_frozen = False

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze convolutional feature extraction layers."""

        for parameter in self.model.features.parameters():
            parameter.requires_grad = False

        self.backbone_frozen = True

    def unfreeze_backbone(self):
        """Enable backbone fine-tuning."""

        for parameter in self.model.features.parameters():
            parameter.requires_grad = True

        self.backbone_frozen = False

    def train(self, mode: bool = True):
        """
        Keep frozen backbone BatchNorm statistics fixed
        during classifier-head training.
        """

        super().train(mode)

        if self.backbone_frozen:
            self.model.features.eval()

        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw classification logits."""

        return self.model(images)

    @torch.inference_mode()
    def predict_proba(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return Light and Heavy probabilities."""

        was_training = self.training

        try:
            self.eval()

            logits = self.forward(images)

            return torch.softmax(logits, dim=1)

        finally:
            self.train(was_training)

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor,
    ) -> list[dict]:
        """Return class labels and predicted probabilities."""

        probabilities = self.predict_proba(images)

        results = []

        for probability in probabilities:
            predicted_index = int(
                torch.argmax(probability).item()
            )

            results.append({
                "predicted_class": CLASS_NAMES[
                    predicted_index
                ],
                "predicted_index": predicted_index,
                "confidence": float(
                    probability[predicted_index].item()
                ),
                "probabilities": {
                    name: float(probability[index].item())
                    for index, name in enumerate(CLASS_NAMES)
                },
            })

        return results


def create_severity_model(
    pretrained: bool = False,
    freeze_backbone: bool = True,
    device: str = "cpu",
) -> RainSeverityClassifier:
    """Create and place a binary severity classifier."""

    model = RainSeverityClassifier(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    return model.to(torch.device(device))
