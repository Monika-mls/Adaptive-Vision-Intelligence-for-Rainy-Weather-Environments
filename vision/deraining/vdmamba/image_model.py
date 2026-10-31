"""Experimental VDMamba-inspired full image-to-image prototype.

NOT official VDMamba, not architecturally verified, not pretrained.
Designed for small-patch shape/gradient feasibility tests only.
"""
import torch
from torch import nn

from vision.deraining.vdmamba.wavelet import HaarDWT, HaarIDWT
from vision.deraining.vdmamba.vdsm import ExperimentalVDSM
from vision.deraining.vdmamba.rmu import ExperimentalRMU


class ExperimentalFEM(nn.Module):
    """Haar decomposition -> learnable feature extraction -> experimental RMU.

    This is a plausible implementation choice, not an exact paper specification.
    """
    def __init__(self, channels=8, state_size=4):
        super().__init__()
        if channels < 1 or state_size < 1:
            raise ValueError("channels and state_size must be positive")
        self.channels = channels
        self.dwt = HaarDWT()
        self.project = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.refine = ExperimentalRMU(channels, state_size)

    def forward(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a tensor")
        if image.ndim != 4 or image.shape[1] != 3 or any(d < 1 for d in image.shape):
            raise ValueError("expected RGB image [B,3,H,W]")
        if not image.is_floating_point():
            raise TypeError("image must be floating point")
        ll, lh, hl, hh, original_size = self.dwt(image)
        features = self.refine(self.project(torch.cat((ll, lh, hl, hh), 1)))
        return features, original_size


class ExperimentalReconstruction(nn.Module):
    """Feature processing -> four wavelet-band corrections -> inverse Haar.

    The identity skip is deliberately added in image space. Prediction is not
    clamped during training because clamping suppresses useful gradients.
    """
    def __init__(self, channels=8):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 12, 3, padding=1),
        )
        # Small output correction initially; final bias begins at zero.
        nn.init.zeros_(self.refine[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.idwt = HaarIDWT()

    def forward(self, background_features, original_image):
        if not isinstance(background_features, torch.Tensor) or not isinstance(original_image, torch.Tensor):
            raise TypeError("inputs must be tensors")
        if background_features.ndim != 4 or background_features.shape[1] != self.channels:
            raise ValueError("invalid background feature shape")
        if original_image.ndim != 4 or original_image.shape[1] != 3:
            raise ValueError("expected RGB [B,3,H,W]")
        if background_features.shape[0] != original_image.shape[0]:
            raise ValueError("batch sizes must match")
        expected_h = (original_image.shape[2] + 1) // 2
        expected_w = (original_image.shape[3] + 1) // 2
        if background_features.shape[-2:] != (expected_h, expected_w):
            raise ValueError("feature resolution does not match wavelet image size")
        bands = self.refine(background_features).chunk(4, dim=1)
        correction = self.idwt(*bands, original_image.shape[-2:])
        return original_image + self.scale * correction


class ExperimentalVDMambaImageModel(nn.Module):
    """RGB image -> experimental FEM -> experimental VDSM -> experimental RM.

    Outputs are untrained and MUST NOT be presented as derained images.
    """
    def __init__(self, channels=8, state_size=4):
        super().__init__()
        self.fem = ExperimentalFEM(channels, state_size)
        self.vdsm = ExperimentalVDSM(channels, state_size)
        self.reconstruction = ExperimentalReconstruction(channels)

    def forward(self, rainy_image, return_details=False):
        features, _ = self.fem(rainy_image)
        decomposition = self.vdsm(features)
        prediction = self.reconstruction(decomposition["background"], rainy_image)
        if return_details:
            return {"prediction": prediction, **decomposition}
        return prediction
