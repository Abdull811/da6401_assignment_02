"""Unified multi-task model
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn

np.random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth"
    ):
        super().__init__()

        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        if os.path.exists(classifier_path):
            classifier_ckpt = torch.load(classifier_path, map_location="cpu")
            if isinstance(classifier_ckpt, dict) and "state_dict" in classifier_ckpt:
                self.classifier.load_state_dict(classifier_ckpt["state_dict"])
            else:
                self.classifier.load_state_dict(classifier_ckpt)

        if os.path.exists(localizer_path):
            localizer_ckpt = torch.load(localizer_path, map_location="cpu")
            if isinstance(localizer_ckpt, dict) and "state_dict" in localizer_ckpt:
                self.localizer.load_state_dict(localizer_ckpt["state_dict"])
            else:
                self.localizer.load_state_dict(localizer_ckpt)

        if os.path.exists(unet_path):
            unet_ckpt = torch.load(unet_path, map_location="cpu")
            if isinstance(unet_ckpt, dict) and "state_dict" in unet_ckpt:
                self.segmenter.load_state_dict(unet_ckpt["state_dict"])
            else:
                self.segmenter.load_state_dict(unet_ckpt)

    def forward(self, x: torch.Tensor):
        return {
            "classification": self.classifier(x),
            "localization": self.localizer(x),
            "segmentation": self.segmenter(x)
        }
