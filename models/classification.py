"""Classification components
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

from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11 + ClassificationHead."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        
        super().__init__()
        # Encoder: Extract features from image using VGG11
        self.encoder = VGG11Encoder(in_channels, use_batchnorm=use_batchnorm)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 768),
            *([nn.BatchNorm1d(768)] if use_batchnorm else []),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(768, 256),
            *([nn.BatchNorm1d(256)] if use_batchnorm else []),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """

        x = self.encoder(x)
        avg_feat = torch.flatten(self.avg_pool(x), 1)
        max_feat = torch.flatten(self.max_pool(x), 1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        return self.classifier(feat)

class VGG11(VGG11Encoder):
    pass
