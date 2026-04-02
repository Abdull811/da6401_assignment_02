"""Localization modules
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

from models.vgg11 import VGG11


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        
        super().__init__()
        # Extract features from image
        self.encoder = VGG11()
        
        # Convert features into bounding box coordinates
        self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(),
        nn.Linear(1024, 4))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        x = self.encoder(x)
        x = self.head(x)

        centers = torch.sigmoid(x[:, :2]) * 224.0
        sizes = torch.sigmoid(x[:, 2:]) * 224.0

        return torch.cat([centers, sizes], dim=1)
    