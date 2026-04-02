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
from models.vgg11 import VGG11


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11 + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        
        super().__init__()
        # Encoder: Extract features from image using VGG11
        self.encoder = VGG11()

        # Classifier head: Convert extracted features into class predictions 
        self.classifier = nn.Sequential( nn.Flatten(), # Convert C,H,W > 1D vector
                                        nn.Linear(512*7*7, 4096), # fully connected layer
                                        nn.ReLU(), CustomDropout(dropout_p), # activation + dropout for regularization
                                        nn.Linear(4096, num_classes)) # final output layer (logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """

        x = self.encoder(x)
        return self.classifier(x)
    