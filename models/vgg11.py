"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Helper block: Conv + BN + ReLU
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # VGG11 Structure
        self.conv1 = block(in_channels, 64)
        self.conv2 = block(64, 128)

        self.conv3 = block(128, 256)
        self.conv4 = block(256, 256)

        self.conv5 = block(256, 512)
        self.conv6 = block(512, 512)

        self.conv7 = block(512, 512)
        self.conv8 = block(512, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """

        features: Dict[str, torch.Tensor] = {}

        # Block 1 
        x = self.conv1(x)
        features["enc1"] = x      # High resolution
        x = self.pool(x)

        # Block 2 
        x = self.conv2(x)
        features["enc2"] = x
        x = self.pool(x)

        # Block 3 
        x = self.conv3(x)
        x = self.conv4(x)
        features["enc3"] = x
        x = self.pool(x)

        # Block 4 
        x = self.conv5(x)
        x = self.conv6(x)
        features["enc4"] = x
        x = self.pool(x)

        # Block 5 
        x = self.conv7(x)
        x = self.conv8(x)
        features["enc5"] = x      # Lowest resolution
        #x = self.pool(x)

        # Return
        if return_features:
            return x, features

        return x    
    
