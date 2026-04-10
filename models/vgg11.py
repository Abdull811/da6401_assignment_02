"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Helper block: Conv + BN + ReLU
        def block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # VGG11 Structure
        self.conv1 = block(in_channels, 64)
        self.conv2 = block(64, 128)

        self.conv3 = block(128, 256)
        self.conv4 = block(256, 256)

        self.conv5 = block(256, 512)
        self.conv6 = block(512, 512)

        self.conv7 = block(512, 512)
        self.conv8 = block(512, 512)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)

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
        features["enc1"] = x
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        features["enc2"] = x
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        features["enc3"] = x
        x = self.pool3(x)
        
        # Block 4
        x = self.conv5(x)
        x = self.conv6(x)
        features["enc4"] = x
        x = self.pool4(x)
        
        # Block 5
        x = self.conv7(x)
        x = self.conv8(x)
        features["enc5"] = x
        x = self.pool5(x)
        # Return
        if return_features:
            return x, features

        return x    
    
