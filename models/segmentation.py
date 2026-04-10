"""Segmentation model
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

from models.vgg11 import VGG11Encoder


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        # Extract high-level features from image
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)

        # Decoder blocks
        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.up1 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv1 = conv_block(1024, 512)   # 512 + 512

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = conv_block(768, 256)    # 256 + 512

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = conv_block(384, 128)    # 128 + 256

        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = conv_block(192, 64)     # 64 + 128

        self.up5 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.conv5 = conv_block(128, 64)     # 64 + 64

        self.final = nn.Conv2d(64, num_classes, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        
        bottleneck, feats = self.encoder(x, return_features=True)

        # 7 → 14
        x = self.up1(bottleneck)
        x = torch.cat([x, feats["enc5"]], dim=1)
        x = self.conv1(x) 

        # 14 → 28
        x = self.up2(x)
        x = torch.cat([x, feats["enc4"]], dim=1)
        x = self.conv2(x)

        # 28 → 56
        x = self.up3(x)
        x = torch.cat([x, feats["enc3"]], dim=1)
        x = self.conv3(x)

        # 56 → 112
        x = self.up4(x)
        x = torch.cat([x, feats["enc2"]], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = torch.cat([x, feats["enc1"]], dim=1)
        x = self.conv5(x)

        return self.final(x)
        
