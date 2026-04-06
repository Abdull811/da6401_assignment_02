import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        # Shared Backbone
        #self.encoder = VGG11Encoder(in_channels)

        # Head
        self.classifier_head = VGG11Classifier(num_breeds).classifier
        self.localizer_head = VGG11Localizer().head
        self.segmenter = VGG11UNet(seg_classes)

        # Load pretrained weights
        self._load_weights(self.classifier_head, classifier_path)
        self._load_weights(self.localizer_head, localizer_path)
        self._load_weights(self.segmenter, unet_path)

    def _load_weights(self, module, path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            if "state_dict" in ckpt:
                module.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                module.load_state_dict(ckpt, strict=False)
        except:
            print(f"Warning: could not load {path}")    
       
    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        
        # Shared feature extraction
        features = self.encoder(x)

        # Classification
        cls_out = self.classifier_head(features)

        # Localization
        loc_out = self.localizer_head(features)

        # Segmentation
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
    
