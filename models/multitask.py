import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3,
                 classifier_path="classifier.pth",
                 localizer_path="localizer.pth",
                 unet_path="unet.pth"):
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
        #import gdown
        #gdown.download(id="1mRmo7uThs3pXZR9VTDMsvqBHBeaQ4uFE", output=classifier_path, quiet=False)
        #gdown.download(id="1J_OBIz7Ho-74hxEqXgTNeQWWojpaP29b", output=localizer_path, quiet=False)
        #gdown.download(id="1--Gm_jEMJC0d7f98jg6tE0WphV28VByh", output=unet_path, quiet=False)
                      
        # Shared Backbone
        #self.encoder = VGG11Encoder(in_channels)

        # Heads
        self.classifier = VGG11Classifier(num_breeds)
        self.localizer = VGG11Localizer()
        self.segmenter = VGG11UNet(seg_classes)

        # Load weights
        self._load_weights(self.classifier, classifier_path)
        self._load_weights(self.localizer, localizer_path)
        self._load_weights(self.segmenter, unet_path)

    def _resolve_checkpoint_path(self, path):
        candidates = [
            path,
            os.path.join(PROJECT_ROOT, path),
            os.path.join(PROJECT_ROOT, "checkpoints", os.path.basename(path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return path

    def _load_weights(self, module, path):
        resolved_path = self._resolve_checkpoint_path(path)
        try:
            ckpt = torch.load(resolved_path, map_location="cpu")
            if "state_dict" in ckpt:
                module.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                module.load_state_dict(ckpt, strict=False)
        except Exception as exc:
            print(f"Warning: could not load {resolved_path}: {exc}")
       
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
        
        return {
        "classification": self.classifier(x),  
        "localization": self.localizer(x),
        "segmentation": self.segmenter(x),
    }
    
