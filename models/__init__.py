"""Model package exports for Assignment-2 skeleton.

Import from this package in training/inference scripts to keep paths stable.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.layers import CustomDropout
from models.localization import VGG11Localizer
from models.classification import VGG11Classifier
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11Encoder
from models.multitask import MultiTaskPerceptionModel

__all__ = [
    "CustomDropout",
    "VGG11Classifier",
    "VGG11Encoder",
    "VGG11Localizer",
    "VGG11UNet",
    "MultiTaskPerceptionModel",
]
