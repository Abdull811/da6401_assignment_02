"""Loss package exports for Assignment-2 skeleton."""

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from losses.iou_loss import IoULoss

__all__ = ["IoULoss"]
