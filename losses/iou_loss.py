"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
       IoU loss range [0,1], where 0 means no overlap and 1 means perfect overlap.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        
        assert reduction in ["none", "mean", "sum"]
        self.eps = eps
        self.reduction = reduction


    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
       
         # Convert center → corner format
        def convert(box):
            xc, yc, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            return x1, y1, x2, y2

        px1, py1, px2, py2 = convert(pred_boxes)
        tx1, ty1, tx2, ty2 = convert(target_boxes)

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

        # Areas (safe)
        area_p = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
        area_t = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)

        union = area_p + area_t - inter + self.eps

        iou = inter / (union + self.eps)

        loss = 1 - iou

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss   
