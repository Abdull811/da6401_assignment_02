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
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be one of ['mean', 'sum', 'none']")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
       
        px, py = pred_boxes[:, 0], pred_boxes[:, 1]
        pw = torch.clamp(pred_boxes[:, 2], min=self.eps)
        ph = torch.clamp(pred_boxes[:, 3], min=self.eps)

        tx, ty = target_boxes[:, 0], target_boxes[:, 1]
        tw = torch.clamp(target_boxes[:, 2], min=self.eps)
        th = torch.clamp(target_boxes[:, 3], min=self.eps)

        pred_x1 = px - pw / 2.0
        pred_y1 = py - ph / 2.0
        pred_x2 = px + pw / 2.0
        pred_y2 = py + ph / 2.0

        target_x1 = tx - tw / 2.0
        target_y1 = ty - th / 2.0
        target_x2 = tx + tw / 2.0
        target_y2 = ty + th / 2.0

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        pred_area = pw * ph
        target_area = tw * th
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area
        iou = torch.clamp(iou, 0.0, 1.0)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
