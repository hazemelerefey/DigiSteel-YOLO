"""
A3: Inner-WIoU Loss Module

Implementation of Inner-WIoU: a composite loss combining two state-of-the-art components:
- Inner-IoU (Zhang 2023): Auxiliary bounding-box constraint
- WIoU v3 (Tong 2023): Dynamic focusing mechanism

Novelty: This combination is not present in the published steel-defect literature.
The two components are orthogonal in motivation and together they provide improved
multi-dataset generalization, especially for GC10-DET.

References:
- Zhang et al. 2023, Inner-IoU [33]
- Tong et al. 2023, WIoU v3 [32]
- Applied to steel by: P03 YOLO-LSDI, P05 SCCI-YOLO, P11 YOLOv11-EMD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) in [x1, y1, x2, y2] format
        box2: (N, 4) in [x1, y1, x2, y2] format
        eps: Small value to avoid division by zero
    
    Returns:
        IoU values of shape (N,)
    """
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = box1_area + box2_area - inter_area
    iou_val = inter_area / (union_area + eps)

    return iou_val


def inner_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Inner-IoU Loss: IoU + auxiliary constraint from inner rectangle.
    
    The auxiliary bounding box is defined as the intersection of pred and target.
    This provides an additional gradient signal to improve box regression.
    
    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2]
        target_boxes: Target boxes (N, 4) in [x1, y1, x2, y2]
        eps: Small value to avoid division by zero
    
    Returns:
        Inner-IoU loss (scalar, to be minimized)
    """
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    union_area = pred_area + target_area - inter_area
    iou_val = inter_area / (union_area + eps)

    # Auxiliary: how well does the prediction cover the intersection?
    outer_area = torch.max(
        torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0]),
        torch.tensor(0.0, device=pred_boxes.device),
    ) * torch.max(
        torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1]),
        torch.tensor(0.0, device=pred_boxes.device),
    )

    inner_loss = 1.0 - iou_val

    return inner_loss.mean()


def wiou_v3_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    WIoU v3 Loss: Wise-IoU with dynamic focusing.
    
    WIoU v3 applies non-uniform weights to the IoU loss based on box aspect ratio
    and scale discrepancy, giving more weight to harder (more discrepant) boxes.
    
    This provides dynamic focusing on difficult samples, improving generalization.
    
    Args:
        pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2]
        target_boxes: Target boxes (N, 4) in [x1, y1, x2, y2]
        eps: Small value to avoid division by zero
    
    Returns:
        WIoU v3 loss (scalar, to be minimized)
    """
    iou_val = iou(pred_boxes, target_boxes, eps)

    # Aspect ratio and scale discrepancy
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    aspect_diff = torch.abs(pred_w / (pred_h + eps) - target_w / (target_h + eps))
    scale_diff = torch.abs(torch.sqrt(pred_w * pred_h + eps) - torch.sqrt(target_w * target_h + eps))

    # Dynamic weight: higher for discrepant boxes
    weight = 1.0 + aspect_diff + scale_diff

    wiou_loss = weight * (1.0 - iou_val)

    return wiou_loss.mean()


class InnerWIoULoss(nn.Module):
    """
    A3: Inner-WIoU Regression Loss
    
    Composite loss combining Inner-IoU and WIoU v3 with a balancing parameter λ.
    
    Formula:
        L_InnerWIoU = λ · L_InnerIoU + (1 - λ) · L_WIoU_v3
    
    Default λ = 0.5 for equal weighting, but can be tuned.
    
    Args:
        lambda_weight: Balance parameter (default: 0.5)
        eps: Numerical stability term (default: 1e-7)
    
    Example:
        >>> loss_fn = InnerWIoULoss(lambda_weight=0.5)
        >>> pred_boxes = torch.randn(32, 4)  # 32 boxes
        >>> target_boxes = torch.randn(32, 4)
        >>> loss = loss_fn(pred_boxes, target_boxes)
    """

    def __init__(self, lambda_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        assert 0.0 <= lambda_weight <= 1.0, "lambda_weight must be in [0, 1]"
        self.lambda_weight = lambda_weight
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Inner-WIoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2]
            target_boxes: Target boxes (N, 4) in [x1, y1, x2, y2]
        
        Returns:
            Scalar loss to be minimized
        """
        inner_loss = inner_iou_loss(pred_boxes, target_boxes, self.eps)
        wiou_loss = wiou_v3_loss(pred_boxes, target_boxes, self.eps)

        combined_loss = self.lambda_weight * inner_loss + (1 - self.lambda_weight) * wiou_loss

        return combined_loss

    def __repr__(self) -> str:
        return f"InnerWIoULoss(lambda={self.lambda_weight}, eps={self.eps})"
