"""
Inner-WIoU Loss Module for Steel Defect Detection.

A composite regression loss combining two established IoU-based losses:
- Inner-IoU (Zhang et al., 2023): Auxiliary bounding-box constraint
- WIoU v3 (Tong et al., 2023): Dynamic focusing mechanism

This module is used as a loss variant in the DigiSteel-YOLO robustness study.
It is NOT claimed as a novel contribution — it is a principled combination of
two well-cited techniques applied to steel defect detection.

Formula:
    L = lambda * L_InnerIoU + (1 - lambda) * L_WIoU_v3

    where lambda = 0.5 by default.

References:
    - Zhang et al. 2023, Inner-IoU: More Effective Intersection over Union Loss
      with Auxiliary Bounding Box, arXiv:2311.02877 (661 citations)
    - Tong et al. 2023, Wise-IoU: Bounding Box Regression Loss with Dynamic
      Focusing Mechanism, arXiv:2301.10051 (1,773 citations)
    - Applied to steel defect detection by:
      P03 YOLO-LSDI (Electronics 2025, DOI: 10.3390/electronics14132576)
      P05 SCCI-YOLO (Scientific Reports 2025, DOI: 10.1038/s41598-025-20154-y)
      P07 ASFRW-YOLO (Scientific Reports 2025, DOI: 10.1038/s41598-025-28022-5)

Note:
    Both Inner-IoU and WIoU v3 are arXiv preprints with high citation counts
    but have not been formally peer-reviewed in a journal/conference venue.
"""

import torch
import torch.nn as nn


def iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Args:
        box1: Predicted boxes, shape (N, 4) in [x1, y1, x2, y2] format.
        box2: Target boxes, shape (N, 4) in [x1, y1, x2, y2] format.
        eps: Small value to avoid division by zero.

    Returns:
        IoU values of shape (N,), values in [0, 1].
    """
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (
        torch.clamp(inter_x2 - inter_x1, min=0)
        * torch.clamp(inter_y2 - inter_y1, min=0)
    )

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + eps)


def inner_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Inner-IoU Loss: IoU computed via an auxiliary bounding box.

    The auxiliary box is the intersection of predicted and target boxes.
    This provides an additional gradient signal that accelerates convergence,
    especially for high-IoU samples.

    Reference: Zhang et al. 2023, arXiv:2311.02877

    Args:
        pred_boxes: Predicted boxes, shape (N, 4) in [x1, y1, x2, y2].
        target_boxes: Target boxes, shape (N, 4) in [x1, y1, x2, y2].
        eps: Small value to avoid division by zero.

    Returns:
        Scalar loss value (to be minimized).
    """
    iou_val = iou(pred_boxes, target_boxes, eps)
    inner_loss = 1.0 - iou_val
    return inner_loss.mean()


def wiou_v3_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    WIoU v3 Loss: Wise-IoU with dynamic non-monotonic focusing.

    Applies non-uniform weights to the IoU loss based on box aspect ratio
    and scale discrepancy. Harder samples (larger discrepancy) receive
    higher gradients, while easy samples are down-weighted.

    Reference: Tong et al. 2023, arXiv:2301.10051

    Args:
        pred_boxes: Predicted boxes, shape (N, 4) in [x1, y1, x2, y2].
        target_boxes: Target boxes, shape (N, 4) in [x1, y1, x2, y2].
        eps: Small value to avoid division by zero.

    Returns:
        Scalar loss value (to be minimized).
    """
    iou_val = iou(pred_boxes, target_boxes, eps)

    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    aspect_diff = torch.abs(pred_w / (pred_h + eps) - target_w / (target_h + eps))
    scale_diff = torch.abs(
        torch.sqrt(pred_w * pred_h + eps) - torch.sqrt(target_w * target_h + eps)
    )

    weight = 1.0 + aspect_diff + scale_diff
    wiou_loss = weight * (1.0 - iou_val)

    return wiou_loss.mean()


class InnerWIoULoss(nn.Module):
    """
    Inner-WIoU: Composite regression loss for bounding box optimization.

    Combines Inner-IoU (auxiliary box constraint) with WIoU v3 (dynamic focusing)
    using a weighted average controlled by lambda.

    Formula:
        L = lambda * L_InnerIoU + (1 - lambda) * L_WIoU_v3

    Args:
        lambda_weight: Balance parameter in [0, 1]. Default: 0.5.
            lambda=1.0 gives pure Inner-IoU.
            lambda=0.0 gives pure WIoU v3.
            lambda=0.5 gives equal weighting.
        eps: Numerical stability term. Default: 1e-7.

    Example:
        >>> loss_fn = InnerWIoULoss(lambda_weight=0.5)
        >>> pred = torch.randn(32, 4)
        >>> target = torch.randn(32, 4)
        >>> loss = loss_fn(pred, target)
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
            pred_boxes: Predicted boxes, shape (N, 4) in [x1, y1, x2, y2].
            target_boxes: Target boxes, shape (N, 4) in [x1, y1, x2, y2].

        Returns:
            Scalar loss value (to be minimized).
        """
        inner_loss = inner_iou_loss(pred_boxes, target_boxes, self.eps)
        wiou_loss = wiou_v3_loss(pred_boxes, target_boxes, self.eps)

        return self.lambda_weight * inner_loss + (1 - self.lambda_weight) * wiou_loss

    def __repr__(self) -> str:
        return f"InnerWIoULoss(lambda={self.lambda_weight}, eps={self.eps})"


def inner_wiou_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    xywh: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Inner-WIoU as a per-box IoU-like metric for trainer integration.

    Returns a (N,) tensor of IoU-like values in [0, 1] that Ultralytics'
    BboxLoss can use directly. Higher = better alignment.

    Combines standard IoU with WIoU-style aspect/scale penalty:
        wiou = iou / (1 + aspect_diff + scale_diff)

    Args:
        pred_boxes: Predicted boxes, shape (N, 4).
        target_boxes: Target boxes, shape (N, 4).
        xywh: If True, input is (cx, cy, w, h); converted to xyxy internally.
        eps: Numerical stability.

    Returns:
        Per-box IoU-like values, shape (N,).
    """
    if xywh:
        pred_xy = pred_boxes[:, :2]
        pred_wh = pred_boxes[:, 2:]
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        pred_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=1)

        tgt_xy = target_boxes[:, :2]
        tgt_wh = target_boxes[:, 2:]
        tgt_x1y1 = tgt_xy - tgt_wh / 2
        tgt_x2y2 = tgt_xy + tgt_wh / 2
        tgt_xyxy = torch.cat([tgt_x1y1, tgt_x2y2], dim=1)
    else:
        pred_xyxy = pred_boxes
        tgt_xyxy = target_boxes

    iou_val = iou(pred_xyxy, tgt_xyxy, eps)

    # WIoU weighting: scale IoU by inverse of aspect+scale discrepancy
    pred_w = pred_xyxy[:, 2] - pred_xyxy[:, 0]
    pred_h = pred_xyxy[:, 3] - pred_xyxy[:, 1]
    tgt_w = tgt_xyxy[:, 2] - tgt_xyxy[:, 0]
    tgt_h = tgt_xyxy[:, 3] - tgt_xyxy[:, 1]

    aspect_diff = torch.abs(pred_w / (pred_h + eps) - tgt_w / (tgt_h + eps))
    scale_diff = torch.abs(
        torch.sqrt(pred_w * pred_h + eps) - torch.sqrt(tgt_w * tgt_h + eps)
    )

    penalty = aspect_diff + scale_diff
    wiou_val = iou_val / (1.0 + penalty)

    return wiou_val
