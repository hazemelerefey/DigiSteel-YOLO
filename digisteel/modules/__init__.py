"""
DigiSteel-YOLO modules package.

Provides:
- GhostConv: Lightweight convolution module (Han et al., CVPR 2020)
- InnerWIoULoss: Composite IoU loss (Zhang 2023 + Tong 2023)
"""

from digisteel.modules.ghost_conv import GhostConv, GhostModule
from digisteel.modules.inner_wiou import InnerWIoULoss, inner_iou_loss, wiou_v3_loss

__all__ = [
    "GhostConv",
    "GhostModule",
    "InnerWIoULoss",
    "inner_iou_loss",
    "wiou_v3_loss",
]
