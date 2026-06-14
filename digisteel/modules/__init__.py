"""
DigiSteel-YOLO modules package.

Provides:
- DAFE: Defect-Aware Feature Enhancement (novel — for flat steel defects)
- GhostConv: Lightweight convolution module (Han et al., CVPR 2020)
- WFCA: Wavelet Frequency Channel Attention (novel)
- EMA: Efficient Multi-scale Attention (Ouyang et al., ICASSP 2023)
- InnerWIoULoss: Composite IoU loss (Zhang 2023 + Tong 2023)
"""

from digisteel.modules.dafe import DAFE, DAFEEdgeOnly
from digisteel.modules.ema import EMA
from digisteel.modules.ghost_conv import GhostConv, GhostModule
from digisteel.modules.inner_wiou import InnerWIoULoss, inner_iou_loss, inner_wiou_iou, wiou_v3_loss
from digisteel.modules.wfca import WFCA

__all__ = [
    "DAFE",
    "EMA",
    "GhostConv",
    "GhostModule",
    "InnerWIoULoss",
    "WFCA",
    "inner_iou_loss",
    "inner_wiou_iou",
    "wiou_v3_loss",
]
