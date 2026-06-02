"""
DigiSteel-YOLO v2: Comprehensive Robustness Study of Lightweight YOLO Detectors
for Steel Surface Defect Detection.

This package provides:
- WFCA: Wavelet Frequency Channel Attention (novel contribution)
- GhostConv: Lightweight convolution module (Han et al., CVPR 2020)
- EMA: Efficient Multi-scale Attention (Ouyang et al., ICASSP 2023)
- InnerWIoULoss: Composite IoU loss (Zhang 2023 + Tong 2023)
- Perturbation toolkit: Standardized image degradation for robustness testing
- Evaluation framework: Multi-metric robustness evaluation

Team: Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham
Supervisor: Dr. Tarek Ghoneimy
Program: Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics
"""

__version__ = "2.0.0"
__author__ = "DigiSteel-YOLO Team"

from digisteel.modules.ema import EMA
from digisteel.modules.ghost_conv import GhostConv, GhostModule
from digisteel.modules.inner_wiou import InnerWIoULoss, inner_iou_loss, inner_wiou_iou, wiou_v3_loss
from digisteel.modules.wfca import WFCA

__all__ = [
    "EMA",
    "GhostConv",
    "GhostModule",
    "InnerWIoULoss",
    "WFCA",
    "inner_iou_loss",
    "inner_wiou_iou",
    "wiou_v3_loss",
]
