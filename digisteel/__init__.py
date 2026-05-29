"""
DigiSteel-YOLO: Comprehensive Robustness Study of Lightweight YOLO Detectors
for Steel Surface Defect Detection.

This package provides:
- GhostConv: Lightweight convolution module (Han et al., CVPR 2020)
- InnerWIoULoss: Composite IoU loss (Zhang 2023 + Tong 2023)
- Perturbation toolkit: Standardized image degradation for robustness testing
- Evaluation framework: Multi-metric robustness evaluation

Team: Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham
Supervisor: Dr. Tarek Ghoneimy
Program: Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics
"""

__version__ = "0.2.0"
__author__ = "DigiSteel-YOLO Team"

from digisteel.modules.ghost_conv import GhostConv, GhostModule
from digisteel.modules.inner_wiou import InnerWIoULoss, inner_iou_loss, wiou_v3_loss

__all__ = [
    "GhostConv",
    "GhostModule",
    "InnerWIoULoss",
    "inner_iou_loss",
    "wiou_v3_loss",
]
