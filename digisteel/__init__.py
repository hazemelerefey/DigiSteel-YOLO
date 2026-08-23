"""
DAFEGate-YOLO: Dual-Branch Defect-Aware Feature Enhancement for Real-Time
Surface Defect Detection in Hot-Rolled Flat Steel Production.

Product: DigiSteel
Novel Contribution: DAFEGate v4 — Dual-branch Sobel-initialized edge detection
    and local variance texture analysis with channel attention and additive
    residual highway for morphology-specialized steel defect feature extraction.

Dataset: NEU-DET (6-class, 1,800 images, 70/20/10 clean protocol)
Result:  mAP@0.5 = 81.98% (+2.63pp over optimized YOLOv11n baseline)

Team: Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat,
      Mahmoud Hisham, Mohamed Awni
Supervisor: Dr. Tarek Ghoneimy
Program: Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics
"""

__version__ = "4.0.0"
__author__ = "DigiSteel Team"

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
