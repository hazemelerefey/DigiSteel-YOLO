"""
DigiSteel-YOLO: Robust Real-Time Steel Surface Defect Detection

A lightweight, production-ready YOLO-based detector for industrial steel surface defects.
Two architectural modifications (A2 GhostConv + A3 Inner-WIoU) with quantitative robustness validation.

Team: Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham
Supervisor: Dr. Tarek Ghoneimy
Program: Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics
"""

__version__ = "0.1.0"
__author__ = "DigiSteel-YOLO Team"

from digisteel.modules.ghost_conv import GhostConv
from digisteel.modules.inner_wiou import InnerWIoULoss

__all__ = [
    "GhostConv",
    "InnerWIoULoss",
]
