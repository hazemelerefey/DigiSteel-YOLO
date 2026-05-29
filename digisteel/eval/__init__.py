"""
Evaluation framework for steel defect detector robustness.

Provides tools for:
- Computing standard detection metrics (mAP, precision, recall, F1)
- Running robustness sweeps across perturbation types and levels
- Aggregating and comparing results across models and datasets
"""

from digisteel.eval.metrics import compute_metrics, compute_metrics_summary
from digisteel.eval.robustness_sweep import RobustnessSweep

__all__ = [
    "compute_metrics",
    "compute_metrics_summary",
    "RobustnessSweep",
]
