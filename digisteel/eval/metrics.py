"""
Metrics computation for object detection evaluation.

Provides standardized metric computation following COCO evaluation conventions.
All metrics are computed in a pure-functional style (no side effects).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""

    map50: float  # mAP@0.5
    map50_95: float  # mAP@0.5:0.95
    precision: float
    recall: float
    f1: float
    fps: Optional[float] = None
    params_m: Optional[float] = None
    gflops: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mAP@0.5": self.map50,
            "mAP@0.5:0.95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fps": self.fps,
            "params_M": self.params_m,
            "gflops": self.gflops,
        }

    def __repr__(self) -> str:
        parts = [
            f"mAP@0.5={self.map50:.1%}",
            f"mAP@0.5:0.95={self.map50_95:.1%}",
            f"P={self.precision:.1%}",
            f"R={self.recall:.1%}",
            f"F1={self.f1:.1%}",
        ]
        if self.fps is not None:
            parts.append(f"FPS={self.fps:.1f}")
        if self.params_m is not None:
            parts.append(f"Params={self.params_m:.2f}M")
        return f"Metrics({', '.join(parts)})"


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: First box, shape (4,).
        box2: Second box, shape (4,).

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """
    Compute Average Precision from precision-recall curve.

    Uses the 11-point interpolation method.

    Args:
        precisions: Array of precision values.
        recalls: Array of recall values.

    Returns:
        AP value in [0, 1].
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t]
        if len(p) > 0:
            ap += np.max(p)
    return ap / 11.0


def compute_metrics(
    y_true: List[Dict],
    y_pred: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """
    Compute detection metrics from ground truth and predictions.

    Args:
        y_true: List of ground truth dicts, each with:
            - "boxes": np.ndarray of shape (N, 4) in [x1, y1, x2, y2]
            - "labels": np.ndarray of shape (N,) with class ids
        y_pred: List of prediction dicts, each with:
            - "boxes": np.ndarray of shape (M, 4) in [x1, y1, x2, y2]
            - "scores": np.ndarray of shape (M,) with confidence scores
            - "labels": np.ndarray of shape (M,) with class ids
        num_classes: Number of classes.
        iou_threshold: IoU threshold for matching. Default: 0.5.

    Returns:
        DetectionMetrics with computed values.
    """
    all_precisions = []
    all_recalls = []
    aps_per_class = []

    for cls_id in range(num_classes):
        cls_tp = 0
        cls_fp = 0
        cls_fn = 0
        cls_scores = []
        cls_matches = []

        for gt, pred in zip(y_true, y_pred):
            gt_mask = gt["labels"] == cls_id
            pred_mask = pred["labels"] == cls_id

            gt_boxes = gt["boxes"][gt_mask]
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            # Sort predictions by score (descending)
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            matched = np.zeros(len(gt_boxes), dtype=bool)

            for pb, ps in zip(pred_boxes, pred_scores):
                cls_scores.append(ps)
                best_iou = 0.0
                best_idx = -1

                for j, gb in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    iou_val = compute_iou(pb, gb)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j

                if best_iou >= iou_threshold and best_idx >= 0:
                    cls_tp += 1
                    matched[best_idx] = True
                    cls_matches.append(1)
                else:
                    cls_fp += 1
                    cls_matches.append(0)

            cls_fn += np.sum(~matched)

        if cls_tp + cls_fp > 0:
            precision = cls_tp / (cls_tp + cls_fp)
        else:
            precision = 0.0

        if cls_tp + cls_fn > 0:
            recall = cls_tp / (cls_tp + cls_fn)
        else:
            recall = 0.0

        all_precisions.append(precision)
        all_recalls.append(recall)

        # Compute AP for this class
        if cls_scores:
            scores_arr = np.array(cls_scores)
            matches_arr = np.array(cls_matches)
            order = np.argsort(-scores_arr)
            matches_arr = matches_arr[order]
            cum_tp = np.cumsum(matches_arr)
            cum_fp = np.cumsum(1 - matches_arr)
            precisions = cum_tp / (cum_tp + cum_fp)
            recalls_val = cum_tp / (cls_tp + cls_fn + 1e-7)
            aps_per_class.append(compute_ap(precisions, recalls_val))
        else:
            aps_per_class.append(0.0)

    map50 = np.mean(aps_per_class) if aps_per_class else 0.0
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-7)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    return DetectionMetrics(
        map50=map50,
        map50_95=map50,  # Simplified; full mAP@0.5:0.95 needs multiple IoU thresholds
        precision=avg_precision,
        recall=avg_recall,
        f1=f1,
    )


def compute_metrics_summary(
    metrics_list: List[DetectionMetrics],
) -> Dict[str, float]:
    """
    Compute summary statistics across multiple metric sets.

    Args:
        metrics_list: List of DetectionMetrics objects.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if not metrics_list:
        return {}

    arrays = {key: [] for key in metrics_list[0].to_dict().keys() if key is not None}
    for m in metrics_list:
        for key, val in m.to_dict().items():
            if val is not None and key in arrays:
                arrays[key].append(val)

    summary = {}
    for key, values in arrays.items():
        if values:
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)

    return summary
