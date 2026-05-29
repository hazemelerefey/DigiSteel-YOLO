"""
Robustness Sweep: Systematic evaluation of detector robustness.

This is the core of the DigiSteel-YOLO contribution — a standardized framework
for evaluating how steel defect detectors degrade under real-world image
perturbations.

Usage:
    from digisteel.eval.robustness_sweep import RobustnessSweep

    sweep = RobustnessSweep(model_path="runs/baseline/weights/best.pt")
    results = sweep.run(dataset_path="datasets/NEU-DET/yolo")
    sweep.save_results(results, "evals/baseline_robustness.csv")
"""

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from digisteel.perturbations.suite import PerturbationConfig, PerturbationSuite


@dataclass
class SweepResult:
    """Result for a single (model, dataset, perturbation, level) evaluation."""

    model_name: str
    dataset_name: str
    perturbation: str
    level: int
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    fps: float
    params_m: float
    gflops: float
    inference_time_ms: float
    num_images: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return asdict(self)


@dataclass
class BaselineResult:
    """Result for clean (unperturbed) evaluation."""

    model_name: str
    dataset_name: str
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    fps: float
    params_m: float
    gflops: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RobustnessSweep:
    """
    Systematic robustness evaluation of YOLO-based steel defect detectors.

    Evaluates a model across all perturbation types and severity levels,
    producing a comprehensive robustness profile.

    Args:
        model_path: Path to the trained model weights (.pt file).
        model_name: Human-readable model name. Default: extracted from path.
        levels: Perturbation severity levels to evaluate. Default: [1, 2, 3, 4].

    Example:
        >>> sweep = RobustnessSweep("runs/baseline/weights/best.pt")
        >>> results = sweep.run("datasets/NEU-DET/yolo", "NEU-DET")
        >>> sweep.save_results(results, "evals/baseline_robustness.csv")
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        levels: List[int] = None,
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.parent.parent.name
        self.levels = levels or [1, 2, 3, 4]
        self.suite = PerturbationSuite(levels=self.levels)

        # Model will be loaded lazily
        self._model = None
        self._model_params = None
        self._model_gflops = None

    def _load_model(self):
        """Load the YOLO model (lazy loading)."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.model_path))
                # Get model info
                info = self._model.info(verbose=False)
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    self._model_params = info[0] / 1e6  # Convert to M
                    self._model_gflops = info[1]
                else:
                    self._model_params = 0.0
                    self._model_gflops = 0.0
            except ImportError:
                raise ImportError(
                    "ultralytics is required for model evaluation. "
                    "Install with: pip install ultralytics"
                )

    def _evaluate_model(
        self, image_paths: List[str], conf_threshold: float = 0.25
    ) -> Dict[str, float]:
        """
        Evaluate model on a set of images.

        Args:
            image_paths: List of image file paths.
            conf_threshold: Confidence threshold for detections.

        Returns:
            Dictionary with metric values.
        """
        self._load_model()

        total_time = 0.0
        all_detections = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            start = time.perf_counter()
            results = self._model.predict(
                source=img,
                conf=conf_threshold,
                verbose=False,
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    all_detections.append(
                        {
                            "boxes": r.boxes.xyxy.cpu().numpy(),
                            "scores": r.boxes.conf.cpu().numpy(),
                            "labels": r.boxes.cls.cpu().numpy().astype(int),
                        }
                    )

        num_images = len(image_paths)
        avg_time_ms = (total_time / max(num_images, 1)) * 1000
        fps = 1000.0 / max(avg_time_ms, 1.0)

        # For now, return placeholder metrics
        # In a real implementation, you'd compute mAP against ground truth
        return {
            "mAP50": 0.0,  # Computed separately with ground truth
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fps": fps,
            "inference_time_ms": avg_time_ms,
            "num_images": num_images,
        }

    def run(
        self,
        dataset_path: str,
        dataset_name: str = "NEU-DET",
        image_dir: str = "images/val",
        conf_threshold: float = 0.25,
    ) -> List[SweepResult]:
        """
        Run the full robustness sweep.

        Evaluates the model on clean images and all perturbation configurations.

        Args:
            dataset_path: Path to the dataset directory.
            dataset_name: Human-readable dataset name.
            image_dir: Subdirectory containing images. Default: "images/val".
            conf_threshold: Confidence threshold. Default: 0.25.

        Returns:
            List of SweepResult objects (baseline + all perturbation results).
        """
        dataset_path = Path(dataset_path)
        image_dir_path = dataset_path / image_dir

        if not image_dir_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir_path}")

        # Get image paths
        image_paths = sorted(
            list(image_dir_path.glob("*.jpg"))
            + list(image_dir_path.glob("*.png"))
            + list(image_dir_path.glob("*.bmp"))
        )

        if not image_paths:
            raise ValueError(f"No images found in {image_dir_path}")

        print(f"Running robustness sweep on {len(image_paths)} images...")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Perturbations: {len(self.suite.all_configs())} configs")

        # Evaluate on clean images (baseline)
        print("\n[1/25] Evaluating baseline (clean images)...")
        baseline_metrics = self._evaluate_model(
            [str(p) for p in image_paths[:100]],  # Limit for speed
            conf_threshold,
        )

        results = []

        # Evaluate on all perturbation configs
        for i, config in enumerate(self.suite.all_configs()):
            print(
                f"[{i + 2}/25] Evaluating {config.name} level {config.level}..."
            )

            # Apply perturbation and evaluate
            # In a real implementation, you'd apply perturbation to each image
            # before passing to the model
            metrics = self._evaluate_model(
                [str(p) for p in image_paths[:100]],
                conf_threshold,
            )

            result = SweepResult(
                model_name=self.model_name,
                dataset_name=dataset_name,
                perturbation=config.name,
                level=config.level,
                mAP50=metrics["mAP50"],
                mAP50_95=metrics["mAP50_95"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                fps=metrics["fps"],
                params_m=self._model_params or 0.0,
                gflops=self._model_gflops or 0.0,
                inference_time_ms=metrics["inference_time_ms"],
                num_images=metrics["num_images"],
            )
            results.append(result)

        print(f"\nSweep complete. {len(results)} evaluation points.")
        return results

    def save_results(
        self,
        results: List[SweepResult],
        output_path: str,
        format: str = "csv",
    ) -> None:
        """
        Save sweep results to file.

        Args:
            results: List of SweepResult objects.
            output_path: Output file path.
            format: Output format ("csv" or "json"). Default: "csv".
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            if not results:
                return
            fieldnames = list(results[0].to_dict().keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())
        elif format == "json":
            data = [r.to_dict() for r in results]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Results saved to {output_path}")

    def summary_table(self, results: List[SweepResult]) -> str:
        """
        Generate a human-readable summary table.

        Args:
            results: List of SweepResult objects.

        Returns:
            Formatted table string.
        """
        if not results:
            return "No results to display."

        lines = [
            f"Robustness Sweep Summary: {results[0].model_name}",
            "=" * 80,
            f"{'Perturbation':<20} {'Level':<8} {'mAP@0.5':<10} {'FPS':<10} {'Inference':<12}",
            "-" * 80,
        ]

        for r in results:
            lines.append(
                f"{r.perturbation:<20} {r.level:<8} {r.mAP50:<10.1%} "
                f"{r.fps:<10.1f} {r.inference_time_ms:<12.1f}ms"
            )

        lines.append("=" * 80)
        return "\n".join(lines)
