"""
Robustness Evaluation: 24-point perturbation sweep.

Evaluates a model's mAP@0.5 under 6 perturbation types x 4 severity levels.
Uses ultralytics model.val() for proper mAP computation.

Usage:
    python scripts/run_robustness_eval.py --model runs/baseline_optimized/weights/best.pt --name baseline_v2
    python scripts/run_robustness_eval.py --model runs/digisteel_v2/weights/best.pt --name digisteel_v2
"""

import sys
sys.path.insert(0, r"D:\DigiSteel-Yolo\DigiSteel-YOLO")

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from digisteel.perturbations.suite import PerturbationSuite

PROJECT_ROOT = Path(r"D:\DigiSteel-Yolo\DigiSteel-YOLO")
DATASET_PATH = PROJECT_ROOT / "datasets" / "NEU-DET" / "yolo"
VAL_IMAGES = DATASET_PATH / "images" / "val"
VAL_LABELS = DATASET_PATH / "labels" / "val"
TEMP_DIR = PROJECT_ROOT / "datasets" / "temp_perturbed"


def create_perturbed_dataset(perturbation: str, level: int):
    """Create a temporary dataset with perturbed validation images."""
    # Clean up temp directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    temp_images = TEMP_DIR / "images" / "val"
    temp_labels = TEMP_DIR / "labels" / "val"
    temp_images.mkdir(parents=True, exist_ok=True)
    temp_labels.mkdir(parents=True, exist_ok=True)

    # Copy labels
    for label_file in VAL_LABELS.glob("*.txt"):
        shutil.copy2(label_file, temp_labels / label_file.name)

    # Apply perturbation to images
    suite = PerturbationSuite()
    for img_file in sorted(VAL_IMAGES.glob("*.jpg")) + sorted(VAL_IMAGES.glob("*.bmp")):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        degraded = suite.apply(img, perturbation, level)
        cv2.imwrite(str(temp_images / img_file.name), degraded)

    # Create temp YAML
    temp_yaml = TEMP_DIR / "data.yaml"
    data_config = {
        "path": str(TEMP_DIR),
        "train": "images/val",  # Use val for both (we only need val)
        "val": "images/val",
        "nc": 6,
        "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    }
    with open(temp_yaml, "w") as f:
        yaml.dump(data_config, f)

    return str(temp_yaml)


def evaluate_model(model_path: str, data_yaml: str):
    """Evaluate model and return metrics."""
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=800, verbose=False, workers=0)

    p = results.box.mp
    r = results.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {
        "mAP50": results.box.map50,
        "mAP50_95": results.box.map,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def run_sweep(model_path: str, model_name: str, output_path: str):
    """Run full 24-point robustness sweep."""
    suite = PerturbationSuite()
    configs = suite.all_configs()

    print("=" * 60)
    print(f"  ROBUSTNESS SWEEP: {model_name}")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Perturbations: {len(configs)} (6 types x 4 levels)")
    print()

    results = []

    # Baseline (clean images)
    print("[0/24] Evaluating baseline (clean images)...")
    baseline_yaml = str(DATASET_PATH / "data.yaml")
    # Need to use the actual data yaml
    baseline_yaml_content = {
        "path": str(DATASET_PATH),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    }
    temp_baseline = TEMP_DIR.parent / "temp_baseline.yaml"
    with open(temp_baseline, "w") as f:
        yaml.dump(baseline_yaml_content, f)

    baseline_metrics = evaluate_model(model_path, str(temp_baseline))
    results.append({
        "perturbation": "clean",
        "level": 0,
        **baseline_metrics,
    })
    print(f"  mAP@0.5: {baseline_metrics['mAP50']:.4f}")

    # Perturbation sweep
    for i, config in enumerate(configs):
        print(f"[{i+1}/24] {config.name} level {config.level}...")

        temp_yaml = create_perturbed_dataset(config.name, config.level)
        metrics = evaluate_model(model_path, temp_yaml)

        results.append({
            "perturbation": config.name,
            "level": config.level,
            **metrics,
        })
        print(f"  mAP@0.5: {metrics['mAP50']:.4f}")

    # Clean up
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    if temp_baseline.exists():
        temp_baseline.unlink()

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["perturbation", "level", "mAP50", "mAP50_95", "precision", "recall", "f1"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Perturbation':<25} {'Level':<8} {'mAP@0.5':<10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['perturbation']:<25} {r['level']:<8} {r['mAP50']:.4f}")

    # Calculate robustness score
    perturbed_results = [r for r in results if r["perturbation"] != "clean"]
    avg_mAP = np.mean([r["mAP50"] for r in perturbed_results])
    print(f"\n  Robustness Score (avg mAP under perturbation): {avg_mAP:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robustness evaluation sweep")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--name", required=True, help="Model name for output")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROJECT_ROOT / "evals" / f"robustness_{args.name}.csv")

    run_sweep(args.model, args.name, args.output)
