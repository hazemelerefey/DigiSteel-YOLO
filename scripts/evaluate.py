#!/usr/bin/env python3
"""
Unified evaluation script for DigiSteel-YOLO experiments.

Runs clean mAP evaluation, optional robustness sweep, and comparison table.

Usage:
    # Evaluate a single model
    python scripts/evaluate.py --weights runs/baseline_seed42/weights/best.pt \
        --data configs/data/neu_det.yaml --name baseline

    # Compare baseline vs v2
    python scripts/evaluate.py \
        --weights runs/baseline_seed42/weights/best.pt runs/digisteel_v2_seed42/weights/best.pt \
        --names baseline digisteel_v2 \
        --data configs/data/neu_det.yaml

    # With robustness sweep
    python scripts/evaluate.py \
        --weights runs/digisteel_v2_seed42/weights/best.pt \
        --names digisteel_v2 \
        --data configs/data/neu_det.yaml \
        --robustness
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_clean(weights_path: str, data_path: str, imgsz: int = 640):
    """
    Run clean (unperturbed) evaluation using Ultralytics val().

    Returns dict with mAP50, mAP50_95, precision, recall, per-class mAP50.
    """
    from digisteel.engine.trainer import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()

    model = YOLO(weights_path)
    results = model.val(data=data_path, imgsz=imgsz, verbose=True)

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_mAP50": {},
    }

    if hasattr(results.box, "maps") and results.box.maps is not None:
        class_names = results.names if hasattr(results, "names") else {}
        for i, ap in enumerate(results.box.maps):
            name = class_names.get(i, f"class_{i}")
            metrics["per_class_mAP50"][name] = float(ap)

    return metrics


def evaluate_robustness(weights_path: str, data_path: str, imgsz: int = 640):
    """
    Run robustness sweep across 6 perturbation types x 4 levels.

    Creates perturbed copies of validation images, evaluates model on each,
    and returns per-configuration results.
    """
    import shutil
    import tempfile

    import cv2
    import yaml
    from digisteel.engine.trainer import register_custom_modules
    from digisteel.perturbations.suite import PerturbationSuite
    from ultralytics import YOLO

    register_custom_modules()

    model = YOLO(weights_path)
    suite = PerturbationSuite()

    # Load data config to find dataset path
    data_path_obj = Path(data_path)
    if not data_path_obj.is_absolute():
        data_path_obj = PROJECT_ROOT / data_path_obj

    with open(data_path_obj) as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = Path(data_cfg.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    val_dir = dataset_path / data_cfg.get("val", "images/val")
    val_labels_dir = val_dir.parent.parent / "labels" / val_dir.name

    image_paths = sorted(
        list(val_dir.glob("*.jpg"))
        + list(val_dir.glob("*.png"))
        + list(val_dir.glob("*.bmp"))
    )

    if not image_paths:
        print(f"  WARNING: No images found in {val_dir}")
        return []

    results = []
    total = len(suite.all_configs())

    for i, config in enumerate(suite.all_configs()):
        print(f"  [{i+1}/{total}] {config.name} L{config.level}...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_images = Path(tmpdir) / "images" / "val"
            tmp_labels = Path(tmpdir) / "labels" / "val"
            tmp_images.mkdir(parents=True)
            tmp_labels.mkdir(parents=True)

            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                perturbed = suite.apply(img, config.name, config.level)
                cv2.imwrite(str(tmp_images / img_path.name), perturbed)

                label_path = val_labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy2(label_path, tmp_labels / label_path.name)

            tmp_data = Path(tmpdir) / "data.yaml"
            tmp_cfg = {
                "path": tmpdir,
                "val": "images/val",
                "nc": data_cfg["nc"],
                "names": data_cfg["names"],
            }
            with open(tmp_data, "w") as f:
                yaml.dump(tmp_cfg, f)

            val_results = model.val(data=str(tmp_data), imgsz=imgsz, verbose=False)
            result = {
                "perturbation": config.name,
                "level": config.level,
                "mAP50": float(val_results.box.map50),
                "mAP50_95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
            }
            results.append(result)
            print(f"mAP@0.5={result['mAP50']:.3f}")

    return results


def compute_comprehensive_score(
    clean_metrics: dict,
    robustness_results: list,
    params_m: float,
    fps: float,
    multi_dataset_consistency: float = None,
):
    """
    Compute comprehensive evaluation score (unified formula).

    Single-dataset weights (from rebuild spec Section 5):
    - Clean mAP@0.5: 23.5%
    - Avg perturbed mAP@0.5: 29.5%
    - Robustness stability: 17.5%
    - Parameter efficiency: 12.0%
    - Inference speed: 12.0%
    - Code availability: 5.5%
    """
    clean_map = clean_metrics["mAP50"]

    if robustness_results:
        perturbed_maps = [r["mAP50"] for r in robustness_results]
        avg_perturbed = sum(perturbed_maps) / len(perturbed_maps)
        worst_perturbed = min(perturbed_maps)
        stability = 1.0 - (clean_map - worst_perturbed) / max(clean_map, 1e-7)
    else:
        avg_perturbed = 0.0
        stability = 0.0

    # Normalize param efficiency (mAP/params_M, capped at ~50)
    param_eff = min(clean_map / max(params_m, 0.1), 50.0) / 50.0
    # Normalize speed (FPS, capped at 300)
    speed_norm = min(fps / 300.0, 1.0)

    score = (
        0.235 * clean_map
        + 0.295 * avg_perturbed
        + 0.175 * max(stability, 0.0)
        + 0.12 * param_eff
        + 0.12 * speed_norm
        + 0.055 * 1.0  # code availability (always 1 for us)
    )

    return {
        "comprehensive_score": score,
        "clean_mAP50": clean_map,
        "avg_perturbed_mAP50": avg_perturbed,
        "robustness_stability": stability,
        "param_efficiency_mAP_per_M": clean_map / max(params_m, 0.1),
        "fps": fps,
    }


def print_comparison_table(all_results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON TABLE: DigiSteel-YOLO v2 vs Baseline")
    print("=" * 90)

    header = f"{'Model':<20} {'mAP@0.5':>8} {'mAP@.5:.95':>10} {'Prec':>6} {'Recall':>7} {'AvgPert':>8} {'Score':>7}"
    print(header)
    print("-" * 90)

    for name, data in all_results.items():
        clean = data["clean"]
        avg_pert = "N/A"
        score_str = "N/A"

        if data.get("robustness"):
            maps = [r["mAP50"] for r in data["robustness"]]
            avg_pert = f"{sum(maps)/len(maps):.3f}"

        if data.get("comprehensive"):
            score_str = f"{data['comprehensive']['comprehensive_score']:.4f}"

        print(
            f"{name:<20} {clean['mAP50']:>8.3f} {clean['mAP50_95']:>10.3f} "
            f"{clean['precision']:>6.3f} {clean['recall']:>7.3f} {avg_pert:>8} {score_str:>7}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="DigiSteel-YOLO Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights", type=str, nargs="+", required=True,
        help="Path(s) to model weights (.pt files)",
    )
    parser.add_argument(
        "--names", type=str, nargs="+",
        help="Model names (same order as weights)",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Run robustness sweep (slow: 24 evaluations per model)",
    )
    parser.add_argument(
        "--output", type=str, default="evals/results.json",
        help="Output JSON path (default: evals/results.json)",
    )
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.weights):
        print("ERROR: --names count must match --weights count")
        sys.exit(1)

    names = args.names or [Path(w).parent.parent.name for w in args.weights]
    all_results = {}

    for name, weights in zip(names, args.weights):
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Weights: {weights}")
        print(f"{'='*60}")

        # Clean evaluation
        print("\n--- Clean Evaluation ---")
        clean = evaluate_clean(weights, args.data, args.imgsz)
        all_results[name] = {"clean": clean}

        print(f"  mAP@0.5:    {clean['mAP50']:.3f}")
        print(f"  mAP@0.5:0.95: {clean['mAP50_95']:.3f}")
        print(f"  Precision:  {clean['precision']:.3f}")
        print(f"  Recall:     {clean['recall']:.3f}")

        if clean.get("per_class_mAP50"):
            print("\n  Per-class mAP@0.5:")
            for cls_name, cls_map in clean["per_class_mAP50"].items():
                print(f"    {cls_name:<15} {cls_map:.3f}")

        # Robustness evaluation
        if args.robustness:
            print("\n--- Robustness Sweep ---")
            robustness = evaluate_robustness(weights, args.data, args.imgsz)
            all_results[name]["robustness"] = robustness

            if robustness:
                maps = [r["mAP50"] for r in robustness]
                print(f"\n  Summary: avg={sum(maps)/len(maps):.3f}, "
                      f"worst={min(maps):.3f}, best={max(maps):.3f}")

    # Print comparison if multiple models
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
