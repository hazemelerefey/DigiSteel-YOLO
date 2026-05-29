#!/usr/bin/env python3
"""
Evaluate DigiSteel-YOLO robustness across perturbation types and levels.

This script runs the comprehensive robustness evaluation that is the core
contribution of the DigiSteel-YOLO project.

Usage:
    python scripts/eval_robustness.py --model runs/digisteel/weights/best.pt --dataset NEU-DET
    python scripts/eval_robustness.py --model runs/baseline/weights/best.pt --dataset NEU-DET
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from digisteel.eval.robustness_sweep import RobustnessSweep
from digisteel.perturbations.suite import PerturbationSuite


def run_evaluation(
    model_path: str,
    dataset: str,
    model_name: str = None,
    output_dir: str = "evals",
) -> None:
    """
    Run comprehensive robustness evaluation.

    Args:
        model_path: Path to trained model weights
        dataset: Dataset name ("NEU-DET" or "GC10-DET")
        model_name: Human-readable model name
        output_dir: Output directory for results
    """
    # Dataset paths
    dataset_paths = {
        "NEU-DET": "datasets/NEU-DET/yolo",
        "GC10-DET": "datasets/GC10-DET/yolo",
    }

    if dataset not in dataset_paths:
        print(f"ERROR: Unknown dataset '{dataset}'")
        sys.exit(1)

    dataset_path = dataset_paths[dataset]
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Run tools/download_datasets.sh first")
        sys.exit(1)

    # Model name
    if model_name is None:
        model_name = Path(model_path).parent.parent.name

    # Output path
    output_path = Path(output_dir) / f"{model_name}_{dataset}_robustness.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  DigiSteel-YOLO Robustness Evaluation")
    print(f"  Model: {model_name}")
    print(f"  Model Path: {model_path}")
    print(f"  Dataset: {dataset}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Show perturbation matrix
    suite = PerturbationSuite()
    print(f"\nPerturbation Matrix ({len(suite.all_configs())} configs):")
    print(suite.summary())

    # Run evaluation
    sweep = RobustnessSweep(
        model_path=model_path,
        model_name=model_name,
    )

    results = sweep.run(
        dataset_path=dataset_path,
        dataset_name=dataset,
    )

    # Save results
    sweep.save_results(results, str(output_path))

    # Print summary
    print("\n" + sweep.summary_table(results))

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DigiSteel-YOLO robustness")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--dataset", type=str, default="NEU-DET", help="Dataset name")
    parser.add_argument("--name", type=str, default=None, help="Model name for output")
    parser.add_argument("--output", type=str, default="evals", help="Output directory")

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        dataset=args.dataset,
        model_name=args.name,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
