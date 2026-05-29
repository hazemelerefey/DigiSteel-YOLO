#!/usr/bin/env python3
"""
Train YOLOv11n Baseline on NEU-DET and GC10-DET.

This script trains the baseline YOLOv11n model without any modifications.
Results are used as reference for comparison with DigiSteel-YOLO variants.

Usage:
    python scripts/train_baseline.py --dataset NEU-DET --epochs 200 --seed 42
    python scripts/train_baseline.py --dataset GC10-DET --epochs 200 --seed 42
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def train_baseline(dataset: str, epochs: int, imgsz: int, batch: int, seed: int) -> str:
    """
    Train YOLOv11n baseline on specified dataset.

    Args:
        dataset: Dataset name ("NEU-DET" or "GC10-DET")
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        seed: Random seed for reproducibility

    Returns:
        Path to best weights
    """
    # Dataset config mapping
    dataset_configs = {
        "NEU-DET": "configs/yolov11n_baseline.yaml",
        "GC10-DET": "configs/yolov11n_baseline_gc10.yaml",
    }

    if dataset not in dataset_configs:
        print(f"ERROR: Unknown dataset '{dataset}'. Use: {list(dataset_configs.keys())}")
        sys.exit(1)

    config_path = dataset_configs[dataset]
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Create the config file first or check the path.")
        sys.exit(1)

    # Run name
    run_name = f"baseline_{dataset.lower().replace('-', '_')}_seed{seed}"

    print("=" * 60)
    print(f"  Training YOLOv11n Baseline")
    print(f"  Dataset: {dataset}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Seed: {seed}")
    print(f"  Run Name: {run_name}")
    print("=" * 60)

    # Load pretrained YOLOv11n
    model = YOLO("yolo11n.pt")

    # Train
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        project="runs",
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    # Best weights path
    best_weights = f"runs/{run_name}/weights/best.pt"

    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Best weights: {best_weights}")
    print("=" * 60)

    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11n baseline")
    parser.add_argument("--dataset", type=str, default="NEU-DET", help="Dataset name (NEU-DET or GC10-DET)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_baseline(
        dataset=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
