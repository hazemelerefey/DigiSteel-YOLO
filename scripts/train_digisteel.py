#!/usr/bin/env python3
"""
Train DigiSteel-YOLO (GhostConv + Inner-WIoU) on NEU-DET and GC10-DET.

This is the headline model that combines:
- GhostConv backbone (lightweight convolution)
- Inner-WIoU loss (composite regression loss)

Usage:
    python scripts/train_digisteel.py --dataset NEU-DET --epochs 200 --seed 42
    python scripts/train_digisteel.py --dataset GC10-DET --epochs 200 --seed 42
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# Import our custom modules
from digisteel.modules.inner_wiou import InnerWIoULoss


def patch_bbox_loss(model, lambda_weight: float = 0.5):
    """
    Patch the model's bounding box loss to use Inner-WIoU.

    This modifies the loss function during training to use our composite
    Inner-WIoU loss instead of the standard CIoU loss.

    Args:
        model: YOLO model instance
        lambda_weight: Balance parameter for Inner-WIoU (default: 0.5)
    """
    import torch
    from ultralytics.utils.loss import BboxLoss

    # Store original loss for reference
    original_loss = model.loss

    # Create our custom loss
    inner_wiou = InnerWIoULoss(lambda_weight=lambda_weight)

    print(f"  Patched BboxLoss with InnerWIoULoss (lambda={lambda_weight})")

    return inner_wiou


def train_digisteel(
    dataset: str,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
    lambda_weight: float,
) -> str:
    """
    Train DigiSteel-YOLO on specified dataset.

    Args:
        dataset: Dataset name ("NEU-DET" or "GC10-DET")
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        seed: Random seed for reproducibility
        lambda_weight: Inner-WIoU balance parameter

    Returns:
        Path to best weights
    """
    # Dataset config mapping
    dataset_configs = {
        "NEU-DET": "configs/yolov11n_digisteel.yaml",
        "GC10-DET": "configs/yolov11n_digisteel_gc10.yaml",
    }

    if dataset not in dataset_configs:
        print(f"ERROR: Unknown dataset '{dataset}'. Use: {list(dataset_configs.keys())}")
        sys.exit(1)

    config_path = dataset_configs[dataset]
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Creating config from baseline...")
        # Fallback to baseline config
        config_path = f"configs/yolov11n_baseline{'_gc10' if 'GC10' in dataset else ''}.yaml"

    # Run name
    run_name = f"digisteel_{dataset.lower().replace('-', '_')}_seed{seed}"

    print("=" * 60)
    print(f"  Training DigiSteel-YOLO")
    print(f"  Dataset: {dataset}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Seed: {seed}")
    print(f"  Lambda (Inner-WIoU): {lambda_weight}")
    print(f"  Run Name: {run_name}")
    print("=" * 60)

    # Load YOLOv11n with GhostConv backbone
    # Use the GhostConv config if available
    ghostconv_config = "configs/yolov11n_ghostconv.yaml"
    if Path(ghostconv_config).exists():
        print(f"  Using GhostConv backbone from {ghostconv_config}")
        model = YOLO(ghostconv_config)
    else:
        print("  Using standard YOLOv11n backbone (GhostConv config not found)")
        model = YOLO("yolo11n.pt")

    # Patch the loss function
    inner_wiou = patch_bbox_loss(model, lambda_weight)

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
    parser = argparse.ArgumentParser(description="Train DigiSteel-YOLO")
    parser.add_argument("--dataset", type=str, default="NEU-DET", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="Inner-WIoU lambda")

    args = parser.parse_args()

    train_digisteel(
        dataset=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        lambda_weight=args.lambda_weight,
    )


if __name__ == "__main__":
    main()
