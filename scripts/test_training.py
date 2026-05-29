#!/usr/bin/env python3
"""
Quick test training script for DigiSteel-YOLO.

This script runs a short training (5 epochs) to verify everything works
before running the full training pipeline.

Usage:
    python scripts/test_training.py
"""

import sys
from pathlib import Path

from ultralytics import YOLO


def test_training():
    """Run a quick training test."""
    print("=" * 60)
    print("  DigiSteel-YOLO Quick Training Test")
    print("=" * 60)

    # Check if dataset exists
    config_path = "configs/neu_det.yaml"
    if not Path(config_path).exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # Check if images exist
    dataset_path = Path("datasets/NEU-DET/yolo")
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run: python tools/prepare_datasets.py")
        sys.exit(1)

    train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
    print(f"  Training images: {len(train_images)}")

    if len(train_images) == 0:
        print("ERROR: No training images found")
        sys.exit(1)

    # Load YOLOv11n
    print("\n  Loading YOLOv11n...")
    model = YOLO("yolo11n.pt")

    # Run quick training test (5 epochs, small batch)
    print("\n  Starting quick training test (5 epochs)...")
    results = model.train(
        data=config_path,
        epochs=5,
        imgsz=640,
        batch=8,
        device="cpu",  # Use CPU since no GPU
        project="runs",
        name="test_training",
        exist_ok=True,
        verbose=True,
    )

    # Check results
    best_weights = Path("runs/test_training/weights/best.pt")
    if best_weights.exists():
        print(f"\n  ✓ Training successful!")
        print(f"  Best weights: {best_weights}")
        print(f"  Size: {best_weights.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n  ⚠ Training completed but best weights not found")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run full training: python scripts/train_baseline.py --dataset NEU-DET --epochs 200")
    print("  2. Train DigiSteel: python scripts/train_digisteel.py --dataset NEU-DET --epochs 200")


if __name__ == "__main__":
    test_training()
