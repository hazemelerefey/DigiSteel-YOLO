#!/usr/bin/env python3
"""
Unified training script for DigiSteel-YOLO experiments.

Trains either the YOLOv11n baseline or DigiSteel-YOLO v2 with custom modules
and Inner-WIoU loss injection.

Usage:
    # Train YOLOv11n baseline on NEU-DET
    python scripts/train.py --model baseline --data configs/data/neu_det.yaml

    # Train DigiSteel-YOLO v2 on NEU-DET
    python scripts/train.py --model digisteel_v2 --data configs/data/neu_det.yaml

    # Train with custom settings
    python scripts/train.py --model digisteel_v2 --data configs/data/neu_det.yaml \
        --epochs 300 --imgsz 640 --batch 16 --seed 42

    # Train on GC10-DET
    python scripts/train.py --model digisteel_v2 --data configs/data/gc10_det.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_baseline(args):
    """Train YOLOv11n baseline (standard Ultralytics)."""
    from ultralytics import YOLO

    print("\n[1/2] Loading YOLOv11n baseline...")
    model = YOLO("yolo11n.pt")

    print("[2/2] Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=args.project,
        name=f"baseline_seed{args.seed}",
        exist_ok=True,
        patience=args.patience,
        verbose=True,
    )

    best_weights = Path(args.project) / f"baseline_seed{args.seed}" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return str(best_weights)


def train_digisteel_v2(args):
    """Train DigiSteel-YOLO v2 with custom modules and Inner-WIoU loss."""
    from digisteel.engine.trainer import (
        DigiSteelTrainer,
        patch_model_for_digisteel,
        register_custom_modules,
    )
    from ultralytics import YOLO

    # Step 1: Register custom modules
    print("\n[1/4] Registering custom modules (GhostConv, WFCA, EMA)...")
    register_custom_modules()

    # Step 2: Load model from v2 YAML
    print("[2/4] Loading DigiSteel v2 architecture...")
    model_yaml = str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml")
    model = YOLO(model_yaml)

    # Step 3: Load pretrained weights (partial transfer)
    print("[3/4] Loading pretrained weights (partial transfer from yolo11n.pt)...")
    model.load("yolo11n.pt")
    patch_model_for_digisteel(model)

    # Step 4: Train with custom trainer
    print("[4/4] Starting training with Inner-WIoU loss...")

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "project": args.project,
        "name": f"digisteel_v2_seed{args.seed}",
        "exist_ok": True,
        "patience": args.patience,
        "verbose": True,
    }

    trainer = DigiSteelTrainer.create_trainer(overrides)
    trainer.train()

    best_weights = Path(args.project) / f"digisteel_v2_seed{args.seed}" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return str(best_weights)


def main():
    parser = argparse.ArgumentParser(
        description="DigiSteel-YOLO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["baseline", "digisteel_v2"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data config YAML (e.g., configs/data/neu_det.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Training epochs (default: 300)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--project", type=str, default="runs",
        help="Project directory for saving runs (default: runs)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DigiSteel-YOLO Training")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Data:     {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Seed:     {args.seed}")
    print(f"  Patience: {args.patience}")
    print(f"  Project:  {args.project}")
    print("=" * 60)

    if args.model == "baseline":
        train_baseline(args)
    elif args.model == "digisteel_v2":
        train_digisteel_v2(args)


if __name__ == "__main__":
    main()
