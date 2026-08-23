"""
Train DAFE variants for ablation study.

Trains 3 models with identical settings:
1. Baseline v2 (YOLOv11n, no DAFE) — use existing weights
2. DAFE (dual-branch: edge + texture) — use existing weights
3. DAFE Edge-Only (single branch: edge only) — train new

Usage:
    python scripts/train_dafe_variants.py
"""

import sys
sys.path.insert(0, r"D:\DigiSteel-Yolo\DigiSteel-YOLO")

from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from digisteel.modules.dafe import DAFE, DAFEEdgeOnly

# Register custom modules
tasks.DAFE = DAFE
tasks.DAFEEdgeOnly = DAFEEdgeOnly

PROJECT_ROOT = Path(r"D:\DigiSteel-Yolo\DigiSteel-YOLO")
DATA_YAML = str(PROJECT_ROOT / "configs" / "data" / "neu_det.yaml")

# Shared training config (same as baseline v2)
TRAIN_CONFIG = {
    "data": DATA_YAML,
    "imgsz": 800,
    "batch": 8,
    "epochs": 300,
    "patience": 75,
    "cos_lr": True,
    "seed": 42,
    "amp": True,
    "close_mosaic": 15,
    "exist_ok": True,
    "verbose": True,
    "workers": 0,
}


def train_edge_only():
    """Train DAFE Edge-Only variant."""
    print("=" * 60)
    print("  Training DAFE Edge-Only")
    print("=" * 60)

    config_path = str(PROJECT_ROOT / "configs" / "models" / "digisteel_edge_only.yaml")

    model = YOLO(config_path)
    model.load("yolo11n.pt")  # Load pretrained backbone

    results = model.train(
        **TRAIN_CONFIG,
        project=str(PROJECT_ROOT / "runs"),
        name="digisteel_edge_only",
    )

    print(f"\nTraining complete!")
    print(f"Weights: {PROJECT_ROOT / 'runs' / 'digisteel_edge_only' / 'weights' / 'best.pt'}")


def show_status():
    """Show status of all variants."""
    variants = [
        ("Baseline v2", "runs/baseline_optimized/weights/best.pt"),
        ("DAFE (dual-branch)", "runs/digisteel_dafe/weights/best.pt"),
        ("DAFE Edge-Only", "runs/digisteel_edge_only/weights/best.pt"),
    ]

    print("=" * 60)
    print("  DAFE Ablation Study — Status")
    print("=" * 60)

    for name, path in variants:
        p = PROJECT_ROOT / path
        if p.exists():
            print(f"  [OK] {name}: {p}")
        else:
            print(f"  [--] {name}: weights not found")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_status()
    else:
        train_edge_only()
