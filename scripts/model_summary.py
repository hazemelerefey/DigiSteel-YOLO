#!/usr/bin/env python3
"""
Print model summary for YOLOv11n baseline and DigiSteel variants.
Shows all layers, parameters, and architecture details.

Usage:
    python scripts/model_summary.py
    python scripts/model_summary.py --model baseline
    python scripts/model_summary.py --model digisteel_v2
    python scripts/model_summary.py --model digisteel
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_model_summary(model_name: str):
    """Load and print model summary."""

    # Register custom modules if needed
    if model_name != "baseline":
        try:
            from digisteel.engine.trainer import register_custom_modules
            register_custom_modules()
            print("[OK] Custom modules registered (GhostConv, WFCA, EMA, DAFE)\n")
        except ImportError:
            print("[WARN] Could not import custom modules\n")

    from ultralytics import YOLO

    # Select model path
    if model_name == "baseline":
        model_path = "yolo11n.pt"
        print("=" * 80)
        print("  YOLOv11n Baseline Model Summary")
        print("=" * 80)
    elif model_name == "digisteel_v2":
        model_path = str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml")
        print("=" * 80)
        print("  DigiSteel-YOLO v2 Model Summary (GhostConv + WFCA + EMA)")
        print("=" * 80)
    elif model_name == "digisteel":
        model_path = str(PROJECT_ROOT / "configs" / "models" / "digisteel.yaml")
        print("=" * 80)
        print("  DigiSteel-YOLO v4 Model Summary (DAFE)")
        print("=" * 80)
    else:
        print(f"Unknown model: {model_name}")
        return

    print(f"\nModel path: {model_path}\n")

    # Load model
    model = YOLO(model_path)

    # Print full summary (layer-by-layer)
    print("-" * 80)
    print("  LAYER-BY-LAYER SUMMARY")
    print("-" * 80)
    results = model.info(verbose=True)
    print()

    # Print model parameters
    print("-" * 80)
    print("  PARAMETER BREAKDOWN")
    print("-" * 80)

    total_params = 0
    trainable_params = 0
    layer_types = {}

    for name, param in model.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        # Count layer types
        layer_type = name.split(".")[0] if "." in name else name
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += param.numel()

    print(f"\n  Total parameters:     {total_params:>12,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable:        {total_params - trainable_params:>12,}")

    # Print layer type breakdown
    print(f"\n  {'Layer Type':<30} {'Parameters':>12} {'% of Total':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1]):
        pct = count / total_params * 100
        print(f"  {layer_type:<30} {count:>12,} {pct:>9.1f}%")

    # Print module hierarchy
    print("\n" + "-" * 80)
    print("  MODULE HIERARCHY (Top-Level)")
    print("-" * 80)
    for name, module in model.model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30} {type(module).__name__:<25} {num_params:>10,} params")

    # Print named modules (first 2 levels)
    print("\n" + "-" * 80)
    print("  NAMED MODULES (First 2 Levels)")
    print("-" * 80)
    for name, module in model.model.named_modules():
        depth = name.count(".")
        if depth <= 1:
            num_params = sum(p.numel() for p in module.parameters())
            indent = "  " * depth
            print(f"  {indent}{name or 'model':<35} {type(module).__name__:<20} {num_params:>10,} params")

    # Print early layers (backbone start)
    print("\n" + "-" * 80)
    print("  EARLY LAYERS (Backbone Start - First 10)")
    print("-" * 80)
    count = 0
    for name, module in model.model.named_modules():
        if count >= 10:
            break
        if name and "." not in name:  # Top-level only
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  Layer {count}: {name:<25} {type(module).__name__:<20} {num_params:>10,} params")
            count += 1

    # Print top layers (detection head)
    print("\n" + "-" * 80)
    print("  TOP LAYERS (Detection Head)")
    print("-" * 80)
    for name, module in model.model.named_modules():
        if "detect" in name.lower() or "head" in name.lower():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<35} {type(module).__name__:<20} {num_params:>10,} params")

    # Print custom modules if any
    print("\n" + "-" * 80)
    print("  CUSTOM MODULES (if any)")
    print("-" * 80)
    custom_found = False
    custom_types = ["GhostConv", "WFCA", "EMA", "DAFE", "InnerWIoU", "CoordAttention"]
    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        if module_type in custom_types:
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<35} {module_type:<20} {num_params:>10,} params")
            custom_found = True
    if not custom_found:
        print("  No custom modules found (standard YOLOv11n)")

    print("\n" + "=" * 80)
    print("  SUMMARY COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Print model summary")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "digisteel_v2", "digisteel"],
        help="Model to summarize (default: baseline)",
    )
    args = parser.parse_args()
    print_model_summary(args.model)


if __name__ == "__main__":
    main()
