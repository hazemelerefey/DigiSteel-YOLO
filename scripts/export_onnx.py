#!/usr/bin/env python3
"""
Export DigiSteel-YOLO to ONNX for edge deployment.

This script exports the trained model to ONNX format for deployment
on edge devices (Jetson, Raspberry Pi, CPU inference).

Usage:
    python scripts/export_onnx.py --model runs/digisteel/weights/best.pt --output digisteel-yolo.onnx
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def export_model(
    model_path: str,
    output_path: str,
    opset: int = 12,
    simplify: bool = True,
    verify: bool = True,
) -> str:
    """
    Export YOLO model to ONNX format.

    Args:
        model_path: Path to trained model weights
        output_path: Output ONNX file path
        opset: ONNX opset version
        simplify: Whether to simplify the ONNX model
        verify: Whether to verify the exported model

    Returns:
        Path to exported ONNX file
    """
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Exporting DigiSteel-YOLO to ONNX")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  ONNX Opset: {opset}")
    print(f"  Simplify: {simplify}")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Export to ONNX
    export_path = model.export(
        format="onnx",
        opset=opset,
        simplify=simplify,
    )

    # Move to desired output path if different
    if str(export_path) != output_path:
        import shutil
        shutil.move(str(export_path), output_path)
        export_path = output_path

    print(f"\n✓ Exported to: {export_path}")

    # Verify if requested
    if verify:
        print("\nVerifying ONNX model...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            input_shape = session.get_inputs()[0].shape
            print(f"  Input shape: {input_shape}")
            print(f"  Output count: {len(session.get_outputs())}")
            print("  ✓ ONNX model verified")
        except Exception as e:
            print(f"  ⚠ Verification failed: {e}")

    print("\n" + "=" * 60)
    print("  Export Complete!")
    print("=" * 60)

    return export_path


def main():
    parser = argparse.ArgumentParser(description="Export DigiSteel-YOLO to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--output", type=str, default="digisteel-yolo.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")

    args = parser.parse_args()

    export_model(
        model_path=args.model,
        output_path=args.output,
        opset=args.opset,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
