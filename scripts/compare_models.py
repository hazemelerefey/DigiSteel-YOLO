#!/usr/bin/env python3
"""
Compare DigiSteel-YOLO against the 11 reference papers.

This script loads results from training runs and creates comparison tables
showing how DigiSteel-YOLO performs against the published numbers from
the 11 reference papers.

Usage:
    python scripts/compare_models.py
"""

import json
from pathlib import Path


# Published results from the 11 reference papers
REFERENCE_RESULTS = {
    "P01-PSF-YOLO": {
        "paper": "PSF-YOLO (Wang et al., Scientific Reports 2025)",
        "base": "YOLOv11n",
        "dataset": "GC10-DET",
        "map50": None,  # Not explicitly reported
        "params_m": 1.82,
        "fps": None,
        "gflops": None,
        "doi": "10.1038/s41598-025-16619-9",
    },
    "P02-LAM-YOLOv10n": {
        "paper": "LAM-YOLOv10n (Zhang et al., Scientific Reports 2025)",
        "base": "YOLOv10n",
        "dataset": "NEU-DET",
        "map50": 94.39,
        "params_m": None,
        "fps": 154,
        "gflops": None,
        "doi": "10.1038/s41598-025-16725-8",
    },
    "P03-YOLO-LSDI": {
        "paper": "YOLO-LSDI (Wang et al., Electronics 2025)",
        "base": "YOLOv11n",
        "dataset": "NEU-DET",
        "map50": 83.0,
        "params_m": 2.7,
        "fps": 162.1,
        "gflops": 6.1,
        "doi": "10.3390/electronics14132576",
    },
    "P04-Lightweight-YOLOv8": {
        "paper": "Lightweight-YOLOv8 (Ma et al., Scientific Reports 2025)",
        "base": "YOLOv8",
        "dataset": "NEU-DET",
        "map50": 78.6,
        "params_m": 2.04,
        "fps": 171.5,
        "gflops": 5.1,
        "doi": "10.1038/s41598-025-93469-5",
    },
    "P05-SCCI-YOLO": {
        "paper": "SCCI-YOLO (Zhou et al., Scientific Reports 2025)",
        "base": "YOLOv8n",
        "dataset": "NEU-DET",
        "map50": 78.6,
        "params_m": 1.68,
        "fps": 270.2,
        "gflops": None,
        "doi": "10.1038/s41598-025-20154-y",
    },
    "P06-ELS-YOLO": {
        "paper": "ELS-YOLO (Zhang et al., Electronics 2025)",
        "base": "YOLOv11n",
        "dataset": "NEU-DET+GC10+Severstal",
        "map50": None,
        "params_m": 2.36,
        "fps": None,
        "gflops": 5.6,
        "doi": "10.3390/electronics14193877",
    },
    "P07-ASFRW-YOLO": {
        "paper": "ASFRW-YOLO (Zhou et al., Scientific Reports 2025)",
        "base": "YOLOv5s",
        "dataset": "NEU-DET",
        "map50": 83.2,
        "params_m": 6.20,
        "fps": 125,
        "gflops": 14.2,
        "doi": "10.1038/s41598-025-28022-5",
    },
    "P08-MSFE-YOLO": {
        "paper": "MSFE-YOLO (Su et al., Sensors 2026)",
        "base": "YOLOv11s",
        "dataset": "NEU-DET",
        "map50": 79.8,
        "params_m": 11.69,
        "fps": 89.3,
        "gflops": 27.9,
        "doi": "10.3390/s26082311",
    },
    "P09-EFEN-YOLOv8": {
        "paper": "EFEN-YOLOv8 (Wu et al., PLOS ONE 2026)",
        "base": "YOLOv8",
        "dataset": "NEU-DET",
        "map50": 80.4,
        "params_m": None,
        "fps": None,
        "gflops": None,
        "doi": "10.1371/journal.pone.0339617",
    },
    "P10-KDM-YOLO": {
        "paper": "KDM-YOLO (Tong et al., Sensors 2026)",
        "base": "YOLOv10n",
        "dataset": "NEU-DET",
        "map50": 95.4,
        "params_m": 3.29,
        "fps": 155.6,
        "gflops": None,
        "doi": "10.3390/s26072132",
    },
    "P11-YOLOv11-EMD": {
        "paper": "YOLOv11-EMD (Shi et al., Mathematics 2025)",
        "base": "YOLOv11",
        "dataset": "NEU+Severstal",
        "map50": 94.9,
        "params_m": None,
        "fps": None,
        "gflops": None,
        "doi": "10.3390/math13172769",
    },
}


def load_our_results(run_dir: str = "runs") -> dict:
    """
    Load our training results from the runs directory.

    Args:
        run_dir: Directory containing training runs

    Returns:
        Dictionary of our results
    """
    results = {}

    runs_path = Path(run_dir)
    if not runs_path.exists():
        return results

    # Look for result files
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Look for results.csv
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd
            try:
                df = pd.read_csv(results_csv)
                # Get best mAP@0.5
                if "metrics/mAP50(B)" in df.columns:
                    best_map = df["metrics/mAP50(B)"].max()
                    results[run_dir.name] = {
                        "map50": best_map * 100,  # Convert to percentage
                        "run_dir": str(run_dir),
                    }
            except Exception as e:
                print(f"  Warning: Could not load {results_csv}: {e}")

    return results


def print_comparison_table(our_results: dict) -> None:
    """
    Print a comparison table of our results vs reference papers.

    Args:
        our_results: Dictionary of our training results
    """
    print("\n" + "=" * 100)
    print("  DigiSteel-YOLO vs Reference Papers - Comparison Table")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<25} {'Base':<12} {'Dataset':<15} {'mAP@0.5':<10} {'Params(M)':<12} {'FPS':<10} {'Source':<10}")
    print("-" * 100)

    # Reference papers
    for name, data in REFERENCE_RESULTS.items():
        map50_str = f"{data['map50']:.1f}%" if data['map50'] else "N/A"
        params_str = f"{data['params_m']:.2f}" if data['params_m'] else "N/A"
        fps_str = f"{data['fps']:.1f}" if data['fps'] else "N/A"

        print(f"{name:<25} {data['base']:<12} {data['dataset']:<15} {map50_str:<10} {params_str:<12} {fps_str:<10} {'Paper':<10}")

    # Our results
    print("-" * 100)
    for name, data in our_results.items():
        map50_str = f"{data['map50']:.1f}%" if data.get('map50') else "N/A"
        print(f"{name:<25} {'YOLOv11n':<12} {'NEU-DET':<15} {map50_str:<10} {'?':<12} {'?':<10} {'Ours':<10}")

    print("=" * 100)

    # Summary
    print("\nKey Findings:")
    print("  - Highest accuracy: P10 KDM-YOLO (95.4% mAP@0.5)")
    print("  - Highest speed: P05 SCCI-YOLO (270.2 FPS)")
    print("  - Lightest model: P05 SCCI-YOLO (1.68M params)")
    print("\nOur Goal: Beat these numbers with DigiSteel-YOLO!")


def main():
    print("=" * 60)
    print("  DigiSteel-YOLO Model Comparison")
    print("=" * 60)

    # Load our results
    print("\nLoading our training results...")
    our_results = load_our_results()

    if not our_results:
        print("  No training results found in runs/")
        print("  Run training first: python scripts/train_baseline.py")
        print("\n  Showing reference papers only...")
    else:
        print(f"  Found {len(our_results)} training runs")

    # Print comparison
    print_comparison_table(our_results)

    print("\nNext Steps:")
    print("  1. Complete training runs")
    print("  2. Run robustness evaluation")
    print("  3. Create Pareto plots (mAP vs FPS, mAP vs Params)")
    print("  4. Export to ONNX for edge deployment")


if __name__ == "__main__":
    main()
