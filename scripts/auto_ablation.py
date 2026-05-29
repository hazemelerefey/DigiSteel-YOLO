#!/usr/bin/env python3
"""
DigiSteel-YOLO Automated Ablation Study
Runs all experiments automatically and generates comprehensive report.
"""

import json
import time
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import torch

# Configuration
DATASET = "NEU-DET"
SEED = 42
EPOCHS = 100
BASELINE_EPOCHS = 200

# Results storage
all_results = {}

def train_and_evaluate(name, model_config, imgsz=640, batch=16, 
                       cos_lr=False, mosaic=1.0, mixup=0.0, copy_paste=0.0,
                       epochs=EPOCHS, patience=30):
    """Train a model variant and return results."""
    
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Config: {model_config}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}")
    
    run_name = f"ablation_{name.lower().replace(' ', '_')}_{DATASET.lower()}_seed{SEED}"
    config_path = f"configs/{DATASET.lower().replace('-', '_')}.yaml"
    
    # Load model
    if model_config.endswith('.yaml'):
        model = YOLO(model_config)
    else:
        model = YOLO(model_config)
    
    # Train
    start_time = time.time()
    try:
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=SEED,
            project='runs',
            name=run_name,
            exist_ok=True,
            verbose=True,
            patience=patience,
            cos_lr=cos_lr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
        )
        training_time = time.time() - start_time
        
        # Get results
        results_csv = f"runs/{run_name}/results.csv"
        if Path(results_csv).exists():
            df = pd.read_csv(results_csv)
            best_map = df["metrics/mAP50(B)"].max()
            best_map50_95 = df["metrics/mAP50-95(B)"].max()
            
            result = {
                'experiment': name,
                'mAP50': float(best_map),
                'mAP50_95': float(best_map50_95),
                'training_time_hours': training_time / 3600,
                'imgsz': imgsz,
                'batch': batch,
                'epochs': epochs,
                'config': model_config,
            }
            
            print(f"\n  Results for {name}:")
            print(f"    mAP@0.5: {best_map:.1%}")
            print(f"    mAP@0.5:0.95: {best_map50_95:.1%}")
            print(f"    Training time: {training_time/3600:.1f} hours")
            
            return result
        else:
            print(f"  Warning: Results file not found: {results_csv}")
            return None
            
    except Exception as e:
        print(f"  Error training {name}: {e}")
        return None

def run_ablation_study():
    """Run the complete ablation study."""
    
    print("="*60)
    print("  DigiSteel-YOLO Automated Ablation Study")
    print("="*60)
    print(f"  Dataset: {DATASET}")
    print(f"  Seed: {SEED}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Experiment 1: C3Ghost Architecture
    print("\n\n" + "="*60)
    print("  EXPERIMENT 1: C3Ghost Architecture")
    print("="*60)
    
    # Create C3Ghost config
    c3ghost_config = """
nc: 6

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 1, C3Ghost, [128]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, C3Ghost, [256]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 1, C3Ghost, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 1, C3Ghost, [1024]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [1024, False]]
  - [[15, 18, 21], 1, Detect, [nc]]
"""
    
    Path('configs').mkdir(exist_ok=True)
    Path('configs/yolov11n_c3ghost.yaml').write_text(c3ghost_config)
    
    result_c3ghost = train_and_evaluate(
        name="C3Ghost_Architecture",
        model_config="configs/yolov11n_c3ghost.yaml",
        epochs=EPOCHS
    )
    if result_c3ghost:
        all_results['C3Ghost'] = result_c3ghost
    
    # Experiment 2: Image Size 800
    print("\n\n" + "="*60)
    print("  EXPERIMENT 2: Image Size 800")
    print("="*60)
    
    result_img800 = train_and_evaluate(
        name="Image_Size_800",
        model_config="yolo11n.pt",
        imgsz=800,
        batch=12,  # Reduced for larger images
        epochs=EPOCHS
    )
    if result_img800:
        all_results['ImgSize800'] = result_img800
    
    # Experiment 3: Image Size 1280
    print("\n\n" + "="*60)
    print("  EXPERIMENT 3: Image Size 1280")
    print("="*60)
    
    result_img1280 = train_and_evaluate(
        name="Image_Size_1280",
        model_config="yolo11n.pt",
        imgsz=1280,
        batch=8,  # Reduced for larger images
        epochs=EPOCHS
    )
    if result_img1280:
        all_results['ImgSize1280'] = result_img1280
    
    # Experiment 4: Enhanced Augmentation
    print("\n\n" + "="*60)
    print("  EXPERIMENT 4: Enhanced Augmentation")
    print("="*60)
    
    result_aug = train_and_evaluate(
        name="Enhanced_Augmentation",
        model_config="yolo11n.pt",
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        epochs=EPOCHS
    )
    if result_aug:
        all_results['EnhancedAug'] = result_aug
    
    # Experiment 5: Cosine Learning Rate
    print("\n\n" + "="*60)
    print("  EXPERIMENT 5: Cosine Learning Rate")
    print("="*60)
    
    result_coslr = train_and_evaluate(
        name="Cosine_LR",
        model_config="yolo11n.pt",
        cos_lr=True,
        epochs=EPOCHS
    )
    if result_coslr:
        all_results['CosineLR'] = result_coslr
    
    # Save all results
    results_path = 'evals/ablation_study_results.json'
    Path('evals').mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print("  ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_path}")
    
    # Generate report
    generate_report()
    
    return all_results

def generate_report():
    """Generate comprehensive ablation study report."""
    
    # Load baseline results
    baseline_csv = f"runs/baseline_{DATASET.lower()}_seed{SEED}/results.csv"
    baseline_map = 0.779  # Default from previous training
    baseline_map50_95 = 0.450
    
    if Path(baseline_csv).exists():
        df = pd.read_csv(baseline_csv)
        baseline_map = df["metrics/mAP50(B)"].max()
        baseline_map50_95 = df["metrics/mAP50-95(B)"].max()
    
    # Generate report
    report = f"""# DigiSteel-YOLO Ablation Study - Final Report

## Executive Summary

This ablation study evaluates different architectural and hyperparameter configurations
for the DigiSteel-YOLO steel defect detection system. The goal is to find the optimal
configuration that achieves the best performance on the NEU-DET dataset.

## Hardware & Configuration

- **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- **Dataset**: {DATASET}
- **Seed**: {SEED} (for reproducibility)
- **Baseline Epochs**: {BASELINE_EPOCHS}
- **Ablation Epochs**: {EPOCHS}

## Results Comparison

| Experiment | mAP@0.5 | mAP@0.5:0.95 | Training Time | vs Baseline | Status |
|---|---|---|---|---|---|
| **Baseline (YOLOv11n)** | {baseline_map:.1%} | {baseline_map50_95:.1%} | 1.3h | — | ✅ Reference |
| **GhostConv (Previous)** | 77.2% | 44.3% | 1.3h | -0.7% | ⚠️ Decreased |
"""
    
    # Add all experiments
    for name, data in all_results.items():
        mAP50 = data.get('mAP50', 0)
        mAP50_95 = data.get('mAP50_95', 0)
        time_h = data.get('training_time_hours', 0)
        diff = mAP50 - baseline_map
        status = "✅ Improved" if diff > 0 else "⚠️ Decreased" if diff < 0 else "➖ Same"
        
        report += f"| **{name}** | {mAP50:.1%} | {mAP50_95:.1%} | {time_h:.1f}h | {diff:+.1%} | {status} |\n"
    
    # Find best experiment
    if all_results:
        best_name = max(all_results.keys(), key=lambda k: all_results[k].get('mAP50', 0))
        best_data = all_results[best_name]
        best_map = best_data.get('mAP50', 0)
        
        report += f"""
## Key Findings

### Best Configuration: {best_name}
- **mAP@0.5**: {best_map:.1%}
- **mAP@0.5:0.95**: {best_data.get('mAP50_95', 0):.1%}
- **Training Time**: {best_data.get('training_time_hours', 0):.1f} hours
- **Improvement over Baseline**: {best_map - baseline_map:+.1%}

### Recommendations
1. **Use {best_name} configuration** for final model training
2. **Train for full {BASELINE_EPOCHS} epochs** to maximize performance
3. **Run robustness evaluation** on the best model
4. **Export to ONNX** for edge deployment
"""
    
    # Add reference papers comparison
    report += """
## Comparison with Reference Papers

| Paper | mAP@0.5 | Our Best | Difference |
|---|---|---|---|
| P10 KDM-YOLO | 95.4% | TBD | — |
| P02 LAM-YOLOv10n | 94.39% | TBD | — |
| P03 YOLO-LSDI | 83.0% | TBD | — |
| P07 ASFRW-YOLO | 83.2% | TBD | — |
| P09 EFEN-YOLOv8 | 80.4% | TBD | — |
| P08 MSFE-YOLO | 79.8% | TBD | — |
| P04 Lightweight-YOLOv8 | 78.6% | TBD | — |
| P05 SCCI-YOLO | 78.6% | TBD | — |
| **Our Baseline** | 77.9% | — | — |

## Next Steps

1. **Train final model** with best configuration for {BASELINE_EPOCHS} epochs
2. **Run robustness evaluation** (6 perturbations × 4 levels)
3. **Compare with all 11 reference papers**
4. **Export to ONNX** for edge deployment
5. **Generate final dissertation results**

---

*Report generated automatically by DigiSteel-YOLO ablation study automation*
"""
    
    # Save report
    report_path = 'evals/ablation_study_final_report.md'
    Path(report_path).write_text(report)
    
    print(report)
    print(f"\n✓ Report saved to: {report_path}")

if __name__ == "__main__":
    run_ablation_study()
