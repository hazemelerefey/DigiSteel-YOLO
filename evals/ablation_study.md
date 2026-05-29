# DigiSteel-YOLO Ablation Study

## Overview

This document tracks the ablation study for the DigiSteel-YOLO graduation project. Each step builds upon the previous one, documenting the impact of each modification.

**Goal:** Achieve state-of-the-art performance on steel defect detection while maintaining lightweight properties.

**Target:** Beat 95.4% mAP@0.5 (P10 KDM-YOLO) with <3M parameters.

---

## Study Progress

| Step | Model | Status | mAP@0.5 | Params (M) | FPS | Report |
|---|---|---|---|---|---|---|
| 1 | YOLOv11n Baseline | ✅ Complete | 77.9% | 2.58 | ~270 | [Link](yolov11n_baseline_results.md) |
| 2 | + GhostConv Backbone | ⏳ Pending | — | — | — | — |
| 3 | + Inner-WIoU Loss | ⏳ Pending | — | — | — | — |
| 4 | + Combined (GhostConv + Inner-WIoU) | ⏳ Pending | — | — | — | — |
| 5 | + Hyperparameter Tuning | ⏳ Pending | — | — | — | — |

---

## Step 1: YOLOv11n Baseline ✅

**Configuration:**
- Model: YOLOv11n (pretrained)
- Dataset: NEU-DET
- Image Size: 640
- Batch Size: 16
- Epochs: 200

**Results:**
- mAP@0.5: 77.9%
- mAP@0.5:0.95: 45.0%
- Precision: 80.6%
- Recall: 69.9%
- Parameters: 2.58M
- Training Time: 1.3 hours

**Key Findings:**
- Standard YOLOv11n achieves 77.9% mAP@0.5
- Poor performance on hard classes (crazing: 51.5%, rolled-in_scale: 62.2%)
- Low recall (69.9%) means missing 30% of defects
- 17.5% gap to best model (P10: 95.4%)

**Detailed Report:** [yolov11n_baseline_results.md](yolov11n_baseline_results.md)

---

## Step 2: GhostConv Backbone (Planned)

**Modification:**
- Replace standard Conv2d with GhostConv (Han et al., CVPR 2020)
- Reduces parameters by ~50% while maintaining accuracy

**Expected Results:**
- Parameters: 2.58M → ~2.0M
- mAP@0.5: 77.9% → 79-81%
- FPS: Maintained or improved

**Rationale:**
- GhostConv validated by P01, P02, P04 for steel defects
- Reduces computational cost for edge deployment
- Allows more efficient training and inference

---

## Step 3: Inner-WIoU Loss (Planned)

**Modification:**
- Replace standard CIoU loss with Inner-WIoU
- Formula: L = λ · L_InnerIoU + (1−λ) · L_WIoU_v3

**Expected Results:**
- mAP@0.5: +1-2%
- Recall: +2-3%
- Better bounding box regression

**Rationale:**
- Inner-IoU provides auxiliary bounding box constraint
- WIoU v3 provides dynamic focusing on hard examples
- Combination validated by P03, P05, P07, P11

---

## Step 4: Combined (GhostConv + Inner-WIoU) (Planned)

**Modification:**
- Apply both GhostConv and Inner-WIoU together
- This is the "DigiSteel-YOLO" headline model

**Expected Results:**
- mAP@0.5: 82-85%
- Parameters: ~2.0M
- Recall: 75-80%

**Rationale:**
- GhostConv reduces parameters
- Inner-WIoU improves accuracy
- Combined effect should be additive

---

## Step 5: Hyperparameter Tuning (Planned)

**Modifications:**
- Increase image size: 640 → 800 → 1280
- Learning rate scheduling
- Class-weighted loss for hard classes
- More augmentation (mosaic, mixup, cutmix)

**Expected Results:**
- mAP@0.5: 85-90%
- Recall: 80-85%
- Better detection of hard classes

**Rationale:**
- Larger images help detect fine defects (crazing)
- Class weighting addresses imbalance
- More augmentation improves generalization

---

## Reference Papers Comparison

| Paper | mAP@0.5 | Params (M) | FPS | Key Innovation |
|---|---|---|---|---|
| P10 KDM-YOLO | 95.4% | 3.29 | 155.6 | KWConv + C2f-DRB + MSAF |
| P11 YOLOv11-EMD | 94.9% | N/A | N/A | InnerEIoU + MSDA + DynamicConv |
| P02 LAM-YOLOv10n | 94.39% | N/A | 154 | Ghost + SMA + MFFN |
| P03 YOLO-LSDI | 83.0% | 2.7 | 162.1 | AMSPPF + DSAM + LDConv + Inner-CIoU |
| P07 ASFRW-YOLO | 83.2% | 6.20 | 125 | ASF + RepNCSPELAN4 + WIoU |
| P09 EFEN-YOLOv8 | 80.4% | N/A | N/A | SAConv + LSKA + WASPP + β-FEIoU |
| P08 MSFE-YOLO | 79.8% | 11.69 | 89.3 | MSFC + C2MSDA + AFFE |
| P04 Lightweight-YOLOv8 | 78.6% | 2.04 | 171.5 | GhostNet + MPCA + SIoU |
| P05 SCCI-YOLO | 78.6% | 1.68 | 270.2 | SPD-Conv + C2f_EMA + CCFM + Inner-IoU |
| **Our Baseline** | **77.9%** | **2.58** | **~270** | **Standard YOLOv11n** |
| P01 PSF-YOLO | N/A | 1.82 | N/A | GhostConv + MDF-Neck |

---

## Innovation Validation

### GhostConv (Our A2)

**Validated by:**
- P01 PSF-YOLO: 25% parameter reduction
- P02 LAM-YOLOv10n: Ghost module for lightweight backbone
- P04 Lightweight-YOLOv8: GhostNet backbone, -32% params, -37% GFLOPs

**Our Claim:**
- Apply GhostConv with weight-sharing across pyramid stages
- Expected: 25-35% parameter reduction

### Inner-WIoU (Our A3)

**Validated by:**
- P03 YOLO-LSDI: Inner-CIoU for better localization
- P05 SCCI-YOLO: Inner-IoU for small target detection
- P07 ASFRW-YOLO: WIoU for dynamic focusing
- P11 YOLOv11-EMD: InnerEIoU for improved regression

**Our Claim:**
- Combine Inner-IoU + WIoU v3 (novel combination)
- Expected: Improved multi-dataset generalization

---

## Research Gaps Addressed

| Gap | Our Solution | Status |
|---|---|---|
| **Gap 1:** Multi-dataset validation rare | Train on NEU-DET + GC10-DET | ⏳ Pending |
| **Gap 2:** Quantitative robustness absent | 6×4 perturbation sweep | ⏳ Pending |
| **Gap 3:** Metric reporting inconsistent | Full 8-metric reporting | ✅ Done |
| **Gap 4:** Edge deployment barely studied | ONNX-Runtime CPU export | ⏳ Pending |
| **Gap 5:** Open-source code missing | Public GitHub + Colab demo | ✅ Done |

---

## Timeline

| Week | Milestone | Status |
|---|---|---|
| 1 | Baseline training | ✅ Complete |
| 2 | GhostConv implementation | ⏳ Next |
| 3 | Inner-WIoU implementation | ⏳ Pending |
| 4 | Combined model training | ⏳ Pending |
| 5 | Hyperparameter tuning | ⏳ Pending |
| 6 | Robustness evaluation | ⏳ Pending |
| 7 | Comparison with papers | ⏳ Pending |
| 8 | ONNX export | ⏳ Pending |

---

## Notes

- All experiments use seed=42 for reproducibility
- Training on Google Colab with Tesla T4 GPU
- Results saved to `runs/` directory
- Reports saved to `evals/` directory

---

**Last Updated:** May 29, 2026  
**Current Step:** 1 of 5 (Complete)  
**Next Step:** Train DigiSteel-YOLO with GhostConv backbone

---

*This document is part of the DigiSteel-YOLO graduation project ablation study.*
