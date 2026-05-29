# YOLOv11n Baseline Training Report

## DigiSteel-YOLO Ablation Study — Step 1

**Date:** May 29, 2026  
**Model:** YOLOv11n (Baseline)  
**Dataset:** NEU-DET  
**Hardware:** Tesla T4 (Google Colab)

---

## Training Configuration

| Parameter | Value |
|---|---|
| **Model** | YOLOv11n (pretrained) |
| **Dataset** | NEU-DET (6 classes) |
| **Image Size** | 640 × 640 |
| **Batch Size** | 16 |
| **Epochs** | 200 |
| **Optimizer** | AdamW (lr=0.001, momentum=0.9) |
| **Seed** | 42 |
| **Early Stopping** | patience=50 |
| **Training Time** | 1.3 hours |

---

## Dataset Split

| Split | Images | Labels |
|---|---|---|
| Train | 1,275 | 1,275 |
| Val | 328 | 328 |
| Test | 197 | 197 |
| **Total** | **1,800** | **1,800** |

---

## Results

### Overall Metrics

| Metric | Value |
|---|---|
| **mAP@0.5** | **77.9%** |
| **mAP@0.5:0.95** | **45.0%** |
| **Precision** | **80.6%** |
| **Recall** | **69.9%** |
| **F1 Score** | **74.8%** |

### Model Complexity

| Metric | Value |
|---|---|
| **Parameters** | 2,583,322 (2.58M) |
| **GFLOPs** | 6.3 |
| **Layers** | 101 (fused) |
| **Inference Speed** | 3.7ms/image |
| **FPS** | ~270 |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Status |
|---|---|---|---|---|---|
| **scratches** | 89.5% | 94.9% | 94.9% | 62.1% | ✅ Excellent |
| **patches** | 91.1% | 88.4% | 93.9% | 60.0% | ✅ Excellent |
| **pitted_surface** | 87.9% | 80.3% | 86.7% | 57.4% | ✅ Good |
| **inclusion** | 79.8% | 74.7% | 78.4% | 42.4% | ✅ Good |
| **rolled-in_scale** | 66.4% | 49.5% | 62.2% | 29.9% | ⚠️ Needs Work |
| **crazing** | 68.8% | 31.7% | 51.5% | 18.4% | ❌ Poor |

### Training Progress (Selected Epochs)

| Epoch | box_loss | cls_loss | dfl_loss | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| 1 | 1.742 | 3.386 | 1.785 | 18.4% | 7.8% |
| 50 | ~1.3 | ~1.5 | ~1.5 | ~70% | ~38% |
| 100 | 1.222 | 1.100 | 1.397 | 75.2% | 41.8% |
| 150 | 1.092 | 0.915 | 1.307 | 77.5% | 44.3% |
| 190 | 1.005 | 0.806 | 1.248 | 77.8% | 44.5% |
| **200** | **0.917** | **0.664** | **1.221** | **77.9%** | **45.0%** |

---

## Comparison with Reference Papers

### mAP@0.5 Ranking

| Rank | Model | mAP@0.5 | Params (M) | FPS | vs Baseline |
|---|---|---|---|---|---|
| 1 | **P10 KDM-YOLO** | 95.4% | 3.29 | 155.6 | +17.5% |
| 2 | **P11 YOLOv11-EMD** | 94.9% | N/A | N/A | +17.0% |
| 3 | **P02 LAM-YOLOv10n** | 94.39% | N/A | 154 | +16.5% |
| 4 | P03 YOLO-LSDI | 83.0% | 2.7 | 162.1 | +5.1% |
| 5 | P07 ASFRW-YOLO | 83.2% | 6.20 | 125 | +5.3% |
| 6 | P09 EFEN-YOLOv8 | 80.4% | N/A | N/A | +2.5% |
| 7 | P08 MSFE-YOLO | 79.8% | 11.69 | 89.3 | +1.9% |
| 8 | P04 Lightweight-YOLOv8 | 78.6% | 2.04 | 171.5 | +0.7% |
| 9 | P05 SCCI-YOLO | 78.6% | 1.68 | 270.2 | +0.7% |
| **10** | **Our Baseline** | **77.9%** | **2.58** | **~270** | **—** |
| 11 | P01 PSF-YOLO | N/A | 1.82 | N/A | — |

### Parameter Efficiency Comparison

| Model | Params (M) | mAP@0.5 | mAP/Param |
|---|---|---|---|
| P05 SCCI-YOLO | 1.68 | 78.6% | 46.8 |
| P01 PSF-YOLO | 1.82 | N/A | — |
| P04 Lightweight-YOLOv8 | 2.04 | 78.6% | 38.5 |
| **Our Baseline** | **2.58** | **77.9%** | **30.2** |
| P03 YOLO-LSDI | 2.7 | 83.0% | 30.7 |
| P10 KDM-YOLO | 3.29 | 95.4% | 29.0 |
| P07 ASFRW-YOLO | 6.20 | 83.2% | 13.4 |
| P08 MSFE-YOLO | 11.69 | 79.8% | 6.8 |

---

## Key Observations

### Strengths

1. **Good inference speed** — ~270 FPS, competitive with fastest models
2. **Reasonable parameter count** — 2.58M, in the lightweight range
3. **High precision** — 80.6% means low false positive rate
4. **Excellent on easy classes** — scratches (94.9%), patches (93.9%)

### Weaknesses

1. **Low mAP@0.5** — 77.9% vs 95.4% best (17.5% gap)
2. **Poor recall** — 69.9% means missing 30% of defects
3. **Hard class detection** — crazing (51.5%), rolled-in_scale (62.2%)
4. **No architectural modifications** — Standard YOLOv11n, no innovations

### Root Cause Analysis

| Problem | Cause | Solution |
|---|---|---|
| Low mAP | No attention mechanisms | Add GhostConv, attention modules |
| Poor recall | Conservative predictions | Lower confidence threshold, better loss |
| Hard classes | Fine defects at low resolution | Increase image size, more augmentation |
| 17.5% gap to best | No novel modifications | Implement DigiSteel innovations |

---

## Next Steps (Ablation Study)

### Step 2: DigiSteel-YOLO (GhostConv Backbone)

**Goal:** Reduce parameters while maintaining accuracy

**Expected:**
- Parameters: 2.58M → ~2.0M (-22%)
- mAP@0.5: 77.9% → 79-81% (+1-3%)

### Step 3: DigiSteel-YOLO (Inner-WIoU Loss)

**Goal:** Improve bounding box regression

**Expected:**
- mAP@0.5: +1-2%
- Recall: +2-3%

### Step 4: DigiSteel-YOLO (GhostConv + Inner-WIoU)

**Goal:** Combined improvements

**Expected:**
- mAP@0.5: 82-85%
- Parameters: ~2.0M
- Recall: 75-80%

### Step 5: Hyperparameter Tuning

**Goal:** Optimize for maximum performance

**Options:**
- Image size: 640 → 800 → 1280
- Learning rate scheduling
- Class-weighted loss
- More augmentation

---

## Files Generated

```
runs/detect/runs/baseline_neu_det_seed42/
├── weights/
│   ├── best.pt          # Best model weights (5.5MB)
│   └── last.pt          # Last checkpoint (5.5MB)
├── results.csv          # Training metrics per epoch
├── results.png          # Training curves plot
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
├── PR_curve.png         # PR curve
└── labels.jpg           # Label distribution
```

---

## Conclusion

The YOLOv11n baseline achieves **77.9% mAP@0.5** on NEU-DET, which is:
- **Competitive** with lightweight models (P04: 78.6%, P05: 78.6%)
- **Below** the best models (P10: 95.4%, P02: 94.39%)
- **Expected** for a standard architecture without modifications

The 17.5% gap to the best model confirms that **architectural innovations are necessary**. The DigiSteel-YOLO modifications (GhostConv + Inner-WIoU) should help close this gap while maintaining lightweight properties.

---

**Report Generated:** May 29, 2026  
**Ablation Study Step:** 1 of 5  
**Next Step:** Train DigiSteel-YOLO with GhostConv backbone

---

*This report is part of the DigiSteel-YOLO ablation study documenting each experimental step for the graduation project.*
