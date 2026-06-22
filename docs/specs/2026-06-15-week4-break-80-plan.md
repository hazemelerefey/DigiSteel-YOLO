# DigiSteel-YOLO: Week 4 — Break 80% mAP

> **Goal:** Achieve 80%+ mAP@0.5 on NEU-DET by combining GC10-DET multi-dataset training with Test-Time Augmentation.

**Strategy:** Data, not architecture, is the bottleneck. More training data + smarter evaluation = breakthrough.

**Hardware:** NVIDIA RTX 2000 Ada (17GB VRAM)  
**Framework:** Ultralytics YOLO  
**Target:** 80%+ mAP@0.5 (currently 75.8%)

---

## Research Findings

| Finding | Source | Impact |
|---------|--------|--------|
| GC10-DET: 3570 grayscale images, 10 classes | Kaggle, Roboflow | 3× more data than NEU-DET |
| Ultralytics supports multi-dataset training via data.yaml | GitHub #14040 | No need to physically merge datasets |
| NEU-DET SOTA: 94-95% mAP | Multiple papers | 80% is definitely achievable |
| TTA gives 1-3% mAP boost | Ultralytics docs | Free accuracy at inference time |
| Combined + TTA expected: 80-84% | Analysis | Breaks the 80% barrier |

---

## Phase 1: Data Acquisition (Day 1)

### Task 1.1: Download GC10-DET

Source: `https://www.kaggle.com/datasets/alex000kim/gc10det`

```bash
# Download from Kaggle (requires kaggle CLI or manual download)
# Place in: D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\GC10-DET\
```

**GC10-DET Details:**
- 3570 grayscale images of steel sheet surface defects
- 10 classes: punching, weld_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, rolled_pit, crease, waist_folding
- Image sizes vary (not fixed 200×200 like NEU-DET)
- Already in grayscale (same as NEU-DET ✅)

### Task 1.2: Convert to YOLO Format

GC10-DET may come in VOC format. Convert using our existing tool:

```bash
python tools/voc_to_yolo.py --input datasets/GC10-DET --output datasets/GC10-DET/yolo
```

Or use the Roboflow version which is already in YOLO format:
`https://universe.roboflow.com/datasetsmain/gc10-det-aa7h5`

### Task 1.3: Create Combined Data Config

**File:** `configs/data/neu_gc10_combined.yaml`

```yaml
# Combined NEU-DET + GC10-DET
# 15 classes total (6 from NEU + 10 from GC10, "inclusion" overlaps)

path: D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets

train:
  - NEU-DET/yolo/images/train
  - GC10-DET/yolo/images/train

val:
  - NEU-DET/yolo/images/val

test:
  - NEU-DET/yolo/images/test

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
  6: punching
  7: weld_line
  8: crescent_gap
  9: water_spot
  10: oil_spot
  11: silk_spot
  12: rolled_pit
  13: crease
  14: waist_folding
```

**Note:** Evaluate only on NEU-DET val/test (6 classes). The extra GC10 classes provide feature learning but are not evaluated.

---

## Phase 2: Training Experiments (Day 2-4)

### Experiment 4A: Combined Dataset (Main Experiment)

**Hypothesis:** Training on 4860 images (1290 NEU + 3570 GC10) will learn better features, improving NEU-DET mAP from 75.8% to 78-82%.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # COCO pretrained

results = model.train(
    data="configs/data/neu_gc10_combined.yaml",
    epochs=400,
    imgsz=800,
    batch=16,
    seed=42,
    patience=100,
    cos_lr=True,
    mosaic=1.0,        # Keep mosaic — proven essential
    mixup=0.15,         # Light mixup for regularization
    degrees=10.0,
    translate=0.2,
    scale=0.6,
    shear=5.0,
    fliplr=0.5,
    project="runs",
    name="week4_combined",
    exist_ok=True,
)
```

**Expected:** 78-82% mAP@0.5 on NEU-DET val

### Experiment 4B: Two-Stage Fine-Tuning

**Hypothesis:** Pretraining on combined data, then fine-tuning on NEU-DET only, will specialize the model.

**Stage 1:** Train on combined data (same as 4A)  
**Stage 2:** Fine-tune on NEU-DET only

```python
# Stage 2: Fine-tune on NEU-DET only
model = YOLO("runs/week4_combined/weights/best.pt")

results = model.train(
    data="configs/data/neu_det.yaml",  # NEU-DET only
    epochs=100,
    imgsz=800,
    batch=16,
    seed=42,
    patience=50,
    cos_lr=True,
    mosaic=1.0,
    lr0=0.001,          # Lower LR for fine-tuning
    project="runs",
    name="week4_finetuned",
    exist_ok=True,
)
```

**Expected:** 79-83% mAP@0.5

### Experiment 4C: NEU-Only Baseline with Longer Training

**Hypothesis:** Maybe we didn't train long enough. Try 600 epochs.

```python
model = YOLO("yolo11n.pt")

results = model.train(
    data="configs/data/neu_det.yaml",
    epochs=600,
    imgsz=800,
    batch=16,
    seed=42,
    patience=150,       # More patience
    cos_lr=True,
    mosaic=1.0,
    project="runs",
    name="week4_long_train",
    exist_ok=True,
)
```

**Expected:** 76-78% mAP@0.5 (likely marginal improvement)

---

## Phase 3: Evaluation with TTA (Day 5)

### Task 3.1: Standard Evaluation

```python
model = YOLO("runs/week4_combined/weights/best.pt")
metrics = model.val(data="configs/data/neu_det.yaml", split="val", imgsz=800)
print(f"mAP@0.5: {metrics.box.map50:.4f}")
```

### Task 3.2: TTA Evaluation

```python
# Test-Time Augmentation — flip + scale predictions
metrics_tta = model.val(data="configs/data/neu_det.yaml", split="val", imgsz=800, augment=True)
print(f"mAP@0.5 (TTA): {metrics_tta.box.map50:.4f}")
```

### Task 3.3: Per-Class Analysis

```python
class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
for i, name in enumerate(class_names):
    print(f"  {name}: {metrics.box.maps[i]:.4f}")
```

---

## Phase 4: Report (Day 6-7)

### Deliverables
1. **Week 4 Report** — styled HTML + PDF (same design system)
2. **Ablation log** — all experiments with results
3. **Updated evals/** — JSON results for each experiment
4. **Final recommendation** — which config to use for paper

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| mAP@0.5 | 75.8% | **80%** | 83% |
| mAP@0.5:0.95 | 43.5% | 48% | 52% |
| crazing | 13.3% | 25% | 35% |
| scratches | 54.1% | 60% | 65% |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GC10-DET classes confuse NEU-DET evaluation | Medium | Two-stage fine-tuning (4B) |
| Multi-dataset YAML doesn't work | Low | Fallback: physically merge datasets |
| TTA doesn't help | Low | Still get combined training benefit |
| Overfitting on combined data | Medium | patience=100, early stopping |

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `datasets/GC10-DET/` | New dataset |
| `configs/data/neu_gc10_combined.yaml` | Combined data config |
| `notebooks/week4_combined_training.ipynb` | Main experiment |
| `notebooks/week4_finetuning.ipynb` | Two-stage fine-tuning |
| `evals/week4_*.json` | Results |
| `docs/week4/` | Report |

---

*Plan created: June 15, 2026*
*Based on: GC10-DET research, Ultralytics multi-dataset docs, TTA analysis*
