# EDA v2 — Problem-Specific Diagnostic Plan

## Current Performance Reality

| Model | mAP50 | mAP50-95 | Bottleneck Class |
|-------|-------|----------|------------------|
| YOLOv11n baseline | 0.788 | 0.452 | crazing (0.403) |
| YOLOv11n + DAFE | 0.803 | 0.442 | crazing (0.486) |
| YOLOv26n | 0.810 | 0.431 | crazing (0.551) |

**Target:** 83%+ mAP50 → need +2-4% improvement

### Per-Class AP50 (Best: YOLOv26n)
```
scratches:        0.986  ✅  (solved)
patches:          0.883  ✅  (good)
inclusion:        0.847  ✅  (good)
pitted_surface:   0.813  ✅  (good)
rolled-in_scale:  0.782  ⚠️  (borderline, needs +5%)
crazing:          0.551  ❌  (BOTTLENECK, needs +28%)
```

### Key Gap: mAP50=0.81 vs mAP50-95=0.43
- **Localization is weak**, not classification
- Models detect defects but bbox precision is poor
- This is a dataset annotation quality OR loss function issue

---

## What the Current EDA is Missing

1. **No per-class diagnostic** — WHY is crazing 25% behind the next class?
2. **No CLAHE comparison** — Two variants exist, never compared
3. **No annotation quality analysis** — Are bboxes tight or loose?
4. **No correlation with model performance** — EDA is disconnected from results
5. **COCO thresholds on 200×200 images** — Misleading size categories
6. **No IoU analysis** — How much overlap between annotations?
7. **No visual difficulty ranking** — Which images does the model fail on?

---

## EDA v2 Structure (Problem-Specific)

### Section 1: Setup & Data Loading
- Load existing eval results from `evals/`
- Load annotations (reuse from v1)
- Build combined DataFrame with model performance

### Section 2: The Bottleneck — Crazing Deep Dive
- **WHY is crazing at 0.551?**
  - Visual sample gallery: crazing images with bboxes
  - Compare crazing bbox sizes vs other classes
  - Compare crazing brightness/contrast vs other classes
  - Compare crazing spatial distribution vs other classes
  - Are crazing annotations consistent? (bbox tightness)
  - How many crazing images have multiple instances?
  - Crazing in train vs val — is the model seeing enough?

### Section 3: Localization Problem (mAP50 vs mAP50-95 Gap)
- **WHY is mAP50-95 stuck at 0.43?**
  - Compute bbox tightness: ratio of bbox area to actual defect area
  - Analyze annotation consistency: same defect type, different annotators?
  - Are bboxes axis-aligned or rotated? (steel scratches are often diagonal)
  - Per-class mAP50 vs mAP50-95 breakdown
  - Hypothesis: loose bboxes → good at IoU=0.5 but bad at IoU=0.75+

### Section 4: CLAHE Variant Comparison
- Compare raw vs yolo_clahe vs yolo_clahe4
  - Side-by-side visual comparison (6 classes × 3 variants)
  - Brightness/contrast histograms for each variant
  - Does CLAHE improve contrast for low-contrast classes?
  - Does CLAHE help or hurt the bottleneck classes?

### Section 5: Per-Class Difficulty Profile
- Build a "difficulty profile" for each class:
  - Instance count
  - Mean bbox area (relative to image)
  - Mean brightness
  - Mean blur score
  - Spatial distribution entropy
  - Number of edge-touching boxes
  - Current model AP50
- Correlation matrix: dataset properties → model performance

### Section 6: Annotation Quality Audit
- **Are annotations reliable?**
  - Bbox tightness: sample images, overlay bboxes, check if bboxes match defects
  - Cross-class confusion: are some defects mislabeled?
  - Bbox consistency: same defect type, are bboxes similar size?
  - Empty label files: images with defects but no annotations?
  - Duplicate annotations: same defect annotated twice?

### Section 7: Robustness vs Dataset Properties
- Load robustness results from `evals/robustness_baseline_v2.csv`
- Which perturbations hurt most? (already known: noise, blur)
- Correlate with dataset: are low-contrast classes hurt more by noise?
- Does CLAHE improve robustness?

### Section 8: Train/Val/Test Distribution Shift
- Are bottleneck classes underrepresented in training?
- Per-class instance density per split
- Are val/test images harder than train? (brightness, blur, bbox size)

### Section 9: Actionable Recommendations
- Specific, ranked by expected mAP impact
- Each recommendation tied to a concrete finding
- Expected mAP gain estimate for each action

---

## Key Questions to Answer

1. **Why is crazing AP50 = 0.551 when other classes are 0.78-0.99?**
2. **Why is mAP50-95 = 0.43 when mAP50 = 0.81? (localization gap)**
3. **Does CLAHE preprocessing actually help?**
4. **Are annotations reliable enough for tight localization?**
5. **Which specific dataset improvements would yield the biggest mAP gain?**
