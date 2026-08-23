# Transfer Learning on YOLOv11n Baseline — Design Spec

**Date:** 2026-06-24
**Author:** Hazem Elerefy + Claude Code
**Status:** Approved

---

## Goal

Apply two-stage transfer learning (feature extraction + fine-tuning) to YOLOv11n on NEU-DET, using the exact same training recipe as `week4_fresh_baseline`. Create a new notebook with proper evaluation on the test set and overfitting/underfitting diagnostics.

---

## Architecture

**Model:** YOLOv11n (`yolo11n.pt` — COCO pretrained, 80 classes → 6 classes)
**No YAML needed:** `YOLO('yolo11n.pt')` loads both architecture and weights directly.

### Layer Distribution (24 top-level layers, 310 leaf modules)

| Section | Layers | Indices | Params | What It Learns |
|---------|--------|---------|--------|----------------|
| BACKBONE (low-level) | 149 (48%) | 0-10 | 1.37M | Edges, textures, gradients |
| NECK (mid-level) | 84 (27%) | 11-22 | 794K | Multi-scale fusion |
| HEAD (high-level) | 77 (25%) | 23 | 465K | Classification + bbox |

---

## Stage 1: Feature Extraction

**Freeze:** ALL layers (0-22)
**Train:** Detection head only (layer 23)
**Goal:** Learn "which pattern = which defect class"

| Parameter | Value |
|-----------|-------|
| lr0 | 0.001 |
| cos_lr | True |
| lrf | 0.01 |
| epochs | 100 |
| patience | 30 |
| Other params | Same as fresh_baseline |

---

## Stage 2: Fine-Tuning

**Freeze:** Early backbone (layers 0-7)
**Train:** Late backbone + neck + head (layers 8-23)
**Goal:** Adapt domain-specific features to steel defects
**LR:** 0.0001 constant (no decay — conservative)

| Parameter | Value |
|-----------|-------|
| lr0 | 0.0001 |
| cos_lr | False |
| lrf | 0.0001 |
| epochs | 200 |
| patience | 50 |
| Other params | Same as fresh_baseline |

---

## Notebook Structure

```
Cell 1:  Title & Explanation (markdown)
Cell 2:  Setup — paths, seeds, device verification
Cell 3:  Load model + layer inspection table
───────── STAGE 1 ──────────────────────────
Cell 4:  Freeze all layers, verify status
Cell 5:  Train Stage 1
Cell 6:  Evaluate Stage 1 on TEST set
Cell 7:  Loss plots + overfitting analysis
───────── STAGE 2 ──────────────────────────
Cell 8:  Unfreeze top layers, verify status
Cell 9:  Train Stage 2
Cell 10: Evaluate Stage 2 on TEST set
Cell 11: Loss plots + overfitting analysis
───────── COMPARISON ───────────────────────
Cell 12: Stage 1 vs Stage 2 vs Baseline comparison
Cell 13: Final summary table
```

---

## Evaluation Protocol

Each stage evaluates on TEST set (not val):
- mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Per-class AP@0.5
- Train vs Val loss gap (overfitting detection)

---

## Expected Results

| Stage | mAP@0.5 | Notes |
|-------|---------|-------|
| Baseline (78.8%) | 78.8% | Reference (600 epochs, no freezing) |
| Stage 1 (Feature Extraction) | ~72-75% | Head-only, limited capacity |
| Stage 2 (Fine-Tuning) | ~78-82% | Neck adaptation improves detection |

---

## Output

**File:** `notebooks/transfer_learning_fresh_baseline.ipynb`
**Weights:** `runs/detect/transfer_learning/stage1*/` and `stage2*/`
**Metrics:** `evals/transfer_learning_results.json`
