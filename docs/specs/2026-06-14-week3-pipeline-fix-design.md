# Week 3 — Pipeline Fix Phase Design

**Date:** 2026-06-14
**Status:** Approved ✅
**Goal:** 75.8% → 80%+ mAP@0.5 on NEU-DET

---

## Phase A1: Config-Only Fix (Notebook 1)

**File:** `notebooks/week3_phase_a1_config_fix.ipynb`

**What changes (no code, only YOLO CLI params):**

| Param | Was | Now | Why |
|-------|-----|-----|-----|
| mosaic | 1.0 | 0.0 | 4× upscale from 200→800px blurs fine craze features |
| mixup | 0.0 | 0.15 | Regularization for small dataset |
| degrees | 0.0 | 10.0 | Defects are orientation-invariant |
| translate | 0.1 | 0.2 | More position diversity, zero artifact risk |
| scale | 0.5 | 0.6 | Moderate size variation |
| shear | 0.0 | 5.0 | Simulates off-angle camera |
| epochs | 300 | 400 | Best epoch was 274 — model wasn't done |
| patience | 75 | 100 | More room before early stop |

**Fix for notebook bug:** Use `yolo11n.pt` (bare model), NOT `yolo11n-cls.pt` (which is a classifier). Add GPU memory check cell before training.

**Train on:** YOLOv11n baseline only (no custom modules)
**Expected:** 78-79% mAP@0.5
**Train time:** ~5 hours

**Cells:**
1. Setup (imports, paths, GPU check `nvidia-smi`, memory report)
2. Dataset verification (count images, check labels)
3. Config & Train (with `!nvidia-smi` checkpoint at epoch 10)
4. Evaluate (mAP, per-class, confusion matrix)
5. Save results to `evals/week3_a1_results.json`

---

## Phase A2: Architecture Fix (Notebook 2)

**File:** `notebooks/week3_phase_a2_arch_fix.ipynb`

**What changes (code required):**

| Change | Was | Now | Why |
|--------|-----|-----|-----|
| Backbone Conv | GhostConv | Standard Conv | Restore spatial filtering for texture defects |
| WFCA | 2× (P2, P3) | Removed | DWT on 25×25 maps loses resolution; +0.1% gain |
| EMA neck | 4× | Removed | Attention stacking — 4 gates before network learns |
| New attention | None | Coordinate Attention at P3 only | Lightweight, preserves spatial info |
| Inner-WIoU | Yes | Keep | Sound, literature-supported |
| Model scale | YOLOv11n | YOLOv11s | 2× backbone capacity for fine features |
| ImgSz | 800 | 640 | VRAM-safe for YOLOv11s + safe upscale from 200px |

**New files to create:**
- `digisteel/modules/coord_attention.py` — CA module (~60 lines)
- `configs/models/digisteel_v3.yaml` — YOLOv11s + CA at P3

**Files to modify:**
- `digisteel/engine/trainer.py` — register CA module alongside existing modules

**Train on:** DigiSteel v3 (YOLOv11s + CA + Inner-WIoU)
**Expected:** 80-82% mAP@0.5
**Train time:** ~8 hours

**Cells:**
1. Setup + register modules
2. Config & architecture overview
3. Train
4. Evaluate
5. Save + compare vs A1

---

## Phase A2 Fallback

If YOLOv11s OOMs at imgsz=640 with batch=16, fall back to batch=8 or imgsz=480.

---

*Generated: 2026-06-14 | Week 3 improvement plan*