# Transfer Learning Master Plan — NEU-DET >85% mAP

**Date:** 2026-07-02
**Goal:** Surpass 85% mAP@0.5 on NEU-DET with a reproducible, research-backed transfer-learning pipeline.

---

## 1. Research Conclusions

### Best Pretrained Baseline
| Model | Params | GFLOPs | COCO mAP | Why for NEU-DET |
|-------|--------|--------|----------|-----------------|
| **YOLOv11n** | 2.62M | 6.6 | ~39.5% | Best accuracy/efficiency trade-off; proven in our project (80.3% mAP) |
| YOLOv12n | 2.60M | 6.7 | ~40.6% | Newer attention-centric design, comparable params |
| YOLOv11s | 9.46M | 21.7 | ~46.1% | More capacity, but higher overfit risk on 1,290 train images |
| YOLOv8n | 3.16M | 8.9 | ~37.3% | Older generation, slightly weaker features |

**Selected baseline:** `yolo11n.pt` (COCO pretrained).
- Already validated on NEU-DET in this project.
- YOLOv12n is promising but less battle-tested in our codebase; can be a follow-up experiment.
- YOLOv11s overfit in prior experiments (Week 3 A2: 73.4%).

### Key Literature Insights (11 reference papers)
- **P07 ASFRW-YOLO** (Nature Sci. Rep., 2025): 83.2% mAP@0.5 on NEU-DET with YOLOv5s + ASF + RepNCSPELAN4 + WIoU v3 — fair competitor, no data leakage.
- **P03 YOLO-LSDI** (MDPI Electronics, 2025): 83.0% with YOLOv11n + AMSPPF + DSAM + LDConv + Inner-CIoU.
- **P10 KDM-YOLO** reports 95.4%, but trains at native 200×200 — not comparable to our 800×800 protocol.
- **P02 LAM-YOLOv10n** 94.39% uses PRO-DataSet with likely pre-split augmentation (data leakage).
- **P11 YOLOv11-EMD** 94.9% combines NEU-DET + Severstal (7× more data) — not comparable.

**Honest ceiling under our clean 70/20/10 split at imgsz=800:** ~83–85% mAP@0.5.
**Target:** 85%+ by combining strong baseline + DAFE + optimized recipe + advanced transfer learning.

---

## 2. Transfer Learning Strategy

### Three-Stage Pipeline
| Stage | Name | Frozen | Trainable | LR | Epochs | Purpose |
|-------|------|--------|-----------|-----|--------|---------|
| 1 | **Warm-up Head** | Backbone + Neck (0–22) | Detection head (23) | 0.001 | 50 | Adapt COCO head to NEU-DET classes quickly |
| 2 | **Neck Fine-Tune** | Backbone (0–10) | Neck (11–22) + Head (23) | 0.0005 | 100 | Learn multi-scale fusion for steel defects |
| 3 | **Full Fine-Tune** | None | All layers | 0.0001 | 200 | Polish low-level filters for grayscale defects |

### Why This Beats the Old Two-Stage
- Old pipeline: head-only (100 ep) → neck+head (200 ep) gave **79.4%** mAP, only +0.6% over no-transfer baseline (78.8%).
- New pipeline adds a **warm-up stage** with higher LR for fast head adaptation, then a **dedicated neck stage** before full fine-tune.
- Uses **discriminative learning rates**: backbone gets lower effective LR via freeze/unfreeze schedule.

### Freeze/Unfreeze Mechanics
- Use `param.requires_grad = False/True` on `model.model.named_parameters()`.
- Match layer names by index: `model.0.*` to `model.22.*` for backbone+neck, `model.23.*` for head.
- After each stage, save `best.pt` and load it as the next stage's pretrained weights.

---

## 3. Hyperparameters Customized for NEU-DET

### Dataset
- **Path:** `datasets/NEU-DET/yolo_preprocessed/dataset.yaml` (CLAHE + 320×320)
- **Splits:** 1,290 train / 344 val / 166 test
- **Classes:** 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)

### Image Size
- **imgsz = 640** (not 800, not 320)
- Rationale: Preprocessed images are 320×320. Training at 640 gives 2× upscaling, preserving fine crazing cracks without the heavy compute of 800.
- Prior best used 800 on raw 200×200 images (4× upscale). With 320×320 preprocessed, 640 is the natural 2× upscale.

### Optimizer & Learning Rate
- **Optimizer:** AdamW
- **Base LR:** stage-dependent (see table)
- **Schedule:** cosine with warmup
- **warmup_epochs:** 3
- **lrf:** 0.01 (final LR = base × 0.01)

### Augmentation (Optimized for Preprocessed 320→640)
| Param | Value | Rationale |
|-------|-------|-----------|
| mosaic | 0.0 | Disable — ruins fine crazing cracks at high upscale |
| mixup | 0.15 | Light mixup for class diversity |
| copy_paste | 0.2 | Boost rare classes (crazing, pitted_surface) |
| degrees | 10 | Small rotations — defects have orientation |
| translate | 0.1 | Small shifts |
| scale | 0.5 | Moderate scale jitter |
| shear | 2 | Minimal shear |
| flipud | 0.0 | No vertical flip — defects have orientation |
| fliplr | 0.5 | Horizontal flip OK |
| hsv_h | 0.0 | Grayscale |
| hsv_s | 0.0 | Grayscale |
| hsv_v | 0.4 | Brightness jitter for steel lighting |
| erasing | 0.4 | Random erasing |

### Loss
- **BBox loss:** Inner-WIoU (already implemented in `digisteel.engine.trainer`)
- **Classification loss:** default BCE
- **DFL loss:** default

### Class Balancing
- Use `cls_pw` inverse-frequency weights from `train_preprocessed.yaml`.
- Alternatively, use focal loss gamma tuning (experiment).

---

## 4. Architecture Enhancements

### Baseline: YOLOv11n
Start from `yolo11n.pt`.

### Add DAFE (Defect-Aware Feature Enhancement)
- Insert DAFE modules at P2 and P3 (high-resolution feature maps).
- Already proven +1.5% mAP over strong baseline (80.3% vs 78.8%).
- In transfer-learning setting, DAFE should learn even better because pretrained backbone provides richer initial features.

### Optional Add-ons (ablation order)
1. **Inner-WIoU loss** — already in trainer.
2. **CoordAttention at P4** — from P04/P09 literature; may help spatial localization.
3. **GhostConv in backbone** — from P01/P02/P04; reduces params but may hurt if overused.

### Model Config
Use `configs/models/digisteel.yaml` (YOLOv11n + DAFE at P2/P3).

---

## 5. Training Notebook Structure

`notebooks/neu_det_transfer_learning_master.ipynb`

1. **Setup & Reproducibility** — seeds, paths, device
2. **Baseline Selection** — compare yolo11n/yolo12n/yolo11s (parameter count, rationale)
3. **Data Config** — point to preprocessed dataset
4. **Stage 1: Warm-up Head** — freeze backbone+neck, train head
5. **Stage 2: Neck Fine-Tune** — load stage 1, freeze backbone, train neck+head
6. **Stage 3: Full Fine-Tune** — load stage 2, unfreeze all
7. **Evaluation on Test Set** — per-class AP, confusion matrix
8. **Comparison & Visualization** — loss curves, mAP progression
9. **Export & Save Results** — JSON, ONNX

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| mAP@0.5 | ≥ 85.0% |
| mAP@0.5:0.95 | ≥ 45.0% |
| crazing AP@0.5 | ≥ 60.0% (bottleneck class) |
| recall | ≥ 75.0% |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfit on small data | Use DAFE (lightweight), strong augmentation, early stopping (patience=50) |
| Crazing remains low | Increase copy_paste, use focal loss, add edge-aware modules |
| YOLOv11n ceiling | Try YOLOv12n or YOLOv11s with heavier regularization |
| Training instability | Gradient clipping, AMP, deterministic seeds |

---

## 8. Next Steps

1. Build `neu_det_transfer_learning_master.ipynb`.
2. Run Stage 1 (fast, ~30 min).
3. Run Stage 2 + Stage 3 sequentially.
4. Evaluate on test set and compare to 80.3% baseline.
5. If <85%, iterate on augmentation, architecture, or baseline.
