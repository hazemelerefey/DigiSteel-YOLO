# DigiSteel-YOLO v2: Rebuild Specification

**Date:** 2026-06-03
**Author:** Hazem Elerefy (lead) + Claude Opus 4.6 (project manager)
**Status:** Approved for implementation
**Deadline:** Sunday June 7, 2026
**Parent spec:** `2026-06-01-digisteel-v2-design.md`

---

## 1. Objective

Fix the 5 critical problems in the current implementation (Mimo's work) and produce a professional-grade training pipeline that Dr. Tarek can evaluate by Sunday. The result must show DigiSteel v2 beating the YOLOv11n baseline on clean mAP, robustness, and comprehensive score.

---

## 2. What We Keep (Verified Correct)

These files correctly implement the original spec and pass unit tests:

| File | Reason to keep |
|---|---|
| `digisteel/modules/wfca.py` | Haar DWT math correct, cross-subband attention, IDWT reconstruction, alpha residual |
| `digisteel/modules/ema.py` | Standard EMA implementation, strip pooling + cross-spatial |
| `digisteel/modules/inner_wiou.py` | IoU + aspect/scale penalty formula correct |
| `configs/models/digisteel_v2.yaml` | Architecture matches spec: GhostConv→WFCA→EMA→Detect |
| `digisteel/perturbations/suite.py` | `all_configs()` returns 24 perturbation points |
| `tests/test_wfca.py` | 2D DWT/IDWT reconstruction tests pass |

---

## 3. What We Rewrite

### 3.1 Trainer (`digisteel/engine/trainer.py`)

**Problem:** Not a real subclass. Fragile monkey-patch on `trainer.train` that may never fire.

**Fix:** Proper `DetectionTrainer` subclass:

```python
from ultralytics.models.yolo.detect import DetectionTrainer

class DigiSteelTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        # Patch criterion AFTER it's built (reliable timing)
        if hasattr(model, 'criterion') and hasattr(model.criterion, 'bbox_loss'):
            model.criterion.bbox_loss.iou = InnerWIoUAdapter(lambda_weight=0.5)
        return model
```

Keep `register_custom_modules()` and `InnerWIoUAdapter` (both correct).
Keep `patch_model_for_digisteel()` for inference-only use.

### 3.2 Notebook (`notebooks/DigiSteel_v2_Colab.ipynb`)

Complete rewrite with these cells:

1. **Setup** — install deps, clone repo, `pip install -e .`
2. **Register + Config** — `register_custom_modules()`, shared constants, helpers
3. **Smoke Test** — build v2 model, verify ~2.0M params, forward pass
4. **Dataset** — Kaggle download NEU-DET, VOC→YOLO conversion
5. **Train Baseline** — 200 epochs, `yolo11n.pt` pretrained, patience=50
6. **Evaluate Baseline** — val metrics + FPS + save JSON
7. **Train DigiSteel v2** — 200 epochs, PARTIAL pretrained transfer, Inner-WIoU via DigiSteelTrainer, patience=50
8. **Evaluate v2** — val metrics + FPS + save JSON
9. **Robustness Sweep** — 24 perturbation points, both models, temp dir for perturbed images
10. **Final Comparison** — comprehensive score, per-class breakdown, robustness breakdown
11. **Reference Papers** — comparison table with all 11 papers
12. **Save & Download** — zip results

---

## 4. Critical Training Strategy

| Parameter | Baseline | DigiSteel v2 |
|---|---|---|
| Model init | `YOLO("yolo11n.pt")` | `YOLO("configs/models/digisteel_v2.yaml")` then `.load("yolo11n.pt")` |
| What transfers | All layers (full pretrained) | Matching layers only (C2f, SPPF, Detect) |
| Random init | Nothing | GhostConv, WFCA, EMA |
| Trainer | Standard `model.train()` | `DigiSteelTrainer` (Inner-WIoU loss) |
| Epochs | 200 | 200 |
| Early stopping | patience=50 | patience=50 |
| Batch size | 16 (adjust for VRAM) | 16 |
| Image size | 640 | 640 |
| Seed | 42 | 42 |
| Optimizer | AdamW (Ultralytics default) | AdamW |

### Why partial pretrained transfer is essential:
- COCO pretrained weights give 15-20% mAP boost at 200 epochs
- Without it, v2 would show ~55-65% mAP vs baseline's ~78%
- This is standard practice: all 11 reference papers use pretrained init
- Matching layers transfer knowledge; novel layers learn from scratch

---

## 5. Comprehensive Score Formula

```
Score = w_clean * clean_mAP50 +
        w_robust * avg_perturbed_mAP50 +
        w_stab * stability +
        w_eff * param_efficiency +
        w_speed * speed_normalized +
        w_code * code_availability

Where (single-dataset version, redistributing multi-dataset weight):
  w_clean = 0.235   (was 0.20, absorbs share of 0.15 multi-dataset)
  w_robust = 0.295  (was 0.25)
  w_stab = 0.175    (was 0.15)
  w_eff = 0.12      (was 0.10)
  w_speed = 0.12    (was 0.10)
  w_code = 0.055    (was 0.05)

Definitions:
  stability = 1 - (clean_mAP - worst_perturbed_mAP) / clean_mAP
  param_efficiency = (mAP50 / params_M) / max_efficiency_in_table
  speed_normalized = min(FPS / 300, 1.0)
  code_availability = 1.0 if open-source, else 0.0

For reference papers without robustness data:
  avg_perturbed_mAP50 = 0 (Not Reported = penalized)
  stability = 0
```

---

## 6. Expected Outcomes

With partial pretrained transfer + 200 epochs + Inner-WIoU:
- **Baseline:** ~77-80% mAP@0.5 (consistent with our prior run of 77.9%)
- **DigiSteel v2:** ~82-88% mAP@0.5 (WFCA helps crazing/rolled-in_scale classes)
- **Robustness:** v2 should show +2-5% avg perturbed mAP (frequency awareness helps)
- **Params:** v2 ~2.0M vs baseline 2.58M (lighter due to GhostConv)
- **Comprehensive score:** v2 wins by 10-15 points due to robustness + efficiency

Papers we should beat on comprehensive score: P03, P04, P05, P07, P08, P09 (6 out of 11).
Papers that remain competitive: P02, P10, P11 (95%+ clean mAP but no robustness data reported).

---

## 7. Implementation Tasks

1. Rewrite `digisteel/engine/trainer.py` as proper subclass
2. Rewrite `notebooks/DigiSteel_v2_Colab.ipynb` with correct training strategy
3. Run unit tests to verify nothing broke
4. Code review the notebook for correctness
5. Run training locally (both models, ~6-10h total)
6. Evaluate and generate comparison tables

---

## 8. Quality Standards

- Zero f-strings (Python 3.10 Colab compat)
- No hardcoded API keys
- Resume logic for interrupted training (check `last.pt`)
- Temp dir for robustness eval (never touch original images)
- Deterministic seeding (seed=42)
- JSON metrics saved for programmatic comparison
- Clear error messages with actionable fix instructions

---

*End of rebuild specification.*
