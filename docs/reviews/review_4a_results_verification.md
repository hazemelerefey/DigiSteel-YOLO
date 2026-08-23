# Results Verification: Experiment 4A

**Date:** 2026-06-20  
**Question:** Are the claimed results (78.8% mAP@0.5) real, fake, or training metrics?  
**Verdict:** ⚠️ **SUSPICIOUS — Results are plausible but the 4.4% gap between training val and test needs explanation**

---

## The Claim

```
mAP@0.5:    78.8%
mAP@0.5:0.95: 45.2%
Precision:  71.9%
Recall:     76.3%
```

---

## Evidence Collected

### 1. Training DID Happen — `results.csv` is Real

The training run produced a `results.csv` with **451 epochs** of real data. This is not fabricated — the loss curves, LR schedule, and metrics all show realistic training dynamics.

```
Training epochs: 451 (early stopped, patience=150)
Best val mAP50:  74.4% at epoch 299
Last epoch:      451 (mAP50 = 73.1%)
Early stop gap:  152 epochs since best (patience=150 → stopped)
```

### 2. `args.yaml` Confirms Real Training Config

The actual training used:
```yaml
copy_paste: 0.1        ✓ (confirmed in args.yaml)
mosaic: 0.0            ✓
mixup: 0.15            ✓
optimizer: AdamW       ✓
epochs: 600            (stopped at 451)
patience: 150          ✓
imgsz: 800             ✓
```

### 3. The JSON Was Produced by Running the Evaluation

**Timeline (from file timestamps):**
```
6:23:09 PM  → week4_4a_train_status.json (training finished)
7:00:37 PM  → runs/detect/val/ directory created (model.val() ran)
7:00:38 PM  → week4_4a_results.json (metrics saved)
7:01:47 PM  → week4_4a_training_curves.png (visualization)
7:02:01 PM  → week4_4a_summary.json (summary)
```

The `val` directory (confusion matrices, val batch images) and the JSON were created **1 second apart**. This means `best_model.val()` actually ran. The plots exist. This is NOT a manually typed JSON — it was produced by real evaluation code.

### 4. The JSON Structure Matches the Current Notebook Code

The saved JSON uses `per_class_ap50`, `experiment: "4A_fresh_baseline"`, `copy_paste_used`, etc. — exactly matching Cell 5 of the updated notebook. The notebook was updated AFTER my initial review (bug fixes applied).

### 5. The Per-Class APs Are Internally Consistent

```
crazing:          40.3%
inclusion:        82.9%
patches:          92.8%
pitted_surface:   82.2%
rolled-in_scale:  77.1%
scratches:        97.3%
Average:          78.8% ← matches claimed mAP50
```

The per-class APs average to exactly the claimed mAP50. This is self-consistent.

---

## ⚠️ THE RED FLAG: 4.4% Gap Between Training Val and Test

```
Best training val mAP50:  74.4% (epoch 299, VAL split, 344 images)
Claimed test mAP50:       78.8% (TEST split, 166 images)
Gap:                      +4.4%
```

**This is unusual.** Normally, test performance ≈ val performance or slightly worse. A 4.4% *improvement* from val to test is suspicious.

### Possible Explanations

| Explanation | Likelihood | Notes |
|---|---|---|
| **Test set is genuinely easier** | Medium | 166 test images vs 344 val images. Smaller set can skew higher if it has fewer hard cases. |
| **Val set has more hard cases** | Medium | The val set might contain more borderline/hard examples that drag down mAP. |
| **Different preprocessing** | Low | Training validation uses `rect=True` by default; test eval used explicit `imgsz=800`. Minor effect. |
| **Results were inflated** | Low-Medium | Can't rule out without re-running, but the evaluation DID run (val directory exists with real plots). |
| **Bug in evaluation** | Low | The code looks correct (uses `split='test'`, proper metric extraction). |

### Key Question: Does the Test Set Have Different Distribution?

The dataset split is:
- Train: 1,290 images
- Val: 344 images  
- Test: 166 images

If the test set has proportionally more "easy" classes (like scratches at 97.3% AP) and fewer "hard" classes (like crazing at 40.3% AP), the overall mAP50 would be higher. This is a **class distribution effect**, not fakery.

---

## What's NOT Fake

1. ✅ **Training run is real** — 451 epochs of actual training data in results.csv
2. ✅ **args.yaml matches the spec** — copy_paste, mosaic, mixup, optimizer all correct
3. ✅ **best.pt exists** — 5.5MB, saved during training
4. ✅ **Evaluation actually ran** — val directory with confusion matrices created at evaluation time
5. ✅ **JSON was produced by code** — timestamps and structure match notebook code
6. ✅ **Per-class APs are self-consistent** — average matches overall mAP50

## What's Suspicious

1. ⚠️ **4.4% gap between val and test** — unusual but not impossible
2. ⚠️ **Notebook cells have no outputs** — either cleared or code was run externally
3. ⚠️ **No args.yaml in val/ directory** — unusual for Ultralytics val runs

---

## Recommendation

**Re-run the evaluation to verify.** Execute Cell 5 of the notebook and check if the results reproduce. If the test set genuinely gives 78.8%, then the results are real — the test set is just easier than val.

To verify class distribution effect, run:
```python
# Check if test set has different class distribution than val
from ultralytics import YOLO
model = YOLO('runs/detect/week4_4a_fresh_baseline/weights/best.pt')
val_results = model.val(data='configs/data/neu_det.yaml', split='val')
print(f"Val mAP50: {val_results.box.map50}")
# Compare with test mAP50
```

If val mAP50 (re-evaluated) is also ~78%, then the training-time validation was underreporting (possible with `rect=True` or different settings). If val mAP50 is still ~74%, then the test set genuinely is easier.

---

## Bottom Line

**The results are probably real, not fake.** The evaluation actually ran (timestamps prove it), the JSON was produced by code (structure matches), and the per-class APs are self-consistent. The 4.4% gap is unusual but likely explained by the test set being easier (class distribution, fewer images).

However, **you should not report 78.8% as "the model's performance"** without understanding the val-test gap. The more conservative number is **74.4% mAP50** (from training validation). The test number might be inflated by test set characteristics.

**Safe to report:** "74.4% val mAP50, 78.8% test mAP50 (test set may be easier)"  
**NOT safe to report:** "78.8% mAP50" without qualification.
