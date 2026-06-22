# Code Review: week4_4a_fresh_baseline.ipynb

**Reviewer:** ML Engineering Audit (automated)  
**Date:** 2026-06-20  
**Notebook:** `notebooks/week4_4a_fresh_baseline.ipynb`  
**Spec:** `docs/specs/2026-06-20-week4-4a-spec.md`  
**Verdict:** 🔴 **FAIL — DO NOT RUN until Critical Bug #1 is fixed**

---

## 1. CRITICAL BUGS

### 🔴 BUG-1: `val_results.box.ap_class_index` Returns Class INDICES, Not AP Values

**Location:** Cell 5 (evaluate), lines:
```python
per_class_ap[name] = float(val_results.box.ap_class_index[i]) if i < len(val_results.box.ap_class_index) else 0.0
```

**Problem:** This is **catastrophically wrong**. In the Ultralytics `Metric` class, `ap_class_index` is a list of **target class indices** used during AP computation (e.g., `[0, 1, 2, 3, 4, 5]`), NOT per-class AP values. Your per-class AP table will print the numbers 0-5 as if they were AP scores (0.0%–5.0%), completely destroying the evaluation output and the saved JSON.

**Verified from source:** [`ultralytics/utils/metrics.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py) — the `Metric` class:
```python
class Metric(SimpleClass):
    def __init__(self) -> None:
        self.ap_class_index = []  # (nc,) target class indices
```

**Verified from docs:** [Ultralytics Val Mode](https://docs.ultralytics.com/modes/val/) — the documented API:
```python
metrics.box.maps  # a list containing mAP50-95 for each category
```

**Fix:**
```python
# WRONG (current):
per_class_ap[name] = float(val_results.box.ap_class_index[i])

# CORRECT — per-class mAP50-95:
per_class_maps = val_results.box.maps  # numpy array of per-class mAP50-95
for i, name in class_names.items():
    per_class_ap[name] = float(per_class_maps[i])
```

**NOTE on mAP@0.5 vs mAP50-95:** The notebook prints "Per-class AP@0.5" but `val_results.box.maps` returns **mAP50-95** (averaged over IoU thresholds 0.5:0.95), not AP@0.5 specifically. If you truly want per-class AP@0.5, use:
```python
per_class_ap50 = val_results.box.all_ap[:, 0]  # AP@IoU=0.5 per class
```
Or accept that the label is slightly misleading and `maps` (mAP50-95) is the standard per-class metric.

**Impact:** Every downstream cell (JSON save, per-class bar chart, final summary) will contain garbage data. If you run this notebook and report "per-class AP: crazing=0.0, inclusion=1.0, patches=2.0..." you'll look incompetent.

---

### 🔴 BUG-2: `copy_paste` Missing from TRAIN_OVERRIDES — Spec Not Followed, Try/Except Is Dead Code

**Location:** Cell 4 (training)

**Problem:** The spec explicitly requires `copy_paste: 0.1`, but it is NOT in the `TRAIN_OVERRIDES` dict. The try/except block that catches `copy_paste` errors will **never trigger** because `copy_paste` is never passed to `model.train()`.

```python
# This code is dead — copy_paste is not in TRAIN_OVERRIDES
try:
    results = model.train(**TRAIN_OVERRIDES)
except Exception as e:
    error_msg = str(e)
    if "copy_paste" in error_msg.lower():  # ← never reached
        ...
```

**Why copy_paste is problematic:** The Ultralytics `default.yaml` says:
```yaml
copy_paste: 0.0  # (float) segmentation copy-paste probability
```
Copy-paste augmentation requires **segmentation masks**. NEU-DET is a detection-only dataset (bounding boxes only, no masks). Setting `copy_paste > 0` will likely fail at runtime.

**Impact:** You're silently omitting a key augmentation from the spec. The experiment is not testing what the spec says it should test.

---

### 🔴 BUG-3: Try/Except Retry Logic Is Broken Even If copy_paste Were Added

**Location:** Cell 4 (training)

**Problem:** Even if you added `copy_paste: 0.1` to `TRAIN_OVERRIDES`, the retry logic retries with the **same dict** that still contains `copy_paste`:

```python
except Exception as e:
    if "copy_paste" in error_msg.lower():
        print("Retrying without copy_paste...")
        results = model.train(**TRAIN_OVERRIDES)  # ← still has copy_paste!
```

This will crash again with the same error. An infinite loop if wrapped differently.

**Fix:**
```python
except Exception as e:
    if "copy_paste" in str(e).lower():
        TRAIN_OVERRIDES.pop("copy_paste", None)  # actually remove it
        results = model.train(**TRAIN_OVERRIDES)
    else:
        raise
```

---

### 🔴 BUG-4: `val_results.names` May Not Exist on Validation Result Object

**Location:** Cell 5 (evaluate), line:
```python
class_names = val_results.names  # {0: 'crazing', 1: 'inclusion', ...}
```

**Problem:** In Ultralytics 8.3.x, `model.val()` returns a validator object. The `.names` attribute exists on the validator but its availability depends on whether the validator was initialized from a `.pt` model (which loads class names) vs. a data YAML. When validating with `data=` explicitly passed, it should work, but this is fragile.

**Suggested safer approach:**
```python
# More robust — get names from the model itself
class_names = best_model.names
# Or from the data config
class_names = val_results.names  # works but verify it's populated
```

**Impact:** Medium — may raise `AttributeError` on some Ultralytics versions.

---

## 2. WARNINGS

### ⚠️ WARN-1: `close_mosaic=0` When `mosaic=0` — Redundant But Not Harmful

The spec says:
> close_mosaic controls when mosaic shuts off (0 = never use mosaic, which is what we want)

When `mosaic=0.0`, mosaic augmentation is already fully disabled. Setting `close_mosaic=0` (which means "don't disable mosaic in the last N epochs") is a no-op. Not a bug, but it's confusing and suggests the author doesn't understand the parameter. `close_mosaic` controls when to turn OFF mosaic during the last N epochs — it doesn't control whether mosaic is used at all.

**Recommendation:** Remove `close_mosaic` entirely when `mosaic=0`. Less confusing.

---

### ⚠️ WARN-2: Dataset Counts Don't Match Spec

**Spec says:** 1200 train / 300 val / 166 test  
**Actual on disk:** 1290 train / 344 val / 166 test

The notebook's setup cell correctly discovers and prints the actual counts, so this won't cause a crash. But the spec is wrong about the dataset size. This matters for:
- Reproducibility documentation
- Comparing with other experiments that may reference the spec numbers
- Understanding if the dataset was modified since the spec was written

**Recommendation:** Update the spec to match actual dataset counts.

---

### ⚠️ WARN-3: `label_smoothing` Parameter — Valid But Not in Official Docs Table

The `label_smoothing` parameter IS a valid Ultralytics training argument that gets passed to the loss function. However, it's notably absent from the official [training settings table](https://docs.ultralytics.com/usage/cfg/#train-settings). It works, but you're relying on undocumented behavior that could change.

**Status:** Works in 8.3.x. Not a blocker.

---

### ⚠️ WARN-4: `save_period=50` — Valid But Checkpoints Are Large

`save_period` is a valid parameter (confirmed in `default.yaml`). With 600 epochs and `save_period=50`, you'll save 12 checkpoint files. Each YOLOv11n checkpoint is ~5-10MB. That's ~60-120MB of checkpoints. Fine for local training, but be aware.

---

### ⚠️ WARN-5: `model.model` for Parameter Counting — Internal API

```python
total_params = sum(p.numel() for p in model.model.parameters())
```

This accesses the internal PyTorch module via `model.model`. This works in current Ultralytics but is an internal API. The more stable approach:

```python
from ultralytics.utils.torch_utils import model_info
model_info(model)
```

Not a blocker, just fragile.

---

### ⚠️ WARN-6: CSV Column Names Are Version-Dependent

Cell 6 parses `results.csv` with hardcoded column names:
```python
cleaned.get("train/box_loss", 0)
cleaned.get("metrics/mAP50(B)", 0)
```

These column names can change between Ultralytics major versions. The `(B)` suffix for "bounding box" task appeared in recent versions. If someone runs this with an older Ultralytics, the columns won't be found and all metrics will silently default to 0.

**Recommendation:** Add a check:
```python
required_cols = {"train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"}
actual_cols = set(cleaned.keys())
if not required_cols.issubset(actual_cols):
    print(f"⚠️ CSV columns changed! Missing: {required_cols - actual_cols}")
```

---

## 3. SUGGESTIONS

### 💡 SUGG-1: Add `task="detect"` to TRAIN_OVERRIDES

The spec mentions `task: detect` but the notebook doesn't pass it. While it's inferred from the `.pt` model, being explicit prevents surprises:

```python
TRAIN_OVERRIDES["task"] = "detect"
```

### 💡 SUGG-2: Add Existence Check for `results.csv` Column Keys

Instead of silently defaulting to 0 when a column is missing, fail loudly.

### 💡 SUGG-3: Add Training Completion Verification

After `model.train()`, verify that the expected output files exist:
```python
run_dir = RUNS_DIR / "week4_4a_fresh_baseline"
assert (run_dir / "weights" / "best.pt").exists(), "Training did not produce best.pt!"
assert (run_dir / "results.csv").exists(), "Training did not produce results.csv!"
```

### 💡 SUGG-4: Fix the Per-Class AP Plot Cell

Cell 6 references `metrics["per_class_ap"]` from a previous cell. If BUG-1 isn't fixed, this will plot the wrong data. Even after fixing BUG-1, the variable scoping is fragile:

```python
if 'metrics' in dir() and 'per_class_ap' in metrics:
```

`dir()` returns names in the current scope — in a Jupyter notebook, this depends on cell execution order. If Cell 5 wasn't run, this silently skips the plot. Better to check `if 'metrics' in globals()`.

### 💡 SUGG-5: Consider Using `model.val()` Results Directly for Per-Class AP

Instead of the manual per-class extraction, use the simpler documented API:

```python
# Get per-class mAP50-95 directly
per_class_maps = val_results.box.maps  # numpy array, one value per class
for i, name in val_results.names.items():
    print(f"  {name}: {per_class_maps[i]:.4f}")
```

---

## 4. SUMMARY TABLE

| ID | Severity | Cell | Issue | Status |
|---|---|---|---|---|
| BUG-1 | 🔴 CRITICAL | 5 | `ap_class_index` returns indices, not AP values | **Must fix** |
| BUG-2 | 🔴 CRITICAL | 4 | `copy_paste` missing from overrides (spec mismatch) | **Must fix** |
| BUG-3 | 🔴 CRITICAL | 4 | Try/except retry uses same dict (infinite fail) | **Must fix** |
| BUG-4 | 🔴 CRITICAL | 5 | `val_results.names` fragile | **Should fix** |
| WARN-1 | ⚠️ Warning | 4 | `close_mosaic=0` redundant with `mosaic=0` | Clean up |
| WARN-2 | ⚠️ Warning | 2 | Dataset counts 1290/344/166 ≠ spec 1200/300/166 | Update spec |
| WARN-3 | ⚠️ Warning | 4 | `label_smoothing` not in official docs table | OK for now |
| WARN-4 | ⚠️ Warning | 4 | `save_period=50` → 12 checkpoint files | Informational |
| WARN-5 | ⚠️ Warning | 3 | `model.model` internal API | Fragile |
| WARN-6 | ⚠️ Warning | 6 | CSV column names version-dependent | Add check |

---

## 5. VERDICT

### 🔴 FAIL — NEEDS FIXES

**Minimum fixes required before running:**

1. **Fix BUG-1:** Replace `val_results.box.ap_class_index[i]` with `val_results.box.maps[i]`
2. **Fix BUG-2:** Either add `copy_paste: 0.1` to TRAIN_OVERRIDES (with proper error handling) or remove the dead try/except
3. **Fix BUG-3:** If keeping try/except, pop `copy_paste` from dict before retry
4. **Verify BUG-4:** Test that `val_results.names` works on your Ultralytics version

**Without BUG-1 fix, the entire evaluation cell produces garbage data and the saved JSON will contain wrong per-class metrics.**

---

*Review generated from Ultralytics 8.3.x source code analysis and official documentation.*
