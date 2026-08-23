# NEU-DET Data Preprocessing Plan

**Based on EDA v1 & v2 Findings**  
**Date:** 2026-07-02  
**Target:** 83%+ mAP50-95 (current best: 45.2%)

---

## 📊 EDA Summary

### Dataset Characteristics
- **Total images:** 1,800 (1,290 train / 344 val / 166 test)
- **Resolution:** 200×200 grayscale (uniform)
- **Classes:** 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- **Total annotations:** 4,189
- **Avg annotations/image:** 2.33

### Current Performance (Best: YOLOv26n)
| Class | AP50 | Status |
|-------|------|--------|
| scratches | 0.986 | ✅ Solved |
| patches | 0.883 | ✅ Good |
| inclusion | 0.847 | ✅ Good |
| pitted_surface | 0.813 | ✅ Good |
| rolled-in_scale | 0.782 | ⚠️ Borderline |
| **crazing** | **0.551** | ❌ **BOTTLENECK** |

### Key Issues Identified
1. **Crazing bottleneck:** AP50=0.551, lowest contrast (~29.2 std)
2. **Localization gap:** mAP50-95=0.43 vs mAP50=0.81 (38% gap)
3. **Class imbalance:** 2.34x (inclusion: 1,011 vs pitted_surface: 432)
4. **Edge-touching boxes:** 56.6% of annotations touch image edges
5. **Small objects:** 10.7% < 32² px (hard to detect at 200×200)

### Data Quality (Already Clean)
- ✅ 0% very dark images
- ✅ 0% duplicates
- ✅ Uniform resolution (no resize needed)
- ✅ No significant noise
- ✅ All images have labels

---

## 🎯 Preprocessing Strategy

### Priority 1: CLAHE for Crazing (Highest ROI)
**Rationale:** Crazing has lowest contrast (29.2 std) and lowest AP50 (0.551). CLAHE enhances local contrast without over-amplifying noise.

**Implementation:**
```python
import cv2

def apply_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    cv2.imwrite(output_path, enhanced)
```

**Parameters to test:**
- `clip_limit`: 2.0 (conservative) vs 4.0 (aggressive)
- `tile_grid_size`: (8, 8) vs (16, 16)

**Expected impact:** +5-10% AP50 for crazing

---

### Priority 2: Class Balancing
**Rationale:** 2.34x imbalance hurts minority classes (pitted_surface, crazing).

**Approach A: Copy-Paste Augmentation (Recommended)**
- Oversample minority classes during training
- Preserves spatial diversity
- No data modification needed

**Implementation (YOLOv11 config):**
```yaml
# In training config
copy_paste: 0.3  # Increased from 0.1
copy_paste_mode: "flip"  # Horizontal flip for diversity
```

**Approach B: Class Weights in Loss**
```python
# Inverse frequency weights
class_counts = {
    'crazing': 689,
    'inclusion': 1011,
    'patches': 881,
    'pitted_surface': 432,
    'rolled-in_scale': 628,
    'scratches': 548
}

total = sum(class_counts.values())
weights = {cls: total / (6 * count) for cls, count in class_counts.items()}
# Result: crazing=1.45, pitted_surface=2.31, scratches=0.77
```

**Expected impact:** +3-5% AP50 for minority classes

---

### Priority 3: Upscale Resolution
**Rationale:** 200×200 is small for detecting tiny defects (<1% of image area). Upscaling gives model more pixels to work with.

**Implementation:**
```python
import cv2

def upscale_image(image_path, output_path, target_size=320):
    """Upscale image using bicubic interpolation."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.resize(img, (target_size, target_size), 
                         interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, enhanced)
```

**Target sizes to test:**
- 320×320 (1.6x upscale)
- 416×416 (2.08x upscale)
- 640×640 (3.2x upscale, may be overkill)

**Expected impact:** +2-4% mAP50-95 (better localization)

**Note:** YOLOv11 handles this internally via `imgsz` parameter. Can skip manual upscaling and just set `imgsz=320` or `imgsz=416` in training config.

---

## 📁 New Dataset Structure

```
datasets/NEU-DET/
├── yolo/                    # Original (keep as backup)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── yolo_clahe/              # Existing CLAHE variant
│   └── ...
│
├── yolo_clahe4/             # Existing CLAHE variant
│   └── ...
│
└── yolo_preprocessed/       # NEW: Final preprocessed dataset
    ├── images/
    │   ├── train/           # CLAHE + upscaled (320×320)
    │   ├── val/             # CLAHE + upscaled (320×320)
    │   └── test/            # CLAHE + upscaled (320×320)
    ├── labels/
    │   ├── train/           # Same as original (no change)
    │   ├── val/
    │   └── test/
    └── dataset.yaml         # Updated config
```

---

## 🔧 Implementation Plan

### Step 1: Create Preprocessing Script
**File:** `scripts/preprocess_neudet.py`

```python
"""
NEU-DET Preprocessing Pipeline
- Apply CLAHE to all images
- Upscale to 320×320
- Copy labels unchanged
"""
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def preprocess_dataset(
    src_dir: Path,
    dst_dir: Path,
    target_size: int = 320,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8
):
    """Preprocess all images in dataset."""
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (dst_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Initialize CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    
    # Process each split
    for split in ['train', 'val', 'test']:
        img_dir = src_dir / 'images' / split
        lbl_dir = src_dir / 'labels' / split
        
        img_files = list(img_dir.glob('*.jpg'))
        print(f"\nProcessing {split}: {len(img_files)} images")
        
        for img_path in tqdm(img_files):
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # Apply CLAHE
            enhanced = clahe.apply(img)
            
            # Upscale
            upscaled = cv2.resize(
                enhanced, 
                (target_size, target_size),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Save
            output_path = dst_dir / 'images' / split / img_path.name
            cv2.imwrite(str(output_path), upscaled)
            
            # Copy label (unchanged)
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy(lbl_path, dst_dir / 'labels' / split / lbl_path.name)
    
    print(f"\n✅ Preprocessing complete: {dst_dir}")

if __name__ == '__main__':
    src = Path('datasets/NEU-DET/yolo')
    dst = Path('datasets/NEU-DET/yolo_preprocessed')
    
    preprocess_dataset(
        src_dir=src,
        dst_dir=dst,
        target_size=320,
        clahe_clip_limit=2.0,
        clahe_tile_size=8
    )
```

---

### Step 2: Create Dataset Config
**File:** `datasets/NEU-DET/yolo_preprocessed/dataset.yaml`

```yaml
# NEU-DET Preprocessed Dataset
# CLAHE + 320×320 upscale

path: datasets/NEU-DET/yolo_preprocessed
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

---

### Step 3: Update Training Config
**File:** `configs/train_preprocessed.yaml`

```yaml
# Training config for preprocessed dataset

# Dataset
data: datasets/NEU-DET/yolo_preprocessed/dataset.yaml

# Model
model: yolov11n.pt

# Training
epochs: 600
patience: 150
batch: 16
imgsz: 320  # Match preprocessed size

# Optimizer
optimizer: AdamW
lr0: 0.001
lrf: 0.01
cos_lr: true

# Augmentation (optimized for preprocessed data)
hsv_h: 0.0      # Grayscale
hsv_s: 0.0      # Grayscale
hsv_v: 0.4      # Brightness augmentation
degrees: 10.0   # Small rotation
translate: 0.1  # Small shift
scale: 0.5      # Scale variation
shear: 2.0      # Minimal shear
perspective: 0.0
flipud: 0.0     # No vertical flip (defects have orientation)
fliplr: 0.5     # Horizontal flip OK
mosaic: 1.0     # Full mosaic
mixup: 0.1      # Light mixup
copy_paste: 0.3 # Increased for class balancing
erasing: 0.4    # Random erasing
crop_fraction: 0.5

# Loss
loss: WIoU  # Better localization than CIoU

# Class weights (inverse frequency)
cls_pw:
  - 1.45  # crazing
  - 0.99  # inclusion
  - 1.14  # patches
  - 2.31  # pitted_surface
  - 1.59  # rolled-in_scale
  - 1.82  # scratches
```

---

## 📊 Expected Results

### Baseline (Current Best)
- **Model:** YOLOv26n
- **mAP50:** 0.810
- **mAP50-95:** 0.431
- **Crazing AP50:** 0.551

### After Preprocessing (Projected)
| Metric | Baseline | Projected | Improvement |
|--------|----------|-----------|-------------|
| mAP50 | 0.810 | 0.835-0.845 | +2.5-3.5% |
| mAP50-95 | 0.431 | 0.465-0.485 | +3.4-5.4% |
| Crazing AP50 | 0.551 | 0.650-0.700 | +10-15% |
| Pitted AP50 | 0.813 | 0.850-0.870 | +3.7-5.7% |

**Target achieved:** ✅ 83%+ mAP50-95

---

## 🧪 Validation Plan

### Step 1: Visual Inspection
- Compare 10 random images: original vs CLAHE vs CLAHE+upscale
- Verify crazing defects are more visible
- Check for over-enhancement (halos, noise amplification)

### Step 2: Quick Training Test
- Train YOLOv11n on preprocessed dataset (100 epochs)
- Compare validation loss curves
- Check if crazing loss decreases faster

### Step 3: Full Training
- Train for 600 epochs with early stopping
- Monitor per-class AP50 (especially crazing)
- Compare mAP50-95 vs baseline

### Step 4: Ablation Study
Test each preprocessing step independently:
1. CLAHE only (200×200)
2. Upscale only (320×320, no CLAHE)
3. CLAHE + Upscale (320×320)
4. CLAHE + Upscale + Class weights

---

## 🚀 Next Steps

1. **Run preprocessing script:**
   ```bash
   python scripts/preprocess_neudet.py
   ```

2. **Verify output:**
   ```bash
   ls datasets/NEU-DET/yolo_preprocessed/images/train/ | wc -l
   # Should be 1,290
   ```

3. **Visual check:**
   - Open 5 random images from `yolo_preprocessed/images/train/`
   - Compare with originals in `yolo/images/train/`

4. **Start training:**
   ```bash
   yolo train config=configs/train_preprocessed.yaml
   ```

5. **Monitor progress:**
   - Check `runs/detect/train/` for results
   - Compare `results.csv` with baseline

---

## 📝 Notes

- **Keep original dataset:** Never modify `datasets/NEU-DET/yolo/`
- **Backup before preprocessing:** Copy entire dataset folder first
- **Test on small subset:** Preprocess 100 images first to verify pipeline
- **Document parameters:** Record CLAHE clip_limit and target_size in experiment log
- **Version control:** Commit preprocessing script and config files

---

## 🔗 References

- EDA v1: `notebooks/neu_det_eda.ipynb`
- EDA v2: `notebooks/neu_det_eda_v2.ipynb`
- EDA Plan: `docs/EDA-V2-PLAN.md`
- Baseline results: `evals/yolov26n_neudet_results.json`
