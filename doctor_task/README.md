# 🔬 Doctor Evaluation Task - CNN Baseline Model

**Branch:** `task/doctor-cnn-baseline-eval`  
**Deadline:** May 16, 2026  
**Status:** Ready for Execution

---

## 📌 Task Summary

Your doctor wants to see a **proof-of-concept CNN baseline model** on a 30-50% NEU-DET subset with:
- ✅ Complete data preprocessing pipeline
- ✅ CNN model training and evaluation
- ✅ Ablation study (preprocessing impact)
- ✅ Professional documentation
- ✅ Full reproducibility

**This task is SEPARATE from the main DigiSteel-YOLO project** but demonstrates your team's ML capabilities.

---

## 📖 Complete Execution Guide

### **PREREQUISITES (Do This First)**

#### Step 0.1: Open Command Prompt / PowerShell

On Windows:
1. Press `Windows + R`
2. Type `cmd` or `powershell`
3. Press Enter

#### Step 0.2: Navigate to Project

```bash
# Go to the project directory
cd "D:\Grade Prohect\Robust Real-Time Steel Surface Defect\digisteel-yolo"

# Verify you're in the right place
dir
# Should show: README.md, CONTRIBUTING.md, doctor_task/, etc.
```

#### Step 0.3: Checkout the Doctor Task Branch

```bash
# Make sure you're on the task branch
git checkout task/doctor-cnn-baseline-eval

# Verify
git branch
# Should show: * task/doctor-cnn-baseline-eval
```

#### Step 0.4: Create Python Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows Command Prompt:
venv\Scripts\activate
# On PowerShell:
.\venv\Scripts\Activate.ps1

# Verify (should show venv in prompt)
python --version
# Should show: Python 3.10.x or higher
```

#### Step 0.5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install base dependencies
pip install -r requirements.txt

# Install additional packages for this task
pip install tensorflow keras scikit-learn matplotlib seaborn jupyter

# Verify TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

### **EXECUTION PHASES**

#### Phase 1: Dataset Preparation (20 minutes)

**Step 1: Create 40% Subset**
```bash
python doctor_task/01_create_subset.py
```
Expected: Creates ~720 images subset from 1800 total

**Step 2: Convert VOC XML to YOLO Format**
```bash
python doctor_task/02_voc_to_yolo.py
```
Expected: Converts all annotations to YOLO txt format

**Step 3: Analyze Dataset**
```bash
python doctor_task/03_analyze_dataset.py
```
Expected: Prints dataset statistics and class distribution

#### Phase 2: Data Preprocessing (15 minutes)

**Step 4: Preprocess Images**
```bash
python doctor_task/04_preprocess_data.py
```
Expected: Applies CLAHE + augmentation, saves processed images

#### Phase 3: Model Training (2-4 hours depending on GPU)

**Step 5: Build CNN Baseline**
```bash
python doctor_task/05_cnn_baseline_model.py
```
Expected: Creates model with ~2.5M parameters

**Step 6: Train WITHOUT Preprocessing (Baseline)**
```bash
python doctor_task/06_train_baseline_without_preprocessing.py
```
Expected: Trains CNN on raw images, saves metrics

**Step 7: Train WITH Preprocessing**
```bash
python doctor_task/07_train_baseline_with_preprocessing.py
```
Expected: Trains CNN on preprocessed images, compares results

#### Phase 4: Analysis & Report (10 minutes)

**Step 8: Ablation Study**
```bash
python doctor_task/08_ablation_study.py
```
Expected: Generates comparison chart and report

---

## 📊 What Each Script Does

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_create_subset.py` | Create 40% NEU-DET subset | 1800 images | 720 images split 70/15/15 |
| `02_voc_to_yolo.py` | Convert XML annotations | VOC XML files | YOLO txt files |
| `03_analyze_dataset.py` | Dataset statistics | Images + Labels | Statistics printout |
| `04_preprocess_data.py` | CLAHE + augmentation | Raw images | Preprocessed images |
| `05_cnn_baseline_model.py` | Build CNN model | - | Model architecture (H5) |
| `06_train_baseline_...` | Train without preprocessing | Raw images | Trained model + metrics |
| `07_train_baseline_...` | Train with preprocessing | Preprocessed images | Trained model + metrics |
| `08_ablation_study.py` | Compare results | Results JSON | Comparison chart + report |

---

## 📁 Output Directory Structure

After execution, you'll have:

```
doctor_task/
├── data/
│   ├── NEU-DET-subset/                      (720 images, split)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── NEU-DET-subset-preprocessed/         (Preprocessed copies)
├── models/
│   ├── cnn_baseline.h5                      (Fresh model)
│   ├── cnn_baseline_trained_no_preprocess.h5 (Trained - baseline)
│   └── cnn_baseline_trained_with_preprocess.h5 (Trained - preprocessed)
├── logs/
│   ├── baseline_without_preprocessing_results.json
│   ├── baseline_with_preprocessing_results.json
│   ├── ablation_study_comparison.png
│   ├── ABLATION_STUDY_REPORT.md
│   └── DOCTOR_EVALUATION_REPORT.md
└── notebooks/
    └── (optional Jupyter notebooks)
```

---

## 🎯 Expected Results

### Model Performance (Approximate)

**Without Preprocessing:**
- Accuracy: ~70-75%
- F1 Score: ~65-70%

**With Preprocessing:**
- Accuracy: ~75-82%
- F1 Score: ~72-78%

**Improvement:**
- +2-7% accuracy gain
- +3-5% F1 score gain

### Training Time

- **CPU Only:** 3-4 hours total
- **GPU (RTX 3060):** 45 minutes total
- **GPU (RTX 3090):** 20 minutes total

---

## 🔍 Testing & Evaluation

### How to Test Individual Steps

```bash
# Test dataset creation
python doctor_task/03_analyze_dataset.py
# Shows: Total images, class distribution, stats

# Test model architecture
python -c "import keras; model = keras.models.load_model('doctor_task/models/cnn_baseline.h5'); model.summary()"

# Test data loading
python -c "import cv2; print(len(list(Path('doctor_task/data/NEU-DET-subset/train/images').glob('*.jpg'))))"
```

### How to View Results

```bash
# Read the main report
type doctor_task\logs\DOCTOR_EVALUATION_REPORT.md

# View ablation chart
# Open doctor_task\logs\ablation_study_comparison.png in any image viewer

# Check results JSON
type doctor_task\logs\baseline_with_preprocessing_results.json
```

---

## 📝 Ablation Study Explained

### What is Ablation Study?

An ablation study **removes components one-by-one** to understand their individual impact.

### Our Ablation Study

**Component 1: Raw Images (Baseline)**
- No preprocessing applied
- Direct normalization [0, 255] → [0, 1]
- Establishes baseline performance

**Component 2: Preprocessing Pipeline**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Improves image contrast
  - Helps CNN extract better features
  
- Data Augmentation
  - Gaussian Noise: Simulates sensor noise
  - Blur: Simulates camera blur
  - Brightness/Contrast: Simulates lighting changes
  - Rotation: Simulates viewpoint changes
  - Affine: Simulates perspective distortion

### Expected Findings

1. **CLAHE improves learning:** Better feature visibility
2. **Augmentation prevents overfitting:** Better generalization
3. **Combined = Best:** Preprocessing + Augmentation yields best results

### Why This Matters

- **For Doctor:** Shows you understand data engineering
- **For Project:** Validates preprocessing importance
- **For Publication:** Ablation studies are standard in ML research

---

## 🚨 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: tensorflow" | `pip install tensorflow` |
| "CUDA out of memory" | Reduce batch size to 16, or use CPU |
| "No such file or directory" | Ensure NEU-DET is downloaded in `datasets/` |
| "Git command not found" | Install Git from https://git-scm.com |
| "Python version too old" | Download Python 3.10+ from https://python.org |
| "venv not activating" | Try: `python -m venv venv --upgrade-deps` |

---

## 📋 Checklist Before Submission

- [ ] All 8 scripts executed without errors
- [ ] `doctor_task/logs/DOCTOR_EVALUATION_REPORT.md` exists
- [ ] `doctor_task/logs/ablation_study_comparison.png` exists
- [ ] Both trained models saved in `doctor_task/models/`
- [ ] Results JSON files contain all metrics
- [ ] Test metrics are non-zero (model trained successfully)
- [ ] Ablation study shows preprocessing impact
- [ ] Code is reproducible (same seed = same results)
- [ ] Git history shows all commits

---

## 📤 Submission Package

**Submit these files to Dr. Ghoneimy:**

```
✅ doctor_task/logs/DOCTOR_EVALUATION_REPORT.md
✅ doctor_task/logs/ablation_study_comparison.png  
✅ doctor_task/logs/ablation_study comparison.json
✅ doctor_task/models/cnn_baseline_trained_with_preprocess.h5
✅ doctor_task/logs/baseline_without_preprocessing_results.json
✅ doctor_task/logs/baseline_with_preprocessing_results.json
✅ Git log (git log --oneline --all)
```

**Also include:**
- Brief explanation of findings
- Recommendation for main project
- Link to GitHub branch

---

## 💡 Tips for Success

1. **Read Each Script:** Understand what each does before running
2. **Check GPU:** Use `nvidia-smi` to check GPU availability
3. **Monitor Training:** Watch for loss decreasing over epochs
4. **Save Early Results:** Don't delete intermediate outputs
5. **Document Issues:** If something fails, note the error for debugging

---

## 🔗 Quick Reference

| Command | Purpose |
|---------|---------|
| `git checkout task/doctor-cnn-baseline-eval` | Switch to doctor task branch |
| `venv\Scripts\activate` | Activate virtual environment |
| `pip install tensorflow` | Install TensorFlow for CNN |
| `python doctor_task/0X_*.py` | Run individual scripts |
| `python -c "import tensorflow; print(tensorflow.__version__)"` | Verify TensorFlow |
| `nvidia-smi` | Check GPU status |
| `git log --oneline` | View commit history |

---

## 📞 Support

If stuck:
1. Check the script's docstring for details
2. Read the error message carefully
3. Try reducing batch size if GPU memory error
4. Ask Hazem (team lead) for help

---

## ✅ Success Criteria

Your doctor wants to see:
- ✅ Complete pipeline execution
- ✅ Meaningful metrics (not all 0s or 1s)
- ✅ Preprocessing shows clear impact
- ✅ Professional documentation
- ✅ Reproducible results
- ✅ Proper ablation study

**If you achieve all above, doctor will be impressed!** 🎉

---

**Start here:** `python doctor_task/01_create_subset.py`

**Timeline:** 4-6 hours total (depending on GPU)

**Deadline:** May 16, 2026

Good luck! 🚀
