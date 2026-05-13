# DigiSteel-YOLO: Real-Time Steel Surface Defect Detection

**A robust, real-time defect detector for flat-steel surfaces using lightweight YOLO models and validated across multiple industrial datasets.**

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-team>/digisteel-yolo.git
cd digisteel-yolo
pip install -r requirements.txt
```

### Download Datasets

```bash
bash tools/download_datasets.sh
python tools/voc_to_yolo.py --dataset NEU-DET
python tools/voc_to_yolo.py --dataset GC10-DET
python tools/split_dataset.py --seed 42
```

### Train Baseline

```bash
python scripts/train_baseline.py --dataset NEU-DET --epochs 200 --seed 42
```

### Train DigiSteel-YOLO (A2 + A3)

```bash
python scripts/train_a2_a3.py --dataset NEU-DET --epochs 200 --seed 42
```

### Robustness Evaluation

```bash
python scripts/eval_robustness.py --model runs/a2_a3_neu/weights/best.pt --dataset NEU-DET
```

### Export to ONNX

```bash
python scripts/export_onnx.py --model runs/a2_a3_neu/weights/best.pt --output digisteel-yolo.onnx
```

### Google Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-team>/digisteel-yolo/blob/main/notebooks/99_colab_demo.ipynb)

---

## Project Overview

### Team

- **Lead:** Hazem Elerefy
- **Members:** Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham
- **Supervisor:** Dr. Tarek Ghoneimy
- **Program:** Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics

### Architecture

**DigiSteel-YOLO** is a YOLOv11n-based detector with two architectural modifications:

1. **A2 — GhostConv Weight-Sharing Backbone**
   - Replaces standard Conv blocks with Ghost convolutions
   - Shares weights across pyramid stages P3/P4/P5
   - **Result:** 25–35% parameter reduction

2. **A3 — Inner-WIoU Regression Loss**
   - Combines Inner-IoU (auxiliary bounding-box constraint) with WIoU v3 (dynamic focusing)
   - Formula: `loss = λ · Inner-IoU(box, gt) + (1−λ) · WIoU_v3(box, gt)` where `λ = 0.5`
   - **Result:** Improved multi-dataset generalization

### Validation Suite

- **Multi-dataset:** NEU-DET + GC10-DET with identical hyperparameters
- **Quantitative robustness:** 4 perturbations × 3 severity levels = 12 evaluation points per dataset
  - Gaussian blur (σ ∈ {1, 3, 5})
  - Gaussian noise (σ ∈ {0.05, 0.1, 0.2})
  - Brightness drift (Δ ∈ {−50, +20, +50})
  - JPEG compression (Q ∈ {30, 50, 80})
- **Full eight-metric reporting:** mAP@0.5, mAP@0.5:0.95, precision, recall, F1, FPS, params, GFLOPs
- **Edge deployment:** ONNX-Runtime CPU export (Jetson Orin Nano in Phase 2)
- **Open-source + Pareto comparison:** Against P03, P05, P09, P10 from the literature corpus

---

## Repository Structure

```
digisteel-yolo/
├── configs/                    # YOLO configuration files
│   ├── yolov11n_baseline.yaml
│   ├── yolov11n_a2_ghostconv.yaml
│   ├── yolov11n_a3_innerwiou.yaml
│   └── yolov11n_a2_a3.yaml
│
├── digisteel/                  # Main package
│   ├── __init__.py
│   ├── modules/                # Architectural modules
│   │   ├── ghost_conv.py       # A2: Ghost convolution
│   │   └── inner_wiou.py       # A3: Inner-WIoU loss
│   ├── data/                   # Dataset loaders
│   │   ├── neu_det.py
│   │   └── gc10_det.py
│   ├── perturbations/          # Robustness perturbation toolkit
│   │   ├── blur.py
│   │   ├── gaussian_noise.py
│   │   ├── brightness.py
│   │   └── jpeg.py
│   ├── eval/                   # Evaluation utilities
│   │   ├── metrics.py
│   │   ├── robustness_sweep.py
│   │   └── pareto.py
│   └── export/                 # Model export
│       └── onnx_export.py
│
├── tools/                      # Data preparation scripts
│   ├── download_datasets.sh    # Kaggle CLI dataset download
│   ├── voc_to_yolo.py         # PASCAL-VOC → YOLO conversion
│   ├── split_dataset.py        # 7:2:1 train/val/test split
│   └── requirements-tools.txt
│
├── scripts/                    # Training & evaluation scripts
│   ├── train_baseline.py
│   ├── train_a2.py
│   ├── train_a3.py
│   ├── train_a2_a3.py
│   ├── eval_robustness.py
│   ├── export_onnx.py
│   └── run_all.sh             # Full pipeline (Week 1 smoke test)
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_dataset_inspect.ipynb
│   ├── 02_baseline_train.ipynb
│   ├── 03_robustness_sweep.ipynb
│   └── 99_colab_demo.ipynb     # Public-facing demo
│
├── tests/                      # Unit & integration tests
│   ├── test_ghost_conv.py
│   ├── test_inner_wiou.py
│   ├── test_perturbations.py
│   └── conftest.py
│
├── .github/                    # GitHub Actions CI/CD
│   └── workflows/
│       ├── test.yml           # Run tests on PR
│       └── publish.yml        # Publish releases
│
├── .gitignore
├── LICENSE                     # MIT
├── README.md                   # This file
├── CONTRIBUTING.md             # Team collaboration guide
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
└── PROJECT_GUIDE.md            # Onboarding & operating guide

# Directories created at runtime (gitignored):
datasets/                       # NEU-DET, GC10-DET, Severstal
runs/                          # Training outputs
evals/                         # Robustness evaluation results
figures/                       # Generated plots
weights/                       # Model weights
```

---

## Development Workflow

### For Team Members

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-team>/digisteel-yolo.git
   cd digisteel-yolo
   ```

2. **Create a feature branch** (never work on `main` or `develop`):
   ```bash
   git checkout -b feat/hazem-a2-ghostconv
   ```
   Branch naming: `feat/[your-name]-[feature]` or `fix/[issue]`

3. **Make changes, commit regularly:**
   ```bash
   git add .
   git commit -m "Implement GhostConv weight-sharing layer"
   ```

4. **Push and open a Pull Request:**
   ```bash
   git push origin feat/hazem-a2-ghostconv
   ```

5. **Wait for CI to pass and peer review** (at least 1 approval required)

6. **Merge via GitHub (never force-push to `main`)**

### Branch Strategy

```
main (production/submission)
 ↑
 └─ release/v0.1 (milestone tags)
     ↑
     └─ develop (integration branch)
         ↑
         ├─ feat/hazem-wp1-... (WP1 lead)
         ├─ feat/youssef-wp2-... (WP2 lead)
         ├─ feat/mohamed-wp3-... (WP3 lead)
         ├─ feat/moamen-wp4-... (WP4 lead)
         └─ feat/mahmoud-wp5-... (WP5 lead)
```

---

## CI/CD Pipeline

All PRs must pass:

- ✅ **pytest**: Unit tests + integration tests
- ✅ **ruff**: Linter (style & conventions)
- ✅ **black**: Code formatter (auto-fix on PR)
- ✅ **smoke test**: 1-epoch training on 50-image NEU-DET subset

See `.github/workflows/` for configuration.

---

## Datasets

### Phase 1 (Weeks 1–12)

- **NEU-DET** (primary): 1,800 grayscale 200×200 images, 6 defect classes
- **GC10-DET** (secondary): 2,294 grayscale 2048×1000 images, 10 defect classes

### Phase 2 (Post-graduation, Weeks 13–24)

- **Severstal**: 12,500+ industrial 256×1600 images, 4 defect classes

**All datasets are open-access for academic research.** See `DATASETS.md` for download URLs and the canonical 7:2:1 split protocol (seed = 42).

---

## Reproducibility

To reproduce all Phase-1 results on a clean machine:

```bash
git clone https://github.com/<your-team>/digisteel-yolo.git
cd digisteel-yolo
pip install -r requirements.txt

# Download datasets
bash tools/download_datasets.sh
python tools/voc_to_yolo.py --dataset NEU-DET
python tools/voc_to_yolo.py --dataset GC10-DET
python tools/split_dataset.py --seed 42

# Train all models
bash scripts/run_all.sh

# Evaluate robustness
python scripts/eval_robustness.py --model runs/a2_a3_neu/weights/best.pt --dataset NEU-DET
python scripts/eval_robustness.py --model runs/a2_a3_gc10/weights/best.pt --dataset GC10-DET

# Generate figures
python scripts/generate_figures.py
```

**Expected runtime:** ~12 GPU hours (RTX 3060 / 3070 or Google Colab T4).

---

## Citation

If you use this work, please cite:

```bibtex
@misc{digisteel2026,
  title={DigiSteel-YOLO: Robust Real-Time Steel Surface Defect Detection Using Lightweight YOLO Models and Industrial Condition Testing},
  author={Elerefy, Hazem and Sherif, Youssef and Salah, Mohamed and Esmat, Moamen and Hisham, Mahmoud},
  year={2026},
  howpublished={\url{https://github.com/your-team/digisteel-yolo}},
  note={Graduation project, Digilians (MCIT) Specialized Diploma in Applied AI}
}
```

---

## License

**Code:** MIT License (see `LICENSE`)

**Datasets:** The datasets themselves are covered by their own open-access licenses:
- NEU-DET: Available from http://faculty.neu.edu.cn/songkechen/
- GC10-DET: CC BY 4.0
- Severstal: Kaggle competition terms

---

## Support

- **Questions?** Contact Hazem Elerefy (team lead)
- **Supervisor:** Dr. Tarek Ghoneimy
- **Full project context:** See `PROJECT_GUIDE.md`

---

## Acknowledgments

- Ultralytics for YOLO framework
- Original paper authors: Song & Yan (NEU-DET), Lv et al. (GC10-DET), Severstal PJSC
- The 11-paper Tier-1 corpus reviewed in Chapter 2
