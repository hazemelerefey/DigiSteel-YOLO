<!-- DigiSteel-YOLO Banner -->
<div align="center">

# DigiSteel-YOLO

### <div align="center">Comprehensive Robustness Study of Lightweight YOLO Detectors for Steel Surface Defect Detection</div>

<div align="center">

**The first systematic evaluation of YOLO detector robustness to real-world image degradations in industrial steel inspection.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLO](https://img.shields.io/badge/YOLO-v11-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)

[Quickstart](#-quick-start) • [Documentation](#-documentation) • [Robustness Framework](#-robustness-evaluation-framework) • [Citation](#-citation)

</div>

---

## About

**DigiSteel-YOLO** is a comprehensive robustness study for YOLO-based steel surface defect detectors. While existing research focuses on accuracy benchmarks (mAP on clean images), **no prior work systematically evaluates how these detectors perform under real-world industrial image degradations**.

### Research Question

> *How robust are lightweight YOLO detectors to real-world image degradations in steel surface defect detection, and can perturbation-aware training improve deployment reliability?*

### Key Contributions

1. **Standardized Robustness Evaluation Framework** — 6 perturbation types x 4 severity levels = 24 evaluation points per model per dataset
2. **Perturbation-Aware Training Protocol** — Training with injected degradations to study robustness-accuracy tradeoffs
3. **Multi-Dataset Validation** — Identical hyperparameters across NEU-DET and GC10-DET
4. **Open-Source Benchmark** — Reproducible evaluation toolkit for the community

### Why This Matters

Steel defect detectors are deployed in harsh industrial environments where image quality degrades due to:
- Camera defocus and lens contamination
- Sensor noise and electrical interference
- Lighting variation (over/underexposure)
- Image compression during transmission
- Environmental interference (fog, dust, vibration)

**A detector that achieves 90% mAP on clean images but drops to 60% under blur is not production-ready.** This study quantifies these degradation patterns and proposes training strategies to mitigate them.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hazemelerefey/DigiSteel-YOLO.git
cd DigiSteel-YOLO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run Robustness Evaluation

```python
from digisteel.perturbations import PerturbationSuite
from digisteel.eval import RobustnessSweep

# Initialize perturbation suite (6 types x 4 levels = 24 configs)
suite = PerturbationSuite()
print(suite.summary())

# Run robustness sweep on a trained model
sweep = RobustnessSweep(model_path="runs/baseline/weights/best.pt")
results = sweep.run(dataset_path="datasets/NEU-DET/yolo", dataset_name="NEU-DET")
sweep.save_results(results, "evals/baseline_robustness.csv")
```

### Apply Perturbations to Images

```python
import cv2
from digisteel.perturbations import GaussianBlur, GaussianNoise, JPEGCompression

# Load an image
image = cv2.imread("steel_sample.jpg")

# Apply different perturbations
blurred = GaussianBlur(level=2).apply(image)      # sigma=3
noisy = GaussianNoise(level=3, seed=42).apply(image)  # sigma=0.20
compressed = JPEGCompression(level=2).apply(image)  # quality=50
```

---

## Robustness Evaluation Framework

### Perturbation Matrix

| Perturbation | Level 1 (Mild) | Level 2 (Moderate) | Level 3 (Severe) | Level 4 (Extreme) |
|---|---|---|---|---|
| **Gaussian Blur** | sigma=1 | sigma=3 | sigma=5 | sigma=7 |
| **Motion Blur** | k=3 | k=5 | k=7 | k=9 |
| **Gaussian Noise** | sigma=0.05 | sigma=0.10 | sigma=0.20 | sigma=0.30 |
| **Brightness Shift** | delta=-30 | delta=-50 | delta=+30 | delta=+50 |
| **Contrast Reduction** | factor=0.8 | factor=0.6 | factor=0.4 | factor=0.2 |
| **JPEG Compression** | quality=80 | quality=50 | quality=30 | quality=15 |

### Metrics (8 per evaluation point)

| Metric | Description |
|---|---|
| mAP@0.5 | Mean Average Precision at IoU=0.5 |
| mAP@0.5:0.95 | COCO mAP averaged over IoU 0.5..0.95 |
| Precision | Positive predictive value |
| Recall | True positive rate |
| F1 | Harmonic mean of precision and recall |
| FPS | Frames per second (inference speed) |
| Parameters (M) | Model parameter count in millions |
| GFLOPs | Floating-point operations per inference |

### Output

The framework produces a CSV/JSON with **192 data points per model per dataset** (24 perturbation configs x 8 metrics), enabling:
- Per-perturbation degradation curves
- Cross-model robustness comparison
- Robustness-accuracy Pareto analysis
- Per-defect-class vulnerability analysis

---

## Datasets

| Dataset | Images | Classes | Resolution | Source |
|---|---|---|---|---|
| **NEU-DET** | 1,800 | 6 | 200x200 grayscale | Northeastern University |
| **GC10-DET** | 2,294 | 10 | 2048x1000 grayscale | Lv et al. |

Both datasets are evaluated with **identical hyperparameters** to ensure fair cross-dataset comparison.

---

## Model Variants

We evaluate multiple YOLO configurations to study robustness across architectures:

| Variant | Base | Modification | Purpose |
|---|---|---|---|
| **Baseline** | YOLOv11n | None | Reference performance |
| **GhostConv** | YOLOv11n | GhostConv backbone | Lightweight variant |
| **Inner-WIoU** | YOLOv11n | Inner-WIoU loss | Improved box regression |
| **GhostConv + Inner-WIoU** | YOLOv11n | Both modifications | Combined lightweight variant |

### GhostConv (Lightweight Backbone)

Drop-in replacement for standard Conv2d using Ghost convolutions (Han et al., CVPR 2020). Reduces parameters by ~50% while maintaining accuracy.

```python
from digisteel.modules import GhostConv
# Use as replacement for torch.nn.Conv2d
```

### Inner-WIoU (Composite Loss)

Combines Inner-IoU (Zhang 2023) and WIoU v3 (Tong 2023) for improved bounding box regression.

```python
from digisteel.modules import InnerWIoULoss
loss_fn = InnerWIoULoss(lambda_weight=0.5)
loss = loss_fn(pred_boxes, target_boxes)
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA 12.x (or CPU-only)
- Ultralytics YOLO
- OpenCV, NumPy, Albumentations

### Install from Source

```bash
git clone https://github.com/hazemelerefey/DigiSteel-YOLO.git
cd DigiSteel-YOLO
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "from digisteel.perturbations import PerturbationSuite; print(PerturbationSuite().summary())"
python -c "from digisteel.modules import GhostConv, InnerWIoULoss; print('Modules OK')"
```

---

## Repository Structure

```
DigiSteel-YOLO/
├── digisteel/                        # Main package
│   ├── modules/
│   │   ├── ghost_conv.py            # GhostConv lightweight backbone
│   │   └── inner_wiou.py            # Inner-WIoU composite loss
│   ├── perturbations/               # Robustness evaluation toolkit
│   │   ├── blur.py                  # Gaussian & motion blur
│   │   ├── noise.py                 # Gaussian noise
│   │   ├── brightness.py            # Brightness & contrast
│   │   ├── jpeg.py                  # JPEG compression
│   │   └── suite.py                 # Unified perturbation interface
│   ├── eval/
│   │   ├── metrics.py               # Detection metrics computation
│   │   └── robustness_sweep.py      # Systematic evaluation framework
│   ├── data/                        # Dataset loaders
│   └── export/                      # Model export (ONNX)
│
├── configs/                         # YOLO configurations
│   ├── yolov11n_baseline.yaml
│   ├── yolov11n_a2_ghostconv.yaml
│   ├── yolov11n_a3_innerwiou.yaml
│   └── yolov11n_a2_a3.yaml
│
├── tests/                           # Unit tests
│   ├── test_ghost_conv.py
│   ├── test_inner_wiou.py
│   └── test_perturbations.py
│
├── .github/workflows/               # CI/CD
│   ├── test.yml
│   └── release.yml
│
├── README.md                        # This file
├── PROJECT_GUIDE.md                 # Full project context
├── CONTRIBUTING.md                  # Team collaboration guide
├── LICENSE                          # MIT
└── requirements.txt                 # Dependencies
```

---

## Contributing

See `CONTRIBUTING.md` for team collaboration guidelines.

### Team Members

| Name | Role | GitHub |
|---|---|---|
| **Hazem Elerefy** | Lead, WP1 | [@hazemelerefey](https://github.com/hazemelerefey) |
| **Youssef Sherif** | WP2 | — |
| **Mohamed Salah** | WP3 | — |
| **Moamen Esmat** | WP4 | — |
| **Mahmoud Hisham** | WP5 | — |

---

## License

**Code:** MIT License (see `LICENSE`)

**Datasets:** Open-access under their respective licenses.

---

## Citation

If you use DigiSteel-YOLO in your research, please cite:

```bibtex
@software{digisteel2026,
  title={DigiSteel-YOLO: Comprehensive Robustness Study of Lightweight YOLO Detectors for Steel Surface Defect Detection},
  author={Elerefy, Hazem and Sherif, Youssef and Salah, Mohamed and Esmat, Moamen and Hisham, Mahmoud},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/hazemelerefey/DigiSteel-YOLO}},
  note={Graduation Project, Digilians (MCIT) Specialized Diploma in Applied AI \& Data Analytics}
}
```

---

## Contact & Support

- **Team Lead:** Hazem Elerefy
- **Supervisor:** Dr. Tarek Ghoneimy
- **Program:** Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics
- **Questions?** See `CONTRIBUTING.md` or open an issue on GitHub

---

## Acknowledgments

- **Ultralytics** — YOLO framework and ecosystem
- **Han et al. (CVPR 2020)** — GhostNet / GhostConv
- **Zhang et al. (2023)** — Inner-IoU loss (arXiv:2311.02877)
- **Tong et al. (2023)** — WIoU v3 loss (arXiv:2301.10051)
- **Song & Yan** — NEU-DET dataset
- **Lv et al.** — GC10-DET dataset

---

<div align="center">

**Built by the DigiSteel Team**

![Stars](https://img.shields.io/github/stars/hazemelerefey/DigiSteel-YOLO?style=social)
![Forks](https://img.shields.io/github/forks/hazemelerefey/DigiSteel-YOLO?style=social)

</div>
