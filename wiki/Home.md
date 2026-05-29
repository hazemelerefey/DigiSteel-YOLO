# Home

## What This Repo Is

DigiSteel-YOLO is a **comprehensive robustness study** of lightweight YOLO detectors for steel surface defect detection. It is a graduation project for a 5-student team from the Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics.

### Research Question

> *How robust are lightweight YOLO detectors to real-world image degradations in steel surface defect detection, and can perturbation-aware training improve deployment reliability?*

### Core Contributions

1. **Standardized Robustness Evaluation Framework** — 6 perturbation types x 4 severity levels = 24 evaluation points per model per dataset
2. **Perturbation-Aware Training Study** — Training with injected degradations to study robustness-accuracy tradeoffs
3. **Multi-Dataset Validation** — Identical hyperparameters across NEU-DET and GC10-DET
4. **Open-Source Benchmark** — Reproducible evaluation toolkit for the community

## Quick Navigation

- [Architecture](Architecture.md) — Project structure and design decisions
- [Modules](Modules.md) — GhostConv and Inner-WIoU modules (proven techniques, not claimed as novel)
- [API](API.md) — Perturbation suite and evaluation framework API
- [Dependencies](Dependencies.md) — Required packages and versions
- [Running](Running.md) — How to run evaluations

## Repository Map (Current Snapshot)

```
DigiSteel-YOLO/
├── digisteel/
│   ├── modules/
│   │   ├── ghost_conv.py        # GhostConv lightweight backbone (CVPR 2020)
│   │   └── inner_wiou.py        # Inner-WIoU composite loss
│   ├── perturbations/           # Robustness evaluation toolkit
│   │   ├── blur.py              # Gaussian & motion blur
│   │   ├── noise.py             # Gaussian noise
│   │   ├── brightness.py        # Brightness & contrast
│   │   ├── jpeg.py              # JPEG compression
│   │   └── suite.py             # Unified perturbation interface
│   ├── eval/
│   │   ├── metrics.py           # Detection metrics computation
│   │   └── robustness_sweep.py  # Systematic evaluation framework
│   ├── data/                    # Dataset loaders
│   └── export/                  # Model export (ONNX)
├── configs/                     # YOLO configurations
│   ├── yolov11n_baseline.yaml
│   ├── yolov11n_a2_ghostconv.yaml
│   ├── yolov11n_a3_innerwiou.yaml
│   └── yolov11n_a2_a3.yaml
├── tests/                       # Unit tests
│   ├── test_ghost_conv.py
│   ├── test_inner_wiou.py
│   └── test_perturbations.py
├── .github/workflows/           # CI/CD
├── pyproject.toml
├── requirements.txt
└── setup.sh
```

## What's New (v0.2.0)

- **Removed** GhostConvWeightSharing class (architecturally incorrect — different pyramid scales need different features)
- **Added** Comprehensive perturbation toolkit (6 types x 4 levels)
- **Added** Robustness evaluation framework
- **Fixed** P09 EFEN-YOLOv8 reference (GitHub link was dead, module names were wrong)
- **Updated** All documentation to reflect the robustness-focused contribution
- **Updated** All module docstrings to be honest about what is/is not novel
