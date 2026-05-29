# Architecture

## High-Level View

The repository is structured as a Python package (`digisteel`) plus configuration and testing assets around it.

- **Library layer:** `digisteel/` contains reusable code (modules, perturbations, evaluation).
- **Configuration layer:** `configs/` contains experiment config YAMLs.
- **Quality layer:** `tests/` validates core primitives and ensures imports work in CI.
- **Automation layer:** `.github/workflows/` runs ruff, black, and pytest on PRs/pushes.

## Current Architecture (What Exists Today)

```mermaid
flowchart TD
  Repo["DigiSteel-YOLO repo"] --> Pkg["digisteel (python package)"]
  Repo --> Cfg["configs (yolov11n_*.yaml)"]
  Repo --> Tests["tests (pytest)"]
  Repo --> CI[".github/workflows (CI)"]

  Pkg --> Mods["digisteel/modules"]
  Mods --> Ghost["ghost_conv.py (GhostConv)"]
  Mods --> Loss["inner_wiou.py (Inner-WIoU)"]

  Pkg --> Pert["digisteel/perturbations"]
  Pert --> Blur["blur.py"]
  Pert --> Noise["noise.py"]
  Pert --> Bright["brightness.py"]
  Pert --> JPEG["jpeg.py"]
  Pert --> Suite["suite.py (PerturbationSuite)"]

  Pkg --> Eval["digisteel/eval"]
  Eval --> Metrics["metrics.py"]
  Eval --> Sweep["robustness_sweep.py (RobustnessSweep)"]

  Pkg --> StubData["digisteel/data (stub)"]
  Pkg --> StubExport["digisteel/export (stub)"]
```

## Robustness Evaluation Flow (Core Contribution)

The primary workflow is the robustness evaluation pipeline:

```mermaid
flowchart LR
  Model["Trained YOLO Model"] --> Sweep["RobustnessSweep"]
  Data["Dataset Images"] --> Suite["PerturbationSuite"]
  Suite --> |"24 configs"| Sweep
  Sweep --> |"evaluate each"| Results["SweepResult (192 data points)"]
  Results --> CSV["CSV/JSON export"]
  Results --> Analysis["Robustness analysis"]
```

### Perturbation Pipeline

```mermaid
flowchart LR
  Image["Clean Image"] --> P1["Gaussian Blur (4 levels)"]
  Image --> P2["Motion Blur (4 levels)"]
  Image --> P3["Gaussian Noise (4 levels)"]
  Image --> P4["Brightness Shift (4 levels)"]
  Image --> P5["Contrast Reduction (4 levels)"]
  Image --> P6["JPEG Compression (4 levels)"]
  P1 --> Eval["Model Evaluation"]
  P2 --> Eval
  P3 --> Eval
  P4 --> Eval
  P5 --> Eval
  P6 --> Eval
```

## Key Design Decisions

### Modules: Proven Techniques, Not Novel Claims

- `GhostConv` implements the Ghost convolution from Han et al. (CVPR 2020). It is a proven lightweight convolution technique, not our invention.
- `InnerWIoULoss` combines Inner-IoU (Zhang 2023) and WIoU v3 (Tong 2023). It is a principled combination of existing losses, not our invention.
- **The weight-sharing variant (GhostConvWeightSharing) was removed** because sharing weights across pyramid stages (P3/P4/P5) is architecturally unsound — different scales need different feature extractors.

### Perturbations: The Core Contribution

- `PerturbationSuite` provides a unified interface for 6 perturbation types x 4 severity levels.
- Each perturbation simulates a real-world industrial image degradation.
- The suite is designed for reproducibility (seeded noise, deterministic transforms).

### Evaluation: Standardized and Reproducible

- `RobustnessSweep` runs the full evaluation pipeline (24 configs per model per dataset).
- Results are exported as CSV/JSON with 8 metrics per evaluation point.
- The framework supports any YOLO model loaded via Ultralytics.

## Entry Points

- Package exports: [digisteel/__init__.py](../digisteel/__init__.py)
  - `GhostConv`, `GhostModule`
  - `InnerWIoULoss`, `inner_iou_loss`, `wiou_v3_loss`
- Perturbation toolkit: [digisteel/perturbations/__init__.py](../digisteel/perturbations/__init__.py)
  - `PerturbationSuite`, `GaussianBlur`, `MotionBlur`, `GaussianNoise`, `BrightnessShift`, `ContrastReduction`, `JPEGCompression`
- Evaluation framework: [digisteel/eval/__init__.py](../digisteel/eval/__init__.py)
  - `RobustnessSweep`, `compute_metrics`

For deeper details, see:

- [Modules](Modules.md)
- [API](API.md)
