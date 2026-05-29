# Modules

## Package: `digisteel`

### `digisteel/__init__.py`

Purpose:

- Defines package metadata (`__version__`, `__author__`)
- Re-exports the primary public API symbols:
  - `GhostConv`, `GhostModule` — Lightweight convolution modules
  - `InnerWIoULoss`, `inner_iou_loss`, `wiou_v3_loss` — Composite loss functions

See: [digisteel/__init__.py](../digisteel/__init__.py)

### `digisteel/modules/`

Purpose:

- Hosts proven lightweight modules used as model variants in the robustness study.
- **These are NOT claimed as novel contributions** — they are established techniques applied to steel defect detection.

Contents:

- GhostConv module: [ghost_conv.py](../digisteel/modules/ghost_conv.py) — Ghost convolution (Han et al., CVPR 2020)
- Inner-WIoU loss: [inner_wiou.py](../digisteel/modules/inner_wiou.py) — Composite loss (Zhang 2023 + Tong 2023)

### `digisteel/perturbations/`

Purpose:

- **The core contribution of DigiSteel-YOLO.** Provides standardized image degradations for robustness evaluation.
- 6 perturbation types x 4 severity levels = 24 evaluation points.

Contents:

- `blur.py` — Gaussian blur and motion blur
- `noise.py` — Gaussian noise
- `brightness.py` — Brightness shift and contrast reduction
- `jpeg.py` — JPEG compression
- `suite.py` — Unified `PerturbationSuite` interface

Usage:
```python
from digisteel.perturbations import PerturbationSuite
suite = PerturbationSuite()
degraded = suite.apply(image, "gaussian_blur", level=2)
```

### `digisteel/eval/`

Purpose:

- Evaluation framework for systematic robustness sweeps.
- Computes 8 metrics per evaluation point (mAP, precision, recall, F1, FPS, params, GFLOPs, inference time).

Contents:

- `metrics.py` — Detection metrics computation
- `robustness_sweep.py` — `RobustnessSweep` class for running full evaluation sweeps

Usage:
```python
from digisteel.eval import RobustnessSweep
sweep = RobustnessSweep("runs/baseline/weights/best.pt")
results = sweep.run("datasets/NEU-DET/yolo", "NEU-DET")
sweep.save_results(results, "evals/baseline_robustness.csv")
```

### `digisteel/data/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility:

- Dataset download wrappers, dataset indexing, dataset loader(s), and transforms.

### `digisteel/export/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility:

- Export utilities (e.g., ONNX export + verification).

## Configurations: `configs/`

The repository includes YAML configs for training variants:

- Baseline: `configs/yolov11n_baseline.yaml`
- GhostConv variant: `configs/yolov11n_a2_ghostconv.yaml`
- Inner-WIoU variant: `configs/yolov11n_a3_innerwiou.yaml`
- Combined variant: `configs/yolov11n_a2_a3.yaml`

These are dataset configs; architectural changes are applied in training scripts.

## Tests: `tests/`

Unit tests validate:

- `GhostConv` output shape, stride behavior, parameter count, and backprop
- `InnerWIoULoss` returns finite scalar losses and backprop works
- `PerturbationSuite` applies all 24 perturbation configs correctly

See:

- [test_ghost_conv.py](../tests/test_ghost_conv.py)
- [test_inner_wiou.py](../tests/test_inner_wiou.py)
- [test_perturbations.py](../tests/test_perturbations.py)

## CI: `.github/workflows/`

The CI pipeline performs:

- `ruff check .`
- `black --check .`
- `pytest tests/ -v --cov=digisteel --cov-report=xml`

See: [test.yml](../.github/workflows/test.yml)
