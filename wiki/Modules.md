# Modules

## Package: `digisteel`

### `digisteel/__init__.py`

Purpose:

- Defines package metadata (`__version__`, `__author__`)
- Re-exports the two primary public API symbols:
  - `GhostConv`
  - `InnerWIoULoss`

See: [digisteel/__init__.py](../digisteel/__init__.py)

### `digisteel/modules/`

Purpose:

- Hosts the core research/engineering contributions for DigiSteel-YOLO.

Contents:

- A2 GhostConv modules: [ghost_conv.py](../digisteel/modules/ghost_conv.py)
- A3 Inner-WIoU loss: [inner_wiou.py](../digisteel/modules/inner_wiou.py)

### `digisteel/data/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility (based on README naming):

- Dataset download wrappers, dataset indexing, dataset loader(s), and transforms.

### `digisteel/perturbations/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility:

- Robustness perturbations (blur/noise/brightness/jpeg) and severity control.

### `digisteel/eval/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility:

- Metric computation and reporting (the README mentions 8-metric reporting).

### `digisteel/export/` (stub)

Present but currently only contains `__init__.py`.

Intended responsibility:

- Export utilities (e.g., ONNX export + verification).

## Configurations: `configs/`

The repository includes YAML configs intended for training variants:

- Baseline: `configs/yolov11n_baseline.yaml`
- A2 only: `configs/yolov11n_a2_ghostconv.yaml`
- A3 only: `configs/yolov11n_a3_innerwiou.yaml`
- A2 + A3: `configs/yolov11n_a2_a3.yaml`

These are parsed in CI smoke tests to ensure they are valid YAML.

## Tests: `tests/`

Unit tests validate:

- `GhostConv` output shape, stride behavior, and backprop
- `GhostConvWeightSharing` works across multiple feature map shapes
- `InnerWIoULoss` returns finite scalar losses and backprop works

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
