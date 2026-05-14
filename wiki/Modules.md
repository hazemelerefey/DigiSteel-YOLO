# Modules

## Top-Level Layout

- [digisteel/](file:///workspace/digisteel): installable Python package
- [configs/](file:///workspace/configs): Ultralytics dataset YAMLs (NEU-DET variants)
- [tests/](file:///workspace/tests): unit tests for A2/A3
- [.github/workflows/](file:///workspace/.github/workflows): CI (lint/tests) and release automation

## `digisteel` Package

### `digisteel/__init__.py`

Acts as the library entrypoint and defines the public API surface:

- Re-exports:
  - `GhostConv` from [ghost_conv.py](file:///workspace/digisteel/modules/ghost_conv.py)
  - `InnerWIoULoss` from [inner_wiou.py](file:///workspace/digisteel/modules/inner_wiou.py)

Reference: [digisteel/__init__.py](file:///workspace/digisteel/__init__.py#L28-L37)

### `digisteel/modules/` (Core Implementation)

#### [ghost_conv.py](file:///workspace/digisteel/modules/ghost_conv.py)

Implements A2 GhostConv building blocks:

- `GhostModule`: “primary conv” + cheap depthwise conv to synthesize extra channels, then concatenate.
- `GhostConv`: a thin wrapper to use `GhostModule` like a typical Conv block (drop-in at call site).
- `GhostConvWeightSharing`: a wrapper that holds a single shared `GhostModule` instance; reusing it across stages is how you reduce parameters when applied in multiple places.

External dependencies: `torch`, `torch.nn`, `torch.nn.functional`.

#### [inner_wiou.py](file:///workspace/digisteel/modules/inner_wiou.py)

Implements A3 Inner-WIoU regression loss components:

- `iou(box1, box2)`: IoU computation for boxes in `[x1, y1, x2, y2]`.
- `inner_iou_loss(pred_boxes, target_boxes)`: Inner-IoU-style loss term.
- `wiou_v3_loss(pred_boxes, target_boxes)`: WIoU v3-style dynamically weighted IoU loss term.
- `InnerWIoULoss`: `nn.Module` combining `inner_iou_loss` and `wiou_v3_loss` using `lambda_weight`.

External dependencies: `torch`, `torch.nn`, `torch.nn.functional`.

#### [modules/__init__.py](file:///workspace/digisteel/modules/__init__.py)

Currently only contains a docstring and does not import/re-export module symbols. Several docs and tests use `from digisteel.modules import GhostConv`, which will only work if `digisteel/modules/__init__.py` re-exports those symbols (or if callers import via `digisteel` instead).

### Stub Namespaces (Placeholders)

The following packages exist only as placeholders and currently export no functionality beyond a docstring:

- [digisteel/data](file:///workspace/digisteel/data)
- [digisteel/perturbations](file:///workspace/digisteel/perturbations)
- [digisteel/eval](file:///workspace/digisteel/eval)
- [digisteel/export](file:///workspace/digisteel/export)

## `configs/` (Dataset Configs)

The YAMLs in [configs/](file:///workspace/configs) define dataset location + class names and document which architectural variant a training script would apply:

- [yolov11n_baseline.yaml](file:///workspace/configs/yolov11n_baseline.yaml)
- [yolov11n_a2_ghostconv.yaml](file:///workspace/configs/yolov11n_a2_ghostconv.yaml)
- [yolov11n_a3_innerwiou.yaml](file:///workspace/configs/yolov11n_a3_innerwiou.yaml)
- [yolov11n_a2_a3.yaml](file:///workspace/configs/yolov11n_a2_a3.yaml)

## `tests/` (Unit Tests)

- [test_ghost_conv.py](file:///workspace/tests/test_ghost_conv.py): shape/stride/backprop checks for A2 modules.
- [test_inner_wiou.py](file:///workspace/tests/test_inner_wiou.py): correctness/backprop checks for A3 losses.
- [test_perturbations.py](file:///workspace/tests/test_perturbations.py): placeholder for future robustness toolkit.
- [conftest.py](file:///workspace/tests/conftest.py): shared test fixtures.

