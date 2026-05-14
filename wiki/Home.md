# Home

## What This Repo Is

DigiSteel-YOLO is a Python package and repository skeleton for a graduation project: real-time steel surface defect detection using a lightweight YOLO variant with two core innovations:

- **A2 — GhostConv weight-sharing**: parameter reduction by reusing one Ghost block across multiple pyramid stages.
- **A3 — Inner-WIoU loss**: composite regression loss mixing Inner-IoU and WIoU v3.

This wiki focuses on what is actually present in the current snapshot:

- `digisteel/` Python package with A2/A3 implementations
- `configs/` YAML configs intended for YOLO training variants
- `tests/` unit tests for A2/A3 modules
- CI workflow to run linting/tests

Several items referenced in the README (e.g., `scripts/`, `tools/`, `notebooks/`) are not present in this snapshot. The [Running](Running.md) page documents what you can run today vs what is still planned.

## Quick Navigation

- [Architecture](Architecture.md)
- [Modules](Modules.md)
- [API](API.md)
- [Dependencies](Dependencies.md)
- [Running](Running.md)

## Repository Map (Current Snapshot)

```
DigiSteel-YOLO/
├── digisteel/
│   ├── modules/
│   │   ├── ghost_conv.py
│   │   └── inner_wiou.py
│   ├── data/            # currently stub (__init__.py only)
│   ├── perturbations/   # currently stub (__init__.py only)
│   ├── eval/            # currently stub (__init__.py only)
│   └── export/          # currently stub (__init__.py only)
├── configs/
│   ├── yolov11n_baseline.yaml
│   ├── yolov11n_a2_ghostconv.yaml
│   ├── yolov11n_a3_innerwiou.yaml
│   └── yolov11n_a2_a3.yaml
├── tests/
│   ├── test_ghost_conv.py
│   ├── test_inner_wiou.py
│   └── test_perturbations.py
├── .github/workflows/test.yml
├── pyproject.toml
├── requirements.txt
└── setup.sh
```
