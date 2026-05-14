# Running

This page distinguishes between:

- What you can run in the current repo snapshot
- What the README documents but is not yet present (missing scripts/tools/notebooks)

## Setup (Recommended)

### Option A: One-command setup

```bash
bash setup.sh
```

This creates a `venv/`, installs dependencies, creates runtime folders, and validates imports.

See: [setup.sh](../setup.sh)

### Option B: Manual setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

## Verify Imports

```bash
python -c "from digisteel import GhostConv, InnerWIoULoss; print('OK')"
```

## Run Unit Tests

```bash
pytest -v
```

CI runs the same tests plus formatting/lint checks (ruff + black).

## Lint / Format (Optional)

```bash
ruff check .
black --check .
```

## Config Smoke Check

```bash
python -c "import yaml; yaml.safe_load(open('configs/yolov11n_baseline.yaml'))"
```

## What’s Missing vs README

The README describes commands like:

- `python scripts/train_baseline.py ...`
- `python scripts/train_a2_a3.py ...`
- dataset tools under `tools/`
- notebooks under `notebooks/`

In this snapshot, those directories/files are not present. The current functional scope is the package modules + tests + configs.

If you plan to implement the training pipeline, recommended next steps:

- Add `scripts/` for train/eval/export entrypoints that integrate `digisteel.modules.*`
- Add `tools/` for dataset preparation
- Add `digisteel/data`, `digisteel/perturbations`, `digisteel/eval`, `digisteel/export` implementations (currently stubs)

## Where To Start Reading Code

- Package exports: [digisteel/__init__.py](../digisteel/__init__.py)
- A2: [ghost_conv.py](../digisteel/modules/ghost_conv.py)
- A3: [inner_wiou.py](../digisteel/modules/inner_wiou.py)
- Unit tests:
  - [test_ghost_conv.py](../tests/test_ghost_conv.py)
  - [test_inner_wiou.py](../tests/test_inner_wiou.py)
