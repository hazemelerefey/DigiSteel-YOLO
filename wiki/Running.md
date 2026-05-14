# Running & Development

## Setup

### Option 1: One-Command Bootstrap

Run the repository bootstrap script:

```bash
bash setup.sh
```

It creates a venv, installs dependencies (including `.[dev]`), creates runtime directories, and performs a basic import check (see [setup.sh](file:///workspace/setup.sh)).

### Option 2: Manual Setup

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .[dev]
```

PyTorch (`torch`) is required by the A2/A3 modules; installation may require choosing the correct wheel for your CUDA/CPU environment (follow the official PyTorch install selector).

## Run Unit Tests

```bash
pytest -q
```

Tests cover:

- A2 GhostConv modules ([tests/test_ghost_conv.py](file:///workspace/tests/test_ghost_conv.py))
- A3 Inner-WIoU loss ([tests/test_inner_wiou.py](file:///workspace/tests/test_inner_wiou.py))

## Smoke-Import the Library

Recommended import path (matches what is actually exported today):

```bash
python -c "from digisteel import GhostConv, InnerWIoULoss; print('ok')"
```

Note:

- Several docs and tests use `from digisteel.modules import GhostConv`, but [digisteel/modules/__init__.py](file:///workspace/digisteel/modules/__init__.py) currently does not re-export `GhostConv` or `InnerWIoULoss`.

## Using the Dataset YAMLs

Dataset configs live under [configs/](file:///workspace/configs). They follow the Ultralytics dataset YAML format, and assume a local folder structure like:

`datasets/NEU-DET/yolo/{images,labels}/{train,val,test}/...`

Examples:

- Baseline dataset config: [yolov11n_baseline.yaml](file:///workspace/configs/yolov11n_baseline.yaml)
- Combined A2+A3 dataset config: [yolov11n_a2_a3.yaml](file:///workspace/configs/yolov11n_a2_a3.yaml)

## Training / Evaluation Scripts (Not Present in This Snapshot)

[README.md](file:///workspace/README.md) and [PROJECT_GUIDE.md](file:///workspace/PROJECT_GUIDE.md) reference commands like `python scripts/train_baseline.py ...`, plus dataset tooling under `tools/` and notebooks under `notebooks/`. Those directories are not present under [/workspace](file:///workspace) in the current snapshot, so training/evaluation is not runnable yet from this repo state.

