# DAFEGate v4 — Self-Contained Experiment

**Result:** 82.0% mAP@0.5 | 46.8% mAP@0.5:0.95 on NEU-DET (6 classes)

## What's here

| Path | Description |
|------|-------------|
| `modules/dafe.py` | DAFEGate module (EdgeAwareConv + TextureBranch + DAFEGate) |
| `configs/yolov11n_dafegate.yaml` | YOLOv11n backbone with DAFEGate at P3 |
| `configs/neu_det.yaml` | NEU-DET dataset config (6 classes) |
| `yolo11n.pt` | COCO pretrained YOLOv11n weights |
| `runs/best.pt` | Trained DAFEGate v4 weights (82.0% mAP@0.5) |
| `evals/dafegate_v4_summary.json` | Test-set metrics JSON |
| `docs/DAFE-ABLATION-REPORT.md` | Full ablation study report |
| `exp_dafegate_evaluation.ipynb` | Notebook: build → train → eval → export |

## Quick start

```bash
cd dafegate
# activate your venv first
jupyter notebook exp_dafegate_evaluation.ipynb
```

Run cells 1→5 sequentially. Training takes ~1.5 hours on RTX 2000 Ada (16 GB).

## Dependencies

- Python 3.11+, PyTorch 2.6+, Ultralytics 8.4+
- GPU with ≥8 GB VRAM (batch=32 needs ~6 GB)
- Dataset: `datasets/NEU-DET/yolo/` (relative path in `configs/neu_det.yaml`)

## Reproduce without training

Skip cell 3 (training). In cell 4, point `best_pt` to `runs/best.pt`:

```python
best_pt = ROOT / "runs" / "best.pt"
```

This evaluates the saved weights directly — no training needed.

## Architecture

```
Input x: (B, C, H, W) at P3 (C=256, H=W=80)
  ├── EdgeAwareConv(x)  → (B, 128, 80, 80)  [Sobel-X/Y init]
  ├── TextureBranch(x)  → (B, 128, 80, 80)  [local variance]
  ├── Concat → (B, 256, 80, 80)
  ├── Channel Attention (SE, r=8)
  ├── Fusion (1×1 Conv + BN + SiLU)
  └── y = x + sigmoid(α) · enhanced   [α init: -2.2 → 0.1]
```

## Per-class results

| Class | AP@0.5 |
|-------|--------|
| crazing | 49.1% |
| inclusion | 88.3% |
| patches | 91.7% |
| pitted_surface | 85.0% |
| rolled-in_scale | 78.8% |
| scratches | 98.9% |
