# DigiSteel-YOLO Experiments

## 📊 Results Summary

| # | Experiment | mAP50 | mAP50-95 | Model | Notes |
|---|------------|-------|----------|-------|-------|
| 01 | Baseline YOLOv11n v1 | - | - | yolov11n | Initial baseline |
| 02 | Data Analysis | - | - | - | EDA only |
| 03 | Digisteel Complete | - | - | yolov11n | Full pipeline |
| 04 | Digisteel Model | - | - | yolov11n | Architecture study |
| 05 | Week 2 Ablation | - | - | yolov11n | Hyperparameter tuning |
| 06 | Week 3 A1 Config Fix | 0.748 | 0.416 | yolov11n | Config optimization |
| 07 | Week 3 A2 Arch Fix | 0.734 | 0.374 | yolov11n | Architecture changes |
| 08 | Week 4 Fresh Baseline | 0.788 | 0.452 | yolov11n | Clean baseline |
| 09 | Baseline YOLOv11n v2 | 0.788 | 0.452 | yolov11n | Updated baseline |
| 10 | DAFE v4 | 0.803 | 0.442 | yolov11n+DAFE | Feature enhancement |
| 11 | TTA Eval | - | - | yolov11n | Test-time augmentation |
| 12 | Transfer Learning YOLOv11n | 0.794 | 0.440 | yolov11n | COCO pretrained |
| 13 | Transfer Learning Fresh | 0.786 | 0.433 | yolov11n | Transfer learning |
| 14 | TL DAFE Comparison | - | - | yolov11n | TL + DAFE comparison |
| 15 | YOLOv26n NEU-DET | **0.810** | 0.431 | **yolov26n** | **Best mAP50** |
| 16 | YOLOv26n Transfer Learning | - | - | yolov26n | TL with YOLOv26n |
| 17 | EDA v1 | - | - | - | Generic EDA |
| 18 | EDA v2 | - | - | - | Problem-specific EDA |

## 🏆 Best Results

- **Best mAP50**: 0.810 (YOLOv26n) — Experiment 15
- **Best mAP50-95**: 0.452 (YOLOv11n) — Experiment 08/09
- **Target**: 0.830 mAP50-95

## 📁 Folder Structure

Each experiment folder contains:
```
XX_experiment_name/
├── notebook.ipynb    # Jupyter notebook
├── results/          # JSON results, plots
├── weights/          # Model weights
└── plots/            # Visualizations
```

## 🔧 Running Experiments

1. Open the notebook in the experiment folder
2. Run all cells
3. Results auto-save to `results/` folder
4. Weights save to `weights/` folder

## 📈 Tracking Results

Run the master tracker:
```bash
python tools/results_tracker.py
```

## 🧹 Memory Management

For RAM/VRAM issues:
```bash
python tools/memory_utils.py
```

Or in notebooks:
```python
from tools.memory_utils import clear_memory, get_safe_batch_size
clear_memory()
batch_size = get_safe_batch_size(imgsz=640, model_size='n')
```
