# DigiSteel-YOLO Colab Notebook

## Quick Start

### Step 1: Open in Google Colab

Click the button below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hazemelerefey/DigiSteel-YOLO/blob/main/notebooks/DigiSteel_YOLO_Colab.ipynb)

Or manually:
1. Go to https://colab.research.google.com
2. Click "Upload" tab
3. Upload `notebooks/DigiSteel_YOLO_Colab.ipynb`

### Step 2: Enable GPU

1. Go to **Runtime** → **Change runtime type**
2. Select **T4 GPU** (or better)
3. Click **Save**

### Step 3: Run the Notebook

Run each cell in order by clicking the play button or pressing `Shift+Enter`.

## What the Notebook Does

### 1. Environment Setup
- Installs all dependencies (Ultralytics, PyTorch, OpenCV, etc.)
- Configures Kaggle API for dataset download
- Verifies GPU availability

### 2. Repository Setup
- Clones the DigiSteel-YOLO repository
- Installs the package in development mode
- Creates necessary directories

### 3. Dataset Preparation
- Downloads NEU-DET (1,800 images, 6 classes) from Kaggle
- Downloads GC10-DET (2,294 images, 10 classes) from Kaggle
- Converts annotations to YOLO format
- Splits into train/val/test sets

### 4. Baseline Training
- Trains YOLOv11n baseline on NEU-DET
- 200 epochs with early stopping
- Saves best weights

### 5. DigiSteel-YOLO Training
- Trains YOLOv11n with GhostConv backbone
- Same hyperparameters as baseline
- Saves best weights

### 6. Evaluation
- Evaluates both models on validation set
- Compares results with 11 reference papers
- Creates comparison table

### 7. ONNX Export
- Exports best model to ONNX format
- Verifies the exported model
- Prepares for edge deployment

## Expected Results

| Model | Dataset | mAP@0.5 | Params (M) |
|---|---|---|---|
| YOLOv11n Baseline | NEU-DET | ~77-82% | 2.59 |
| DigiSteel-YOLO | NEU-DET | ~80-85% | 2.59 |
| P10 KDM-YOLO (best paper) | NEU-DET | 95.4% | 3.29 |

## Training Time

| GPU | Time per Epoch | Total Time (200 epochs) |
|---|---|---|
| T4 (Colab Free) | ~30-40 seconds | ~2-3 hours |
| V100 (Colab Pro) | ~15-20 seconds | ~1-1.5 hours |
| A100 | ~10-15 seconds | ~45 minutes |
| CPU only | ~15-20 minutes | ~2-3 days |

## Troubleshooting

### "No GPU detected"
- Go to Runtime → Change runtime type → Select T4 GPU

### "Kaggle API error"
- Make sure your Kaggle credentials are correct
- Check if the datasets are available

### "Out of memory"
- Reduce batch size (e.g., from 16 to 8)
- Reduce image size (e.g., from 640 to 320)

### "Training too slow"
- Make sure GPU is enabled
- Reduce number of epochs for testing

## Files Generated

After running the notebook, you'll have:

```
runs/
├── baseline_neu_det_seed42/
│   └── weights/
│       ├── best.pt      # Best baseline model
│       └── last.pt      # Last checkpoint
└── digisteel_neu_det_seed42/
    └── weights/
        ├── best.pt      # Best DigiSteel model
        └── last.pt      # Last checkpoint

digisteel-yolo.onnx      # ONNX model for deployment
evals/
└── *.csv                # Evaluation results
```

## Next Steps

1. **Download Results** - Use the last cell to download all results
2. **Analyze Results** - Review the comparison table
3. **Run Robustness Evaluation** - Test under perturbations
4. **Deploy** - Use ONNX model on edge devices
5. **Write Paper** - Document findings

## Support

For questions or issues:
- **Team Lead:** Hazem Elerefy
- **GitHub:** https://github.com/hazemelerefey/DigiSteel-YOLO
- **Issues:** https://github.com/hazemelerefey/DigiSteel-YOLO/issues
