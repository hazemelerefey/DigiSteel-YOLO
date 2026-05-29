#!/bin/bash
#############################################################################
# DigiSteel-YOLO Complete Pipeline
# Runs the entire training and evaluation pipeline
#############################################################################

set -e

echo "=========================================="
echo "  DigiSteel-YOLO Complete Pipeline"
echo "=========================================="
echo ""

# Configuration
DATASET="NEU-DET"
EPOCHS=200
SEED=42
IMG_SIZE=640
BATCH_SIZE=16

# Step 1: Setup environment
echo "[Step 1/8] Setting up environment..."
bash setup.sh

# Step 2: Download datasets
echo ""
echo "[Step 2/8] Downloading datasets..."
bash tools/download_datasets.sh

# Step 3: Convert to YOLO format
echo ""
echo "[Step 3/8] Converting annotations to YOLO format..."
python3 tools/voc_to_yolo.py --dataset NEU-DET
python3 tools/voc_to_yolo.py --dataset GC10-DET

# Step 4: Split datasets
echo ""
echo "[Step 4/8] Splitting datasets..."
python3 tools/split_dataset.py --dataset NEU-DET --seed $SEED
python3 tools/split_dataset.py --dataset GC10-DET --seed $SEED

# Step 5: Train baseline
echo ""
echo "[Step 5/8] Training YOLOv11n baseline..."
python3 scripts/train_baseline.py --dataset $DATASET --epochs $EPOCHS --seed $SEED

# Step 6: Train DigiSteel-YOLO (GhostConv + Inner-WIoU)
echo ""
echo "[Step 6/8] Training DigiSteel-YOLO..."
python3 scripts/train_digisteel.py --dataset $DATASET --epochs $EPOCHS --seed $SEED

# Step 7: Evaluate robustness
echo ""
echo "[Step 7/8] Running robustness evaluation..."
python3 scripts/eval_robustness.py --model runs/baseline_${DATASET,,}_seed${SEED}/weights/best.pt --dataset $DATASET
python3 scripts/eval_robustness.py --model runs/digisteel_${DATASET,,}_seed${SEED}/weights/best.pt --dataset $DATASET

# Step 8: Export to ONNX
echo ""
echo "[Step 8/8] Exporting to ONNX..."
python3 scripts/export_onnx.py --model runs/digisteel_${DATASET,,}_seed${SEED}/weights/best.pt --output digisteel-yolo.onnx

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Baseline: runs/baseline_${DATASET,,}_seed${SEED}/"
echo "  DigiSteel: runs/digisteel_${DATASET,,}_seed${SEED}/"
echo "  Robustness: evals/"
echo "  ONNX: digisteel-yolo.onnx"
echo ""
echo "Run comparison: python scripts/compare_models.py"
