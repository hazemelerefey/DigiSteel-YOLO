#!/bin/bash
#############################################################################
# DigiSteel-YOLO Environment Setup
# Run this script to set up the development environment
#############################################################################

set -e

echo "=========================================="
echo "  DigiSteel-YOLO Environment Setup"
echo "=========================================="

# Check Python version
echo "[1/6] Checking Python version..."
python3 --version || { echo "ERROR: Python 3 is required"; exit 1; }

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install -q ultralytics torch torchvision opencv-python albumentations
pip install -q numpy pandas matplotlib seaborn scikit-learn tqdm
pip install -q onnx onnxruntime pyyaml

# Install development dependencies
echo "[3/6] Installing development dependencies..."
pip install -q pytest pytest-cov black ruff

# Install the package in development mode
echo "[4/6] Installing DigiSteel-YOLO package..."
pip install -e .

# Verify installation
echo "[5/6] Verifying installation..."
python3 -c "from ultralytics import YOLO; print('✓ Ultralytics YOLO ready')"
python3 -c "from digisteel.modules import GhostConv, InnerWIoULoss; print('✓ DigiSteel modules ready')"
python3 -c "from digisteel.perturbations import PerturbationSuite; print('✓ Perturbation toolkit ready')"
python3 -c "from digisteel.eval import RobustnessSweep; print('✓ Evaluation framework ready')"

# Create necessary directories
echo "[6/6] Creating project directories..."
mkdir -p datasets/NEU-DET
mkdir -p datasets/GC10-DET
mkdir -p runs
mkdir -p evals
mkdir -p weights

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download datasets: bash tools/download_datasets.sh"
echo "  2. Train baseline:    python scripts/train_baseline.py"
echo "  3. Train DigiSteel:   python scripts/train_digisteel.py"
echo ""
echo "For more info, see README.md"
