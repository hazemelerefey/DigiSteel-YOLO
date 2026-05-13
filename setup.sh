#!/bin/bash

# DigiSteel-YOLO Complete Repository Setup Script
# Run this ONCE after cloning to get your environment ready

set -e

echo "=========================================="
echo "DigiSteel-YOLO Development Environment Setup"
echo "=========================================="
echo ""

# 1. Python virtual environment
echo "[1/5] Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate || . venv/Scripts/activate

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 3. Install dependencies
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt
pip install -e .[dev]

# 4. Create directories
echo "[4/5] Creating runtime directories..."
mkdir -p datasets runs evals weights figures

# 5. Verify installation
echo "[5/5] Verifying installation..."
python -c "
import torch, ultralytics, albumentations
print('✓ PyTorch:', torch.__version__)
print('✓ Ultralytics:', ultralytics.__version__)
print('✓ Albumentations:', albumentations.__version__)

from digisteel.modules import GhostConv, InnerWIoULoss
print('✓ DigiSteel modules imported successfully')
print('')
print('Setup complete! You are ready to start Week 1.')
"

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Read PROJECT_GUIDE.md (onboarding)"
echo "2. Read CONTRIBUTING.md (team process)"
echo "3. Read GITHUB_SETUP.md (repository setup)"
echo "4. Start your first task!"
echo "=========================================="
