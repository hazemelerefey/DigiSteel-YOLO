#!/bin/bash
#############################################################################
# Push Colab Notebook to GitHub
# Run this script to upload the notebook to your repository
#############################################################################

set -e

echo "=========================================="
echo "  Pushing Colab Notebook to GitHub"
echo "=========================================="

# Check if git is configured
if [ -z "$(git config user.name)" ]; then
    echo "ERROR: Git user.name not configured"
    echo "Run: git config --global user.name 'Your Name'"
    exit 1
fi

if [ -z "$(git config user.email)" ]; then
    echo "ERROR: Git user.email not configured"
    echo "Run: git config --global user.email 'your@email.com'"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "notebooks" ]; then
    echo "ERROR: Please run this script from the DigiSteel-YOLO root directory"
    exit 1
fi

# Add notebook files
echo "Adding notebook files..."
git add notebooks/DigiSteel_YOLO_Colab.ipynb
git add notebooks/README.md

# Commit
echo "Committing changes..."
git commit -m "Add Colab notebook for training pipeline

- Complete training pipeline for DigiSteel-YOLO
- Includes environment setup, dataset download, training, evaluation
- Compares results with 11 reference papers
- Exports model to ONNX for edge deployment

Usage:
1. Open in Google Colab
2. Enable GPU (T4 recommended)
3. Run all cells"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "=========================================="
echo "  Push Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/hazemelerefey/DigiSteel-YOLO"
echo "2. Navigate to notebooks/DigiSteel_YOLO_Colab.ipynb"
echo "3. Click 'Open in Colab' button"
echo "4. Enable GPU and run the notebook"
echo ""
echo "Or use this direct link:"
echo "https://colab.research.google.com/github/hazemelerefey/DigiSteel-YOLO/blob/main/notebooks/DigiSteel_YOLO_Colab.ipynb"
