#!/bin/bash
#############################################################################
# Dataset Download Script
# Downloads NEU-DET and GC10-DET datasets from Kaggle
#############################################################################

set -e

echo "=========================================="
echo "  DigiSteel-YOLO Dataset Download"
echo "=========================================="

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI is not installed."
    echo "Install with: pip install kaggle"
    echo "Then set up your API key: ~/.kaggle/kaggle.json"
    exit 1
fi

# Check if kaggle API key exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "Get your API key from: https://www.kaggle.com/settings"
    echo "Create the file with:"
    echo '  {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}'
    exit 1
fi

# Download NEU-DET
echo ""
echo "[1/2] Downloading NEU-DET dataset..."
if [ -d "datasets/NEU-DET/images" ]; then
    echo "  NEU-DET already exists, skipping..."
else
    kaggle datasets download -d zssspyker/neu-det -p datasets/NEU-DET --unzip
    echo "  ✓ NEU-DET downloaded"
fi

# Download GC10-DET
echo ""
echo "[2/2] Downloading GC10-DET dataset..."
if [ -d "datasets/GC10-DET/images" ]; then
    echo "  GC10-DET already exists, skipping..."
else
    kaggle datasets download -d zssspyker/gc10-det -p datasets/GC10-DET --unzip
    echo "  ✓ GC10-DET downloaded"
fi

echo ""
echo "=========================================="
echo "  Download Complete!"
echo "=========================================="
echo ""
echo "Dataset locations:"
echo "  NEU-DET: datasets/NEU-DET/"
echo "  GC10-DET: datasets/GC10-DET/"
echo ""
echo "Next: Convert to YOLO format with tools/voc_to_yolo.py"
