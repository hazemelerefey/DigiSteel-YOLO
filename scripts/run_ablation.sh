#!/bin/bash
#############################################################################
# DigiSteel-YOLO Automated Ablation Study
# Runs all experiments automatically in Colab
#############################################################################

set -e

echo "=========================================="
echo "  DigiSteel-YOLO Automated Ablation Study"
echo "=========================================="

# Configuration
DATASET="NEU-DET"
SEED=42
EPOCHS=100

# Create necessary directories
mkdir -p configs runs evals

# Step 1: Setup environment
echo "[Step 1/8] Setting up environment..."
pip install -q ultralytics albumentations pandas

# Step 2: Configure Kaggle
echo "[Step 2/8] Configuring Kaggle..."
mkdir -p ~/.kaggle
echo '{"username":"hazemelerefy","key":"KGAT_0e5696318d7e5a3caf038db9497466e5"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Step 3: Clone repository
echo "[Step 3/8] Cloning repository..."
cd /content
if [ ! -d "DigiSteel-YOLO" ]; then
    git clone https://github.com/hazemelerefey/DigiSteel-YOLO.git
fi
cd DigiSteel-YOLO

# Step 4: Download datasets
echo "[Step 4/8] Downloading datasets..."
if [ ! -d "datasets/NEU-DET/yolo/images/train" ]; then
    kaggle datasets download -d sovitrath/neu-steel-surface-defect-detect-trainvalid-split -p datasets/NEU-DET --unzip -q
    python tools/prepare_datasets.py
fi

# Step 5: Create C3Ghost config
echo "[Step 5/8] Creating architecture configs..."
cat > configs/yolov11n_c3ghost.yaml << 'EOF'
nc: 6

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 1, C3Ghost, [128]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, C3Ghost, [256]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 1, C3Ghost, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 1, C3Ghost, [1024]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 1, C3Ghost, [1024, False]]
  - [[15, 18, 21], 1, Detect, [nc]]
EOF

# Step 6: Run ablation experiments
echo "[Step 6/8] Running ablation experiments..."

# Experiment 1: C3Ghost Architecture
echo "  [1/5] Training C3Ghost Architecture..."
python -c "
from ultralytics import YOLO
import pandas as pd
import json
import time
from pathlib import Path

model = YOLO('configs/yolov11n_c3ghost.yaml')
start = time.time()
results = model.train(data='configs/neu_det.yaml', epochs=100, imgsz=640, batch=16, seed=42, 
                      project='runs', name='ablation_c3ghost_neu_det_seed42', exist_ok=True, patience=30)
elapsed = time.time() - start

df = pd.read_csv('runs/ablation_c3ghost_neu_det_seed42/results.csv')
best_map = df['metrics/mAP50(B)'].max()
best_map50_95 = df['metrics/mAP50-95(B)'].max()

Path('evals').mkdir(exist_ok=True)
with open('evals/ablation_c3ghost.json', 'w') as f:
    json.dump({'experiment': 'C3Ghost', 'mAP50': float(best_map), 'mAP50_95': float(best_map50_95), 
               'training_time_hours': elapsed/3600}, f, indent=2)
print(f'  C3Ghost Results: mAP@0.5={best_map:.1%}, mAP@0.5:0.95={best_map50_95:.1%}')
"

# Experiment 2: Image Size 800
echo "  [2/5] Training Image Size 800..."
python -c "
from ultralytics import YOLO
import pandas as pd
import json
import time
from pathlib import Path

model = YOLO('yolo11n.pt')
start = time.time()
results = model.train(data='configs/neu_det.yaml', epochs=100, imgsz=800, batch=12, seed=42, 
                      project='runs', name='ablation_imgsize800_neu_det_seed42', exist_ok=True, patience=30)
elapsed = time.time() - start

df = pd.read_csv('runs/ablation_imgsize800_neu_det_seed42/results.csv')
best_map = df['metrics/mAP50(B)'].max()
best_map50_95 = df['metrics/mAP50-95(B)'].max()

with open('evals/ablation_imgsize800.json', 'w') as f:
    json.dump({'experiment': 'ImgSize800', 'mAP50': float(best_map), 'mAP50_95': float(best_map50_95), 
               'training_time_hours': elapsed/3600, 'imgsz': 800}, f, indent=2)
print(f'  ImgSize800 Results: mAP@0.5={best_map:.1%}, mAP@0.5:0.95={best_map50_95:.1%}')
"

# Experiment 3: Enhanced Augmentation
echo "  [3/5] Training Enhanced Augmentation..."
python -c "
from ultralytics import YOLO
import pandas as pd
import json
import time
from pathlib import Path

model = YOLO('yolo11n.pt')
start = time.time()
results = model.train(data='configs/neu_det.yaml', epochs=100, imgsz=640, batch=16, seed=42, 
                      project='runs', name='ablation_enhanced_aug_neu_det_seed42', exist_ok=True, 
                      patience=30, mosaic=1.0, mixup=0.1, copy_paste=0.1)
elapsed = time.time() - start

df = pd.read_csv('runs/ablation_enhanced_aug_neu_det_seed42/results.csv')
best_map = df['metrics/mAP50(B)'].max()
best_map50_95 = df['metrics/mAP50-95(B)'].max()

with open('evals/ablation_enhanced_aug.json', 'w') as f:
    json.dump({'experiment': 'EnhancedAug', 'mAP50': float(best_map), 'mAP50_95': float(best_map50_95), 
               'training_time_hours': elapsed/3600, 'mosaic': 1.0, 'mixup': 0.1, 'copy_paste': 0.1}, f, indent=2)
print(f'  EnhancedAug Results: mAP@0.5={best_map:.1%}, mAP@0.5:0.95={best_map50_95:.1%}')
"

# Experiment 4: Cosine Learning Rate
echo "  [4/5] Training Cosine Learning Rate..."
python -c "
from ultralytics import YOLO
import pandas as pd
import json
import time
from pathlib import Path

model = YOLO('yolo11n.pt')
start = time.time()
results = model.train(data='configs/neu_det.yaml', epochs=100, imgsz=640, batch=16, seed=42, 
                      project='runs', name='ablation_cosine_lr_neu_det_seed42', exist_ok=True, 
                      patience=30, cos_lr=True)
elapsed = time.time() - start

df = pd.read_csv('runs/ablation_cosine_lr_neu_det_seed42/results.csv')
best_map = df['metrics/mAP50(B)'].max()
best_map50_95 = df['metrics/mAP50-95(B)'].max()

with open('evals/ablation_cosine_lr.json', 'w') as f:
    json.dump({'experiment': 'CosineLR', 'mAP50': float(best_map), 'mAP50_95': float(best_map50_95), 
               'training_time_hours': elapsed/3600, 'cos_lr': True}, f, indent=2)
print(f'  CosineLR Results: mAP@0.5={best_map:.1%}, mAP@0.5:0.95={best_map50_95:.1%}')
"

# Step 7: Find best configuration
echo "[Step 7/8] Finding best configuration..."
python -c "
import json
from pathlib import Path

# Load all results
results = {}
for f in Path('evals').glob('ablation_*.json'):
    with open(f) as fh:
        data = json.load(fh)
        results[data['experiment']] = data

# Find best
best_name = max(results.keys(), key=lambda k: results[k].get('mAP50', 0))
best_data = results[best_name]

print(f'\\n  Best Configuration: {best_name}')
print(f'  mAP@0.5: {best_data[\"mAP50\"]:.1%}')
print(f'  mAP@0.5:0.95: {best_data[\"mAP50_95\"]:.1%}')

# Save best config
with open('evals/best_config.json', 'w') as f:
    json.dump({'best_experiment': best_name, **best_data}, f, indent=2)
"

# Step 8: Train final optimized model
echo "[Step 8/8] Training final optimized model..."
python -c "
import json
from ultralytics import YOLO
import pandas as pd
import time
from pathlib import Path

# Load best config
with open('evals/best_config.json') as f:
    best = json.load(f)

best_name = best['best_experiment']
print(f'\\n  Training final model with {best_name} configuration...')

# Get config
config = best.get('config', 'yolo11n.pt')
imgsz = best.get('imgsz', 640)
batch = best.get('batch', 16)

if config.endswith('.yaml'):
    model = YOLO(config)
else:
    model = YOLO(config)

start = time.time()
results = model.train(data='configs/neu_det.yaml', epochs=200, imgsz=imgsz, batch=batch, seed=42, 
                      project='runs', name='final_optimized_neu_det_seed42', exist_ok=True, patience=50)
elapsed = time.time() - start

df = pd.read_csv('runs/final_optimized_neu_det_seed42/results.csv')
best_map = df['metrics/mAP50(B)'].max()
best_map50_95 = df['metrics/mAP50-95(B)'].max()

print(f'\\n  FINAL RESULTS:')
print(f'  mAP@0.5: {best_map:.1%}')
print(f'  mAP@0.5:0.95: {best_map50_95:.1%}')
print(f'  Training time: {elapsed/3600:.1f} hours')

with open('evals/final_results.json', 'w') as f:
    json.dump({'model': 'DigiSteel-YOLO Final', 'config': config, 'mAP50': float(best_map), 
               'mAP50_95': float(best_map50_95), 'training_time_hours': elapsed/3600}, f, indent=2)
"

echo ""
echo "=========================================="
echo "  ABLATION STUDY COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: evals/"
echo "Final model: runs/final_optimized_neu_det_seed42/weights/best.pt"
echo ""
echo "Next steps:"
echo "  1. Review evals/best_config.json"
echo "  2. Run robustness evaluation"
echo "  3. Export to ONNX"
