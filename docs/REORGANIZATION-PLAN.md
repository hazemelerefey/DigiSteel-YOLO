# Project Reorganization Plan — DigiSteel-YOLO

## Current Problems

1. **Notebooks are flat** — 19 notebooks in one folder, no organization
2. **Results are scattered** — evals/ has 40+ files with no structure
3. **Weights are duplicated** — yolo11n.pt in both notebooks/ and weights/
4. **No per-experiment isolation** — all results go to same evals/ folder
5. **RAM/VRAM issues** — 26.6GB used of 31.7GB RAM, no GPU detected
6. **Outdated paths** — EDA v2 references wrong baseline results

## New Directory Structure

```
DigiSteel-YOLO/
├── configs/
│   ├── data/           # Dataset configs
│   └── models/         # Model architecture configs
├── datasets/
│   ├── NEU-DET/
│   │   ├── yolo/       # Raw images + labels
│   │   ├── yolo_clahe/ # CLAHE variant 1
│   │   └── yolo_clahe4/# CLAHE variant 2
│   └── CG10-DET/
├── digisteel/          # Custom modules
├── experiments/        # NEW: Per-experiment folders
│   ├── 01_baseline_yolov11n/
│   │   ├── notebook.ipynb
│   │   ├── results.json
│   │   ├── weights/
│   │   └── plots/
│   ├── 02_week2_ablation/
│   ├── 03_week3_a1_config_fix/
│   ├── 04_week3_a2_arch_fix/
│   ├── 05_week4_fresh_baseline/
│   ├── 06_dafe_v4/
│   ├── 07_tta_eval/
│   ├── 08_transfer_learning_yolov11n/
│   ├── 09_transfer_learning_fresh_baseline/
│   ├── 10_tl_dafe_comparison/
│   ├── 11_yolov26n_neudet/
│   ├── 12_yolov26_transfer_learning/
│   ├── 13_eda_v1/
│   └── 14_eda_v2/
├── evals/              # Keep for backward compat, but symlink
├── notebooks/          # Keep for backward compat, but symlink
├── runs/               # Training outputs
├── scripts/
├── tools/
├── weights/            # Pretrained weights only
│   ├── yolo11n.pt
│   └── yolo26n.pt
└── docs/
```

## Execution Plan

### Phase 1: Create new directory structure
- Create experiments/ folder with per-experiment subfolders
- Move notebooks to their experiment folders
- Move results to corresponding experiment folders
- Move weights to correct locations

### Phase 2: Fix RAM/VRAM issues
- Add memory management to training scripts
- Implement gradient accumulation
- Add batch size auto-tuning
- Clear CUDA cache between runs

### Phase 3: Update all paths in notebooks
- Update import paths
- Update data paths
- Update results paths
- Update weights paths

### Phase 4: Create master results tracker
- JSON file tracking all experiments
- Auto-update when notebooks run
- Dashboard for comparing results

## Expected Outcome
- Clean, navigable project structure
- Each experiment self-contained
- Easy to find any result
- RAM/VRAM issues resolved
- All paths consistent
