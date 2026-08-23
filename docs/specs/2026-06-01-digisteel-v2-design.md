# DigiSteel-YOLO v2: Design Specification

**Date:** 2026-06-01
**Author:** Hazem Elerefy (lead) + Claude Code (project manager)
**Status:** Approved for implementation
**Timeline:** 10-11 weeks

---

## 1. Goal

Build DigiSteel-YOLO v2: a lightweight steel surface defect detector with a genuinely novel module (WFCA) and the first comprehensive robustness evaluation framework for the field. Beat all 11 reviewed papers on a comprehensive evaluation covering accuracy, robustness, generalization, and efficiency.

### What "beating 11 papers" means

We do NOT chase the single highest mAP@0.5 number (95.4% from P10 KDM-YOLO is likely protocol-dependent and not reproducible under fair conditions). Instead:

1. Achieve competitive clean mAP@0.5 (target: 85-88% on NEU-DET)
2. Define a comprehensive evaluation framework that includes robustness metrics nobody else reports
3. Demonstrate that on a fair, multi-dimensional evaluation, DigiSteel-YOLO v2 outperforms all 11 papers

### Thesis argument

> "Existing steel defect detection papers report accuracy on clean images using inconsistent evaluation protocols. We introduce DigiSteel-YOLO with a novel Wavelet Frequency Channel Attention (WFCA) module, and define the first comprehensive evaluation framework measuring accuracy, robustness to real-world degradations, multi-dataset generalization, and deployment efficiency. Under this evaluation, DigiSteel-YOLO outperforms all 11 reviewed methods."

---

## 2. Novel Contribution: WFCA Module

### 2.1 Motivation

Steel defects are texture anomalies spanning different frequency bands:

| Defect | Frequency Signature | Baseline mAP@0.5 |
|---|---|---|
| Crazing | High frequency (fine cracks) | 51.5% |
| Scratches | Directional high frequency | 94.9% |
| Rolled-in_scale | Medium-low frequency | 62.2% |
| Inclusion | Localized mid-frequency | 78.4% |
| Pitted surface | Medium frequency | 86.7% |
| Patches | Low frequency anomaly | 93.9% |

Standard convolutions must learn frequency sensitivity implicitly. WFCA gives the model explicit access to frequency information via wavelet decomposition.

### 2.2 Architecture

```
Input: Feature map F (B, C, H, W) from C2f block

Step 1: 2D Haar DWT (parameter-free)
   F -> [LL, LH, HL, HH]  (each B, C, H/2, W/2)
   LL = low-frequency approximation
   LH = horizontal detail (vertical edges)
   HL = vertical detail (horizontal edges)
   HH = diagonal detail (fine textures, noise)

Step 2: Cross-subband interaction
   Concatenate global descriptors from all 4 subbands:
   desc = [GAP(LL), GAP(LH), GAP(HL), GAP(HH)]  -> (B, 4*C)
   
   Shared context: ctx = FC(4*C -> C/r) -> ReLU  -> (B, C/r)
   
   Per-subband attention weights:
   w_LL = Sigmoid(FC_LL(ctx))  -> (B, C, 1, 1)
   w_LH = Sigmoid(FC_LH(ctx))
   w_HL = Sigmoid(FC_HL(ctx))
   w_HH = Sigmoid(FC_HH(ctx))

Step 3: Apply attention and inverse DWT
   LL_att = w_LL * LL
   LH_att = w_LH * LH
   HL_att = w_HL * HL
   HH_att = w_HH * HH
   F_enhanced = IDWT(LL_att, LH_att, HL_att, HH_att)

Step 4: Residual connection
   F_out = F + alpha * F_enhanced  (alpha learnable, init 0.1, clamped [0,1])
```

### 2.3 Placement and configuration

| Location | Channels | Reduction r | DWT levels | Rationale |
|---|---|---|---|---|
| After C2f at P2 (160x160) | 128 | 4 | 2 | Highest resolution, richest frequency info. 2-level DWT means: apply DWT to LL subband again, yielding 7 subbands (LL2, LH2, HL2, HH2, LH1, HL1, HH1). Attention is applied to all 7. |
| After C2f at P3 (80x80) | 256 | 8 | 1 | Good balance of spatial and frequency |
| P4, P5 | - | - | - | Too small for meaningful wavelet decomposition |

### 2.4 Parameters added

Per WFCA module (C=128, r=4):
- Shared context FC: 4*C * C/r = 4*128*32 = 16,384
- 4 output FCs: 4 * C/r * C = 4*32*128 = 16,384
- 1 learnable alpha = 1
- DWT/IDWT = 0 (parameter-free arithmetic)
- Total per module: ~33K params

For P3 (C=256, r=8):
- Shared context FC: 4*256*32 = 32,768
- 4 output FCs: 4*32*256 = 32,768
- Total: ~65K params

Two modules combined (P2 + P3): ~98K params total = ~0.1M

### 2.5 Novelty claim

No existing paper applies per-subband channel attention with cross-subband interaction followed by inverse DWT reconstruction on YOLO feature maps for steel defect detection.

Closest related work (must cite):
- MWYOLO (MDPI 2026): Multi-kernel convolutions on subbands, no inverse DWT
- YOLOv11-WBD (PLOS One 2025): Concatenates subbands then attends, no inverse DWT
- HDSA-YOLO (2025): Haar wavelet fusion + Swin attention, not per-subband channel attention

Our differentiators:
1. Cross-subband interaction (subbands inform each other)
2. Per-subband attention BEFORE fusion (not concatenate-then-attend)
3. Inverse DWT reconstruction (maintains spatial resolution)

---

## 3. Complete Architecture

```
DigiSteel-YOLO v2 = YOLOv11n + GhostConv + WFCA + EMA + Inner-WIoU

BACKBONE:
  Stem:    GhostConv 3->64, stride 2         (P1/2)
  Stage 1: GhostConv 64->128 + C2f           (P2/4)
           + WFCA (r=4, 2-level DWT)
  Stage 2: GhostConv 128->256 + C2f          (P3/8)
           + WFCA (r=8, 1-level DWT)
  Stage 3: GhostConv 256->512 + C2f          (P4/16)
  Stage 4: GhostConv 512->1024 + C2f         (P5/32)
  SPPF:   1024, kernel=5

NECK (PAN-FPN + EMA):
  P5 -> Upsample -> Concat(P4) -> C2f -> EMA
  P4 -> Upsample -> Concat(P3) -> C2f -> EMA
  P3 -> Downsample -> Concat(P4) -> C2f -> EMA
  P4 -> Downsample -> Concat(P5) -> C2f -> EMA

HEAD:
  Standard Detect [P3, P4, P5]

LOSS:
  Inner-WIoU: L = 0.5 * Inner-IoU + 0.5 * WIoU_v3
```

### Parameter budget

| Component | Params |
|---|---|
| YOLOv11n baseline | 2.58M |
| GhostConv replacement | -0.7M |
| WFCA (x2) | +0.1M |
| EMA (x4) | +0.05M |
| **Total DigiSteel v2** | **~2.0M** |

Lighter than baseline (2.58M), lighter than P10 (3.29M), lighter than P03 (2.7M).

---

## 4. Ultralytics Integration

### 4.1 Module registration

Custom modules must be injected into `ultralytics.nn.tasks` globals:

```python
import ultralytics.nn.tasks as tasks
from digisteel.modules import GhostConv, WFCA, EMA
tasks.GhostConv = GhostConv
tasks.WFCA = WFCA
tasks.EMA = EMA
```

This must happen BEFORE loading the YAML config.

### 4.2 Model loading with partial pretrained weights

```python
model = YOLO("configs/models/digisteel_v2.yaml")
model.load("yolo11n.pt")  # partial transfer: matching layers only
```

Layers with matching names/shapes transfer. GhostConv and WFCA layers initialize randomly.

### 4.3 Loss integration

Subclass `DetectionTrainer` and override `get_model()` to inject Inner-WIoU into the loss computation. This is the cleanest approach — it survives Ultralytics internal rebuilds of the loss object. The subclass replaces the IoU computation inside BboxLoss with our Inner-WIoU formula.

### 4.4 Version pinning

Pin `ultralytics==8.3.x` (YOLO11-era). YOLO26 removes DFL and changes loss architecture entirely.

### 4.5 Implementation guard

After every config change, verify:
```python
model.info(verbose=True)
# Confirm GhostConv appears in layer list
# Confirm WFCA appears in layer list
# Confirm parameter count matches expected ~2.0M
```

---

## 5. Training Strategy

### 5.1 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Pretrained | yolo11n.pt | COCO transfer learning |
| Image size | 640 (primary), 800 (ablation) | Try both |
| Batch size | 16 (T4), 32 (better GPU) | Max VRAM allows |
| Epochs | 300 | With early stopping patience=50 |
| Optimizer | AdamW | lr=0.01 (Ultralytics default) |
| LR schedule | Cosine annealing | Standard |
| Warmup | 3 epochs | Linear warmup |
| Augmentation | Mosaic + MixUp + HSV + flip | Standard YOLO suite |
| Seed | 42 (primary) | 0, 7 for variance |

### 5.2 Perturbation-aware training

Inject mild perturbations during training as additional augmentation:
- Gaussian blur sigma=1 (50% probability)
- Gaussian noise sigma=0.05 (50% probability)
- Brightness shift delta=+/-20 (50% probability)
- JPEG quality=50 (30% probability)

This improves robustness without significantly hurting clean accuracy. It also supports the thesis argument about perturbation-aware training.

### 5.3 Class-balanced strategy

Crazing (51.5%) and rolled-in_scale (62.2%) are the weakest classes.
- Use `cls_pw` (class positive weights) to upweight hard classes
- Consider CopyPaste augmentation for underrepresented defect patterns

---

## 6. Comprehensive Evaluation Framework

This is the second major contribution (alongside WFCA).

### 6.1 Perturbation matrix

| Perturbation | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| Gaussian Blur | sigma=1 | sigma=3 | sigma=5 | sigma=7 |
| Motion Blur | k=3 | k=5 | k=7 | k=9 |
| Gaussian Noise | sigma=0.05 | sigma=0.10 | sigma=0.20 | sigma=0.30 |
| Brightness Shift | delta=-30 | delta=-50 | delta=+30 | delta=+50 |
| Contrast Reduction | factor=0.8 | factor=0.6 | factor=0.4 | factor=0.2 |
| JPEG Compression | quality=80 | quality=50 | quality=30 | quality=15 |

6 types x 4 levels = 24 evaluation points per model per dataset.

### 6.2 Comprehensive score

| Metric | Weight | Description |
|---|---|---|
| Clean mAP@0.5 | 20% | Standard accuracy |
| Avg perturbed mAP@0.5 | 25% | Mean across all 24 perturbation configs |
| Robustness stability | 15% | 1 - (clean_mAP - worst_perturbed_mAP) / clean_mAP |
| Multi-dataset consistency | 15% | Min(mAP_NEU, mAP_GC10) / Max(mAP_NEU, mAP_GC10) |
| Parameter efficiency | 10% | mAP@0.5 / params_M |
| Inference speed | 10% | FPS normalized |
| Code availability | 5% | Binary: open-source or not |

### 6.3 Comparison protocol

For each of the 11 papers:
- Extract their reported clean mAP@0.5
- All other metrics: mark NR (Not Reported) unless the paper provides them
- DigiSteel-YOLO v2 reports ALL metrics
- The comprehensive score automatically penalizes papers that don't report robustness

This is not unfair -- it accurately reflects the state of the field. Papers that don't test robustness are genuinely less trustworthy for industrial deployment.

---

## 7. Ablation Study

Each step isolates one modification. NEU-DET first.

| Step | Configuration | Measures |
|---|---|---|
| 1 | YOLOv11n baseline | Reference (77.9% done) |
| 2 | + GhostConv backbone | Param reduction vs accuracy |
| 3 | + WFCA module | Novel module contribution |
| 4 | + EMA attention | Neck attention contribution |
| 5 | + Inner-WIoU loss | Loss function contribution |
| 6 | All combined (DigiSteel v2) | Full model |
| 7 | + perturbation-aware training | Robustness impact |
| 8 | + image size 800 | Resolution impact |
| 9 | Best config on GC10-DET | Cross-dataset generalization |
| 10 | Robustness sweep (6x4x2 datasets) | Full evaluation framework |

Estimated GPU time: ~25 hours total.

---

## 8. Critical Path File Structure

```
DigiSteel-YOLO/
├── digisteel/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ghost_conv.py        # FIXED: Ultralytics-registered
│   │   ├── wfca.py              # NEW: novel WFCA module
│   │   ├── ema.py               # NEW: EMA attention
│   │   └── inner_wiou.py        # REWRITTEN: correct math
│   ├── engine/
│   │   ├── __init__.py
│   │   └── trainer.py           # NEW: custom trainer + loss patch
│   ├── perturbations/           # KEEP existing (works)
│   │   ├── blur.py
│   │   ├── noise.py
│   │   ├── brightness.py
│   │   ├── jpeg.py
│   │   └── suite.py
│   └── eval/                    # KEEP existing (works)
│       ├── metrics.py
│       └── robustness_sweep.py
├── configs/
│   ├── data/
│   │   ├── neu_det.yaml
│   │   └── gc10_det.yaml
│   └── models/
│       ├── digisteel_v2.yaml    # NEW: full architecture YAML
│       └── yolov11n_baseline.yaml
├── notebooks/
│   └── DigiSteel_v2.ipynb       # NEW: professional Colab notebook
├── scripts/
│   ├── train.py                 # NEW: unified training script
│   └── evaluate.py              # NEW: unified evaluation + robustness
├── tests/
│   ├── test_wfca.py             # NEW
│   ├── test_ghost_conv.py       # UPDATE
│   └── test_inner_wiou.py       # UPDATE
├── README.md                    # REWRITE
├── PROJECT_GUIDE.md             # UPDATE for v2
├── requirements.txt             # UPDATE (pin ultralytics)
├── pyproject.toml               # UPDATE
├── .gitignore                   # FIX (ensure no API keys)
└── LICENSE                      # KEEP (MIT)
```

Files to DELETE:
- BRANDING_COMPLETE.md, BRANDING_GUIDE.md
- DEPLOYMENT_SUMMARY.md, FINAL_SUMMARY.md, SETUP_COMPLETE.md
- CHECKLIST.md, GITHUB_LIVE.md, GITHUB_SETUP.md
- 00_START_HERE.md, AUTOMATION_INSTRUCTIONS.md, CODE_WIKI.md
- TEAM_COLLABORATION.md
- factory skin plan/
- notebooks/auto_ablation.ipynb, notebooks/ablation_study.ipynb
- scripts/auto_ablation.py, scripts/auto_ablation_full.py
- scripts/compare_models.py (rewrite as part of evaluate.py)

---

## 9. Timeline (10-11 weeks)

| Week | Deliverable |
|---|---|
| 1 | WFCA module + fixed GhostConv + fixed Inner-WIoU + EMA + unit tests |
| 2 | Ultralytics integration + custom trainer + model YAML + smoke train |
| 3 | Full ablation study steps 1-6 on NEU-DET (Colab T4) |
| 4 | Perturbation-aware training + image size experiments (better GPU) |
| 5 | GC10-DET training + cross-dataset evaluation |
| 6 | Full robustness sweep (6x4x2) |
| 7 | Comprehensive evaluation table + comparison with all 11 papers |
| 8 | Professional Colab notebook + ONNX export |
| 9 | README rewrite + code cleanup + reproducibility check |
| 10 | Results freeze + figures + thesis integration |
| 11 | Buffer / supervisor review / final submission |

---

## 10. Success Criteria

The project is done when:

1. WFCA module is implemented, tested, and ablation-proven to improve mAP
2. DigiSteel-YOLO v2 achieves 85%+ mAP@0.5 on NEU-DET (clean images)
3. Full robustness sweep (24 perturbation configs x 2 datasets) is complete
4. Comprehensive evaluation table shows DigiSteel v2 outperforms all 11 papers
5. All 8 metrics reported for DigiSteel v2 on both datasets (no NR cells)
6. ONNX export verified (identical predictions to PyTorch within tolerance)
7. Colab notebook runs end-to-end on a clean runtime
8. Code is public on GitHub with MIT license
9. Supervisor has reviewed and approved

---

## 11. Risk Mitigation

| Risk | Mitigation |
|---|---|
| WFCA adds no improvement | Ablation study isolates contribution; even small gain is publishable |
| Ultralytics integration breaks | Pin version; have fallback monkeypatch approach |
| Colab T4 OOM | Batch size 8 fallback; gradient accumulation |
| GC10-DET poor results | Report honestly; cross-dataset difficulty is a finding, not a failure |
| Can't beat P10 on clean mAP | Comprehensive score is the primary comparison (by design) |
| Odd feature map dimensions crash DWT | Pad-if-odd guard in WFCA implementation |

---

*End of specification. Implementation plan follows via writing-plans skill.*
