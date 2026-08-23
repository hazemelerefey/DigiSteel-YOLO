# DigiSteel-YOLO: Reference Papers Comprehensive Summary

## Literature Review — 11 Reference Papers for Steel Surface Defect Detection

**Compiled:** June 24, 2026
**Project:** DigiSteel-YOLO — Comprehensive Robustness Study of Lightweight YOLO Detectors for Steel Surface Defect Detection
**Team:** Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham
**Supervisor:** Dr. Tarek Ghoneimy

---

## Table of Contents

1. [Executive Summary & Comparison Table](#1-executive-summary)
2. [P01 — PSF-YOLO](#2-p01--psf-yolo)
3. [P02 — LAM-YOLOv10n](#3-p02--lam-yolov10n)
4. [P03 — YOLO-LSDI](#4-p03--yolo-lsdi)
5. [P04 — Lightweight-YOLOv8](#5-p04--lightweight-yolov8)
6. [P05 — SCCI-YOLO](#6-p05--scci-yolo)
7. [P06 — ELS-YOLO](#7-p06--els-yolo)
8. [P07 — ASFRW-YOLO](#8-p07--asfrw-yolo)
9. [P08 — MSFE-YOLO](#9-p08--msfe-yolo)
10. [P09 — EFEN-YOLOv8](#10-p09--efen-yolov8)
11. [P10 — KDM-YOLO](#11-p10--kdm-yolo)
12. [P11 — YOLOv11-EMD](#12-p11--yolov11-emd)
13. [Comparative Analysis](#13-comparative-analysis)

---

## 1. Executive Summary

### Master Comparison Table

| ID | Paper | Base Model | mAP@0.5 (NEU-DET) | mAP@0.5:0.95 | Params (M) | FPS | Year | Journal |
|----|-------|-----------|-------------------|--------------|------------|-----|------|---------|
| P01 | PSF-YOLO | YOLOv11n | N/A (82.2% GC10-DET+) | 45.8% (GC10-DET+) | 1.82 | N/A | 2025 | Nature Sci. Reports |
| P02 | LAM-YOLOv10n | YOLOv10n | **94.39%** | N/A | N/A | 154 | 2025 | Nature Sci. Reports |
| P03 | YOLO-LSDI | YOLOv11n | **83.0%** | ~52-57% | 2.7 | 162.1 | 2025 | MDPI Electronics |
| P04 | Lightweight-YOLOv8 | YOLOv8n | 78.6% | 44.5% | **2.04** | **171.5** | 2025 | Nature Sci. Reports |
| P05 | SCCI-YOLO | YOLOv8n | 78.6% | N/A | **1.68** | **270.2** | 2025 | Nature Sci. Reports |
| P06 | ELS-YOLO | YOLOv11n | 79.5% | 43.2% | 2.36 | N/A | 2025 | MDPI Electronics |
| P07 | ASFRW-YOLO | YOLOv5s | **83.2%** | **46.4%** | 6.20 | 125 | 2025 | Nature Sci. Reports |
| P08 | MSFE-YOLO | YOLOv11s | 79.8% | N/A | 11.69 | 89.3 | 2026 | MDPI Sensors |
| P09 | EFEN-YOLOv8 | YOLOv8n | 80.4% | N/A | N/A | N/A | 2026 | PLOS ONE |
| P10 | KDM-YOLO | YOLOv10n | **95.4%** | N/A | 3.29 | 155.6 | 2026 | MDPI Sensors |
| P11 | YOLOv11-EMD | YOLOv11 | **94.9%** | N/A | N/A | N/A | 2025 | MDPI Mathematics |

### Key Findings

- **Highest mAP@0.5:** P10 KDM-YOLO (95.4%) — but uses different training protocol (native 200px, no upscale)
- **Lightest model:** P05 SCCI-YOLO (1.68M params) — ultra-lightweight for edge deployment
- **Fastest inference:** P05 SCCI-YOLO (270.2 FPS) — best real-time performance
- **Best mAP@0.5:0.95:** P07 ASFRW-YOLO (46.4%) — best localization precision
- **Best parameter efficiency:** P04 Lightweight-YOLOv8 (78.6% mAP with only 2.04M params)
- **Most cited venue:** Nature Scientific Reports (4 papers)

### Novel Modules Summary

| Paper | Novel Modules |
|-------|--------------|
| P01 PSF-YOLO | MDF-Neck, Virtual Fusion Head, Attention Concat, GhostConv |
| P02 LAM-YOLOv10n | GhostConv, SMA (Spatial Multi-Scale Attention), MFFN |
| P03 YOLO-LSDI | AMSPPF, DSAM, LDConv, Inner-CIoU |
| P04 Lightweight-YOLOv8 | GhostNet backbone, MPCA, SIoU loss |
| P05 SCCI-YOLO | SPD-Conv, C2f_EMA, CCFM, Inner-IoU |
| P06 ELS-YOLO | C3k2_THK, Staged-Slim-Neck, MSDetect |
| P07 ASFRW-YOLO | ASF (SSFF+TFE+CPAM), RepNCSPELAN4, WIoU v3 |
| P08 MSFE-YOLO | MSFC, C2MSDA, AFFE |
| P09 EFEN-YOLOv8 | SAConv, LSKA, WASPP, gamma-FEIoU |
| P10 KDM-YOLO | KWConv, C2f-DRB, MSAF |
| P11 YOLOv11-EMD | InnerEIoU, MSDA, C3k2_DynamicConv, Transfer Learning |

---

## 2. P01 — PSF-YOLO

### Full Title
"A lightweight YOLOv11-based framework for small steel defect detection with a newly enhanced feature fusion module"

### Authors
Yongyao Wang, Haiyang Sun, Kai Luo, Quanfu Zhu, Haofei Li, Yuyang Sun, Zhenjie Wu, Gang Wang

### Publication
- **Journal:** Scientific Reports (Nature Portfolio)
- **Year:** 2025
- **Volume:** 15, Article 34322
- **DOI:** [10.1038/s41598-025-16619-9](https://doi.org/10.1038/s41598-025-16619-9)

### Architecture & Base Model
- **Base:** YOLOv11n
- **Parameters:** 1.82M (25% reduction from YOLOv11n baseline)
- **Key modifications:**
  - **MDF-Neck:** Multi-Dimensional-Fusion Neck integrating P1 layer (normally omitted), with dense cross-scale connectivity and adaptive weighting. Achieves 16.7% fewer parameters than standard FPN.
  - **Virtual Fusion Head:** Resolution-aware pooling + alignment mechanism
  - **Attention Concat Module:** Lightweight spatial-channel attention (outperforms SE and CBAM)
  - **GhostConv:** Replaces standard CBS layers for parameter reduction

### Performance Metrics

| Metric | Value | Baseline (YOLOv11n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (GC10-DET+) | 82.2% | 79.0% | +3.2% |
| mAP@0.5:0.95 (GC10-DET+) | 45.8% | 42.5% | +3.3% |
| Parameters | 1.82M | 2.43M | -25% |

**Note:** This paper does NOT evaluate on NEU-DET. Primary benchmark is GC10-DET+ (enhanced version of GC10-DET).

### Training Details
- **Optimizer:** SGD
- **Dataset:** GC10-DET+ (enhanced GC10-DET with additional images and re-annotation)
- **Split:** 7:1:2 (train:val:test)
- **Augmentation:** Vertical flipping, horizontal mirroring, rotations, mosaic, mixed image random cropping

### Cross-Domain Validation
- **RSOD Dataset** (remote sensing): +0.2% mAP, +5.3% precision with 25.8% fewer params
- **Wind Turbine Dataset:** +3.2% mAP, +2.1% mAP@0.5:0.95

### Strengths
1. Significant parameter reduction (25%) with simultaneous accuracy gains
2. Strong cross-domain generalization (validated on 3 datasets)
3. Principled architecture design with individual ablation studies

### Weaknesses
1. No FPS/latency benchmarks reported
2. No evaluation on NEU-DET (most common benchmark)
3. Dataset (GC10-DET+) not publicly available

---

## 3. P02 — LAM-YOLOv10n

### Full Title
"Steel surface defect detection algorithm based on improved YOLOv10"

### Authors
Laomo Zhang (corresponding), Zhike Wang, Ying Ma, Guowei Li

### Publication
- **Journal:** Scientific Reports (Nature Publishing Group)
- **Year:** 2025
- **Volume:** 15, Article 32827
- **DOI:** [10.1038/s41598-025-16725-8](https://doi.org/10.1038/s41598-025-16725-8)

### Architecture & Base Model
- **Base:** YOLOv10n
- **LAM = Latent-space Attention Multi-scale**
- **Key modifications:**
  - **GhostConv:** Lightweight ghost module for parameter reduction (inspired by GhostNet, Han et al. CVPR 2020)
  - **SMA Module (Spatial Multi-Scale Attention):** Three parallel branches with multi-scale pooling, cross-spatial attention, and softmax-weighted fusion. Placed in backbone.
  - **MFFN Module (Multi-Branch Feature Fusion Network):** Two-branch design with GAP+conv and local avg/max pooling. Applied at three Neck output layers.

### Performance Metrics

| Metric | Value | Baseline (YOLOv10n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | **94.39%** | 88.0% | +6.39% |
| Precision | 96.96% | 93.49% | +3.47% |
| Recall | 93.73% | 91.02% | +2.71% |
| FPS | 154 | 187.2 | -17.7% |

### Training Details
- **Hardware:** Intel i7-9700K CPU, NVIDIA GTX 1060Ti GPU
- **Epochs:** 100
- **Batch size:** 16
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Dataset:** NEU-DET (augmented as PRO-DataSet)

### Ablation Study

| Configuration | Precision | Recall | mAP@0.5 |
|---------------|-----------|--------|---------|
| YOLOv10n baseline | 93.49% | 91.02% | 88.0% |
| + GhostConv | 93.63% | 91.89% | — |
| + GhostConv + SMA | 93.76% | 92.77% | — |
| + GhostConv + SMA + MFFN | **96.96%** | **93.73%** | **94.39%** |

### Comparison with YOLOv10 Family

| Model | Precision |
|-------|-----------|
| YOLOv10n | 93.49% |
| YOLOv10s | 94.88% |
| YOLOv10m | 95.12% |
| YOLOv10b | 95.68% |
| **LAM-YOLOv10n** | **96.96%** |
| YOLOv10l | 97.34% |

LAM-YOLOv10n (nano-sized) outperforms YOLOv10n through YOLOv10b.

### Strengths
1. Significant accuracy improvement with lightweight design (96.96% precision as nano model)
2. Well-designed modular architecture with clear ablation
3. Competitive inference speed (154 FPS)

### Weaknesses
1. Missing key metrics (parameter count, FLOPs, mAP@0.5:0.95)
2. Only evaluated on NEU-DET (no cross-dataset validation)
3. Outdated hardware (GTX 1060Ti) and limited comparison scope

---

## 4. P03 — YOLO-LSDI

### Full Title
"YOLO-LSDI: An Enhanced Algorithm for Steel Surface Defect Detection Using a YOLOv11 Network"

### Authors
Fuqiang Wang, Xinbin Jiang, Yizhou Han, Lei Wu (corresponding)

### Publication
- **Journal:** Electronics (MDPI)
- **Year:** 2025
- **Volume:** 14, Issue 13, Article 2576
- **DOI:** [10.3390/electronics14132576](https://doi.org/10.3390/electronics14132576)

### Architecture & Base Model
- **Base:** YOLOv11n
- **Parameters:** 2.7M
- **Key modifications:**
  - **AMSPPF (Adaptive Multi-Scale Pooling-Fast):** Enhanced SPPF for global semantic + local edge feature extraction
  - **DSAM (Deformable Spatial Attention Module):** Hybrid deformable + spatial attention for complex backgrounds
  - **LDConv (Linear Deformable Convolution):** Adapts to irregular defect shapes with low computational cost
  - **Inner-CIoU Loss:** Improved bounding box regression loss

### Performance Metrics

| Metric | Value | Baseline (YOLOv11n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | **83.0%** | ~77.2% | +5.8% |
| mAP@0.5:0.95 | — | — | +2.4% |
| F1-Score | — | — | +6.2% |
| GFLOPs | — baseline | — | -6.1 |
| FPS | 162.1 | — | Real-time |

### Cross-Dataset Generalization

| Dataset | mAP@0.5 Improvement | mAP@0.5:0.95 Improvement |
|---------|--------------------|--------------------------|
| GC10-DET | +4.2% | +1.1% |
| APSPC PCB | +2.1% | +1.5% |

### Strengths
1. Strong empirical results: +5.8% mAP with reduced computational cost
2. Real-time inference at 162.1 FPS
3. Cross-dataset generalization validated on 3 datasets

### Weaknesses
1. Published in MDPI Electronics (not top-tier venue)
2. Incremental contributions (individually known techniques)
3. Limited ablation clarity from available data

---

## 5. P04 — Lightweight-YOLOv8

### Full Title
"A lightweight algorithm for steel surface defect detection using improved YOLOv8"

### Authors
Shuangbao Ma, Xin Zhao, Li Wan, Yapeng Zhang, Hongliang Gao (corresponding)

### Publication
- **Journal:** Scientific Reports (Nature Publishing Group)
- **Year:** 2025
- **Volume:** 15, Article 8966
- **DOI:** [10.1038/s41598-025-93469-5](https://doi.org/10.1038/s41598-025-93469-5)
- **Citations:** 37 (highly cited for this field)

### Architecture & Base Model
- **Base:** YOLOv8n
- **Parameters:** 2.04M (35.26% reduction from 3.13M baseline)
- **Key modifications:**
  - **GhostNet Backbone:** Replaces CSPDarknet entirely. Uses GhostModule (primary conv + cheap DepthwiseConv) for dramatic parameter reduction.
  - **MPCA (Multi-Path Coordinate Attention):** Max/avg pooling along W and H directions, producing 4 × 1D feature vectors. Captures long-range dependencies while preserving positional information.
  - **SIoU Loss (SCYLLA-IoU):** Four cost components — angle cost, distance cost, shape cost, IoU cost. Enables faster and more accurate bounding box regression.

### Performance Metrics

| Metric | Value | Baseline (YOLOv8n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | 78.6% | 77.4% | +1.2% |
| mAP@0.5:0.95 | 44.5% | 43.3% | +1.2% |
| Precision | 75.7% | 73.0% | +2.7% |
| Recall | 70.6% | 69.9% | +0.7% |
| Parameters | **2.04M** | 3.13M | **-35.26%** |
| GFLOPs | **5.1G** | 8.1G | **-37%** |
| FPS | **171.5** | 153.6 | +11.65% |

### Per-Class Performance (NEU-DET)

| Defect Class | AP@0.5 |
|-------------|--------|
| Crazing | 21.4% (weakest) |
| Inclusion | 85.5% |
| Patches | 93.7% |
| Pitted Surface | 86.2% |
| Rolled-in Scale | 72.8% |
| Scratches | 91.8% |

### Edge Deployment Results

| Device | FPS |
|--------|-----|
| Raspberry Pi 4B | 115.7 |
| Jetson AGX Xavier | 440.3 |

### Training Details
- **Hardware:** NVIDIA RTX 4090
- **Epochs:** 300
- **Batch size:** 16
- **Optimizer:** SGD (lr=0.01, cosine annealing)
- **Image size:** 640×640
- **Split:** 90% train / 10% val

### Strengths
1. Extreme lightweight design (2.04M params, 5.1 GFLOPs) with improved accuracy
2. Comprehensive multi-metric improvement (ALL metrics improved simultaneously)
3. Practical edge deployment validation (Raspberry Pi, Jetson)

### Weaknesses
1. Limited dataset diversity (NEU-DET only)
2. Poor performance on crazing defects (21.4% AP)
3. No comparison with recent 2024 lightweight SOTA (YOLOv9, v10, v11)

---

## 6. P05 — SCCI-YOLO

### Full Title
"An efficient and lightweight algorithm for detecting surface defects of steel based on SCCI-YOLO"

### Authors
Huixiang Zhou, Hong Zou (corresponding), Gaojun Hu

### Publication
- **Journal:** Scientific Reports (Nature Portfolio)
- **Year:** 2025
- **Volume:** 15, Article 36276
- **DOI:** [10.1038/s41598-025-20154-y](https://doi.org/10.1038/s41598-025-20154-y)

### Architecture & Base Model
- **Base:** YOLOv8n
- **Parameters:** 1.68M (43.9% reduction — lightest model in the comparison)
- **SCCI = SPD-Conv + C2f_EMA + CCFM + Inner-IoU**
- **Key modifications:**
  - **SPD-Conv (Space-to-Depth Convolution):** Replaces stride-2 downsampling, preserving spatial information for small defect detection
  - **C2f_EMA:** C2f module with integrated EMA (Efficient Multi-scale Attention) for enhanced feature extraction
  - **CCFM (Cross-scale Convolutional Feature Module):** Lightweight cross-scale feature fusion in the Neck
  - **Inner-IoU Loss:** Improved IoU loss for better convergence and regression accuracy

### Performance Metrics

| Metric | Value | Baseline (YOLOv8n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | 78.6% | ~76.4% | +2.2% |
| Parameters | **1.68M** | ~3.0M | **-43.9%** |
| FPS | **270.2** | — | Fastest |

### Strengths
1. Ultra-lightweight design (1.68M params) — most parameter-efficient model
2. Fastest inference speed (270.2 FPS)
3. Effective small defect detection via SPD-Conv

### Weaknesses
1. Modest absolute mAP (78.6%)
2. Single dataset evaluation (NEU-DET only)
3. Trade-off between lightweight and accuracy ceiling

---

## 7. P06 — ELS-YOLO

### Full Title
"ELS-YOLO: Efficient Lightweight YOLO for Steel Surface Defect Detection"

### Authors
Zhiheng Zhang, Guoyun Zhong (corresponding), Peng Ding (corresponding), Jianfeng He, Jun Zhang, Chongyang Zhu

### Publication
- **Journal:** Electronics (MDPI)
- **Year:** 2025
- **Volume:** 14, Issue 19, Article 3877
- **DOI:** [10.3390/electronics14193877](https://doi.org/10.3390/electronics14193877)

### Architecture & Base Model
- **Base:** YOLOv11n
- **Parameters:** 2.36M (8.5% reduction from YOLOv11n's 2.58M)
- **FLOPs:** 5.6G (11.1% reduction from 6.3G)
- **Key modifications:**
  - **C3k2_THK Module:** T-shaped convolution + Heterogeneous Kernel Selection (HKS) + SCSA attention. Multi-kernel sizes (3,5,7,9) at different hierarchical levels.
  - **Staged-Slim-Neck:** DGSConv-L (dual group shuffle) in lower layers, DGSConv-H (dilated group shuffle) in higher layers, with GMLCA attention at highest level.
  - **MSDetect Detection Head:** MRFB (Multi-scale Receptive Field Block) for regression, MRFB-L (lightweight) for classification.

### Performance Metrics

| Metric | Value | Baseline (YOLOv11n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | 79.5% | 76.4% | +3.1% |
| mAP@0.5:0.95 | 43.2% | 44.1% | -0.9% |
| Parameters | 2.36M | 2.58M | -8.5% |
| FLOPs | 5.6G | 6.3G | -11.1% |

### Per-Class AP@0.5 (NEU-DET)

| Defect Class | AP@0.5 |
|-------------|--------|
| Crazing | 47.7% |
| Inclusion | 84.8% |
| Patches | 92.4% |
| Pitted Surface | 89.3% |
| Rolled Scale | 70.3% |
| Scratches | 90.2% |

### Cross-Dataset Generalization

| Dataset | mAP@0.5 | Improvement vs YOLOv11n |
|---------|---------|------------------------|
| GC10-DET | 54.0% | +2.4% |
| Severstal | 49.5% | +0.4% |

### Training Details
- **GPU:** NVIDIA RTX 4070Ti Super (16 GB)
- **Epochs:** 400
- **Batch size:** 16
- **Optimizer:** AdamW (lr=0.001)
- **Image size:** 640×640

### Strengths
1. Exceptional parameter efficiency (fewest params + lowest FLOPs among YOLOv11n-based)
2. Strong multi-scale detection via C3k2_THK + Staged-Slim-Neck + MSDetect
3. Cross-dataset generalization on 3 datasets

### Weaknesses
1. mAP@50:0.95 gap (43.2% vs 44.1% baseline)
2. No FPS/inference speed reported
3. Authors acknowledge "insufficient small-scale defect detection accuracy"

---

## 8. P07 — ASFRW-YOLO

### Full Title
"A high precision and lightweight method for steel surface defect detection based on improved YOLOv5"

### Authors
Mudan Zhou, Haoyu Wang, Yuhao Wang

### Publication
- **Journal:** Scientific Reports (Nature Publishing Group)
- **Year:** 2025
- **DOI:** [10.1038/s41598-025-28022-5](https://doi.org/10.1038/s41598-025-28022-5)

### Architecture & Base Model
- **Base:** YOLOv5s
- **Parameters:** 6.20M (reduced from 7.23M baseline)
- **ASFRW = ASF + RepNCSPELAN4 + WIoU v3**
- **Key modifications:**
  - **ASF Module (Attentional Scale Sequence Fusion):** Contains SSFF (3D convolution for inter-scale dependencies), TFE (Triple Feature Encoding), and CPAM (Channel-Position Attention with k-nearest neighbor interactions)
  - **RepNCSPELAN4:** From YOLOv9, replaces all C3 modules. Uses depthwise separable convolution, residual connections, structural re-parameterization.
  - **WIoU v3 Loss:** Dynamic non-monotonic focusing mechanism for bounding box regression

### Performance Metrics

| Metric | Value | Baseline (YOLOv5s) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | **83.2%** | 76.2% | **+7.0%** |
| mAP@0.5:0.95 | **46.4%** | 41.0% | **+5.4%** |
| Parameters | 6.20M | 7.23M | -1.03M |
| GFLOPs | 14.2 | 16.6 | -2.4 |
| FPS | ~125 | — | Real-time |

### Per-Defect Improvements
- Crazing: +14.4%
- Rolled-in Scale: +21.3%
- Inclusion: +3.8%

### Ablation Study

| Configuration | mAP@0.5 | Params | FPS |
|---------------|---------|--------|-----|
| ASF alone | 79.8% | 7.47M | 219 |
| RepNCSPELAN4 alone | 79.2% | 6.04M | 132 |
| ASF + RepNCSPELAN4 | 81.2% | 6.20M | 127 |
| Full ASFRW-YOLO | **83.2%** | **6.20M** | **125** |

### Training Details
- **GPU:** NVIDIA RTX 4060 Laptop (8 GB)
- **Epochs:** 300
- **Batch size:** 16
- **Optimizer:** SGD (momentum 0.937, weight decay 0.0005)
- **Image size:** 640×640

### Strengths
1. Parameter efficiency (6.20M, fewer than YOLOv5s baseline)
2. Strong small-defect detection (crazing +14.4%, rolled-in scale +21.3%)
3. Real-time inference (~125 FPS)

### Weaknesses
1. Struggles with crack-type defects (irregular, low-contrast)
2. Lab-only evaluation (NEU-DET only, no real factory data)
3. Limited dataset scope

---

## 9. P08 — MSFE-YOLO

### Full Title
"MSFE-YOLO: A Steel Surface Defect Detection Algorithm Integrating Multi-Scale Frequency Domain and Defect-Aware Attention"

### Authors
Siqi Su, Jiale Shen, P. Lin, Wanhe Tang, Weijie Zhang, Zhen Chen

### Publication
- **Journal:** Sensors (MDPI)
- **Year:** 2026
- **Volume:** 26, Issue 8, Article 2311
- **DOI:** [10.3390/s26082311](https://doi.org/10.3390/s26082311)

### Architecture & Base Model
- **Base:** YOLOv11s
- **Parameters:** 11.69M (up from 8.99M baseline)
- **GFLOPs:** 27.9 (up from 21.6)
- **Key modifications:**
  - **MSFC (Multi-Scale Frequency-Enhanced Convolution):** Parallel depthwise separable convolutions with depth-adaptive dilation rates + Laplacian frequency domain enhancement. Placed in neck.
  - **C2MSDA (Cross-Stage Partial with Multi-Scale Defect-Aware Attention):** Replaces C2PSA. Contains Edge-Aware Module (Sobel), Multi-Scale Spatial Attention (7×7 conv), and Adaptive Channel Attention. Gated fusion.
  - **AFFE (Adaptive Feature Fusion Enhancement):** Replaces standard Concat. Contains Adaptive Weighted Fusion (GAP + MLP + Softmax) and Cross-Scale Enhancement.

### Performance Metrics

| Metric | Value | Baseline (YOLOv11s) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | 79.8% | 78.1% | +1.7% |
| Precision | 78.9% | 69.8% | +9.1% |
| Recall | 72.6% | 75.5% | -2.9% |
| Parameters | 11.69M | 8.99M | +30% |
| FPS (RTX 3090) | 89.3 | 137.0 | -35% |
| FPS (Jetson Xavier) | 22.1 | 32.3 | -32% |

### GC10-DET Results

| Model | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| YOLOv11s | 64.6% | 32.0% |
| MSFE-YOLO | 66.7% | 34.4% |

### Training Details
- **GPU:** NVIDIA RTX 3090 (24 GB)
- **Epochs:** 300
- **Batch size:** 16
- **Optimizer:** SGD
- **Image size:** 640×640

### Strengths
1. Effective multi-scale and frequency-domain fusion (Laplacian + Sobel)
2. Comprehensive defect-aware attention (triple-mechanism design)
3. Real-time capable (89.3 FPS on RTX 3090, 22.1 FPS on Jetson)

### Weaknesses
1. Modest mAP improvement (+1.7%) at significant computational cost (+30% params, -35% FPS)
2. Low mAP@0.5:0.95 on GC10-DET (34.4%)
3. Recall regression (72.6% vs 75.5% baseline)

---

## 10. P09 — EFEN-YOLOv8

### Full Title
"EFEN-YOLOv8: Surface defect detection network based on spatial feature capture and multi-level weighted attention"

### Authors
Meishun Wu, Jinmin Peng (corresponding), Xinyi Yu, Heng Xu, Haotian Sun

### Publication
- **Journal:** PLOS ONE
- **Year:** 2026
- **DOI:** [10.1371/journal.pone.0339617](https://doi.org/10.1371/journal.pone.0339617)

### Architecture & Base Model
- **Base:** YOLOv8n
- **Key modifications:**
  - **SAConv (Shallow Attention Convolution):** Dual-stage module — multi-scale heterogeneous convolutional kernels + adaptive pooling with attention. Replaces first two C2f modules.
  - **LSKA (Large Separable Kernel Attention):** Decomposes 2D kernels into cascaded 1D separable operations. Optimal kernel: 23 for NEU-DET, 11 for GC10-DET.
  - **WASPP (Weighted Atrous Spatial Pyramid Pooling):** Enhanced ASPP with parallel pathways (1×1, 3×3, atrous dilation 6/12/18) with sigmoid adaptive weighting.
  - **gamma-FEIoU Loss:** Combines EIoU regression + Focal classification + adaptive category weighting factor (gamma).

### Performance Metrics

| Metric | Value | Baseline (YOLOv8n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET, 9:1 split) | **80.4%** | 73.0% | **+7.4%** |
| mAP@0.5 (NEU-DET, 8:2 split) | 76.1% | — | — |
| mAP@0.5 (GC10-DET, 9:1 split) | 72.1% | 68.8% | +3.3% |

### Ablation Study

| Config | Modules Added | mAP |
|--------|---------------|-----|
| YOLOv8n baseline | — | 73.0% |
| +WASPP | WASPP | 76.5% (+3.5%) |
| +SAConv | SAConv | ~73.2% (+0.2%) |
| +LSKA | LSKA | 75.2% (+2.2%) |
| +gamma-FEIoU | gamma-FEIoU | 76.0% (+3.0%) |
| Full EFEN-YOLOv8 | All four | **80.4% (+7.4%)** |

### Statistical Robustness (5 random seeds)

| Dataset | Split | Mean mAP | Std Dev | 95% CI |
|---------|-------|----------|---------|--------|
| NEU-DET | 9:1 | 80.4% | ≤0.5% | [79.95, 80.85] |
| NEU-DET | 8:2 | 76.1% | ≤0.4% | [75.72, 76.48] |
| GC10-DET | 9:1 | 72.1% | ≤2.0% | [70.09, 74.11] |

### Per-Class Highlights
- Patch defects: 95.1% mAP (best)
- Cracking: 49.4% mAP (weakest)

### Training Details
- **GPU:** NVIDIA RTX 3060
- **Epochs:** 350
- **Batch size:** 16
- **Optimizer:** SGD (lr=0.01)
- **Framework:** PyTorch 1.12.0

### Strengths
1. Significant and statistically validated improvements (+7.4% with confidence intervals)
2. Modular and interpretable design (each component independently ablated)
3. Cross-dataset generalization (NEU-DET + GC10-DET)
4. Open access with code on GitHub

### Weaknesses
1. Increased computational cost (WASPP adds substantial params, no FPS reported)
2. Poor performance on cracking defects (49.4%)
3. SAConv alone is marginal (+0.2%), requires synergy with LSKA

---

## 11. P10 — KDM-YOLO

### Full Title
"Lightweight Visual Localization of Steel Surface Defects for Autonomous Inspection Robots Based on Improved YOLOv10n"

### Authors
Jinwu Tong (corresponding), Xin Zhang, Xinyun Lu, Han Cao, Lengtao Yao, Bingbing Gao (corresponding)

### Publication
- **Journal:** Sensors (MDPI)
- **Year:** 2026
- **Volume:** 26, Issue 7, Article 2132
- **DOI:** [10.3390/s26072132](https://doi.org/10.3390/s26072132)

### Architecture & Base Model
- **Base:** YOLOv10n
- **Parameters:** 3.29M (18.8% increase from 2.77M baseline)
- **KDM = KWConv + C2f-DRB + MSAF**
- **Key modifications:**
  - **KWConv (KernelWarehouse Convolution):** Dynamic kernel sharing — maintains N base kernels as reusable units, generates input-adaptive combination weights. K_eff = Σ(αᵢ × Kᵢ). Placed in backbone downsampling.
  - **C2f-DRB (C2f with Dilated Residual Block):** Parallel convolution branches during training, merged into single equivalent conv at inference (re-parameterization). Enlarges effective receptive field.
  - **MSAF (Multi-Scale Attention Fusion):** Two-stage design — Region-Attention Branch (coarse-to-fine grid: 1×1, 2×2, 4×4) + Pixel-Attention Branch. F_out = F_context × α + F_spatial × (1-α). Placed before Detect.

### Performance Metrics

| Metric | Value | Baseline (YOLOv10n) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET) | **95.4%** | 88.0% | **+7.4%** |
| Precision | 91.0% | 80.3% | +10.7% |
| Recall | 93.9% | 81.7% | +12.2% |
| Parameters | 3.29M | 2.77M | +18.8% |
| FPS | 155.6 | 178.2-187.2 | -14.1% |

### Ablation Study

| Experiment | Model | P (%) | R (%) | mAP@50 (%) | Params (M) | FPS |
|---|---|---|---|---|---|---|
| 1 | YOLOv10n baseline | 80.3 | 81.7 | 88.0 | 2.77 | 178.2 |
| 2 | + KWConv | 85.9 | 87.2 | 92.2 | 2.83 | 171.3 |
| 3 | + C2f-DRB | 88.5 | 90.4 | 94.7 | 2.87 | 165.2 |
| 4 | + MSAF | 90.3 | 88.7 | 94.0 | 3.15 | 158.9 |
| 5 | + KWConv + MSAF | 90.6 | 90.1 | 95.0 | 3.20 | 157.8 |
| 6 | **KDM-YOLO (all)** | **91.0** | **93.9** | **95.4** | **3.29** | **155.6** |

### Per-Class AP Improvements

| Defect Category | Baseline AP | KDM-YOLO AP | Improvement |
|----------------|-------------|-------------|-------------|
| Crazing | 0.557 | 0.805 | **+24.8%** |
| Inclusion | 0.917 | 0.988 | +7.1% |
| Rolled-in Scale | 0.898 | 0.967 | +6.9% |

### Cross-Domain Generalization (Bearing Surface Defect Dataset)
- Baseline: 80.5% mAP
- KDM-YOLO: 83.7% mAP (+3.2%, zero-shot)

### Training Details
- **GPU:** NVIDIA RTX 4060 (32 GB)
- **Epochs:** 200
- **Batch size:** 16
- **Optimizer:** SGD (lr=0.01)
- **Image size:** 200×200 (native NEU-DET resolution)
- **Framework:** PyTorch 2.0.1

### Strengths
1. Excellent accuracy-efficiency trade-off (95.4% mAP with only 3.29M params at 155.6 FPS)
2. Well-designed modular ablation with cumulative gains
3. Cross-domain generalization (bearing defect dataset, zero-shot)

### Weaknesses
1. No mAP@0.5:0.95 reported (only mAP@0.5)
2. Small, low-resolution dataset (NEU-DET: 1,800 images at 200×200)
3. Trained at native 200px — protocol difference from other papers using 640px upscale

---

## 12. P11 — YOLOv11-EMD

### Full Title
"YOLOv11-EMD: An Enhanced Object Detection Algorithm Assisted by Multi-Stage Transfer Learning for Industrial Steel Surface Defect Detection"

### Authors
Weipeng Shi, Junlin Dai, Changhe Li, Na Niu

### Publication
- **Journal:** Mathematics (MDPI)
- **Year:** 2025
- **Volume:** 13, Issue 17, Article 2769
- **DOI:** [10.3390/math13172769](https://doi.org/10.3390/math13172769)

### Architecture & Base Model
- **Base:** YOLOv11
- **EMD = Enhanced Multi-scale Dynamic** (inferred from module names)
- **Key modifications:**
  - **InnerEIoU Loss:** Improved EIoU variant for more precise bounding box regression
  - **MSDA (Multi-Scale Dilated Attention):** Dilated convolutions at multiple scales for enhanced multi-scale feature fusion
  - **C3k2_DynamicConv:** Dynamic convolution with attention-weighted kernel combinations for complex defect patterns
  - **Multi-stage Transfer Learning:** Source domain pre-training → target domain fine-tuning

### Performance Metrics

| Metric | Value | Baseline (YOLOv11) | Improvement |
|--------|-------|---------------------|-------------|
| mAP@0.5 (NEU-DET + Severstal) | **94.9%** | ~93.3% | +1.6% |
| Precision | 94.2% | ~90.7% | +3.5% |
| Recall | 86.8% | ~86.0% | +0.8% |
| mAP@0.5 (cross-scenario, NEU-DET+GC10-DET) | 79.9% | — | — |

### Transfer Learning Impact
- Training time reduction: -3.2%
- mAP increase: +8.8%

### Datasets Used
- **NEU-DET + Severstal** (combined for main experiments)
- **NEU-DET + GC10-DET** (cross-scenario generalization)

### Strengths
1. Practical multi-stage transfer learning framework (+8.8% mAP from transfer learning)
2. Comprehensive evaluation across multiple datasets
3. Targeted architectural improvements (each addresses specific challenge)

### Weaknesses
1. Limited performance metrics reported (no FPS, params, mAP@0.5:0.95)
2. Short paper (6 pages) — limited depth
3. Cross-scenario performance gap (94.9% → 79.9%)

---

## 13. Comparative Analysis

### Accuracy vs. Efficiency Trade-off

| Paper | mAP@0.5 | Params (M) | FPS | Efficiency Score |
|-------|---------|------------|-----|-----------------|
| P10 KDM-YOLO | 95.4% | 3.29 | 155.6 | ⭐⭐⭐⭐⭐ |
| P02 LAM-YOLOv10n | 94.39% | N/A | 154 | ⭐⭐⭐⭐ |
| P11 YOLOv11-EMD | 94.9% | N/A | N/A | ⭐⭐⭐ |
| P07 ASFRW-YOLO | 83.2% | 6.20 | 125 | ⭐⭐⭐⭐ |
| P03 YOLO-LSDI | 83.0% | 2.7 | 162.1 | ⭐⭐⭐⭐⭐ |
| P09 EFEN-YOLOv8 | 80.4% | N/A | N/A | ⭐⭐⭐ |
| P08 MSFE-YOLO | 79.8% | 11.69 | 89.3 | ⭐⭐ |
| P06 ELS-YOLO | 79.5% | 2.36 | N/A | ⭐⭐⭐⭐ |
| P04 Lightweight-YOLOv8 | 78.6% | 2.04 | 171.5 | ⭐⭐⭐⭐⭐ |
| P05 SCCI-YOLO | 78.6% | 1.68 | 270.2 | ⭐⭐⭐⭐⭐ |

### Novel Module Categories

| Category | Papers | Common Techniques |
|----------|--------|-------------------|
| **Lightweight backbone** | P01, P02, P04, P05 | GhostConv, GhostNet, SPD-Conv |
| **Attention mechanisms** | P02, P05, P06, P08, P09, P10 | EMA, SCSA, LSKA, MSAF, SMA |
| **Multi-scale fusion** | P01, P03, P07, P08, P09, P10 | ASF, MSFC, WASPP, MSAF, AFFE |
| **Loss function improvement** | P03, P04, P05, P07, P09, P11 | SIoU, Inner-IoU, WIoU v3, gamma-FEIoU, InnerEIoU |
| **Deformable/dynamic conv** | P03, P10, P11 | LDConv, KWConv, DynamicConv |
| **Frequency domain** | P08 | Laplacian enhancement |

### Publication Venue Distribution

| Venue | Count | Papers |
|-------|-------|--------|
| Nature Scientific Reports | 4 | P01, P02, P04, P05, P07 |
| MDPI Sensors | 2 | P08, P10 |
| MDPI Electronics | 2 | P03, P06 |
| PLOS ONE | 1 | P09 |
| MDPI Mathematics | 1 | P11 |

### Datasets Used

| Dataset | Papers Using It |
|---------|----------------|
| NEU-DET | P02, P03, P04, P05, P06, P07, P08, P09, P10, P11 |
| GC10-DET | P01, P03, P06, P08, P09, P11 |
| Severstal | P06, P11 |

### Critical Observations for DigiSteel-YOLO

1. **No paper reports robustness evaluation** — All 11 papers only test on clean images. DigiSteel-YOLO's perturbation framework (6 types × 4 levels) is a unique differentiator.

2. **Training protocol inconsistency** — P10 trains at native 200px while others upscale to 640px. This makes direct mAP comparison unfair.

3. **GhostConv is popular** — Used by P01, P02, P04 (3 papers). Validates our use in DigiSteel v2.

4. **Inner-IoU variants are trending** — P03 (Inner-CIoU), P05 (Inner-IoU), P11 (InnerEIoU) all use Inner-IoU family. Validates our Inner-WIoU choice.

5. **EMA attention is used** — P05 uses EMA (same as DigiSteel v2). Common technique, not novel.

6. **Best achievable on NEU-DET:** ~95% mAP@0.5 (P10 KDM-YOLO, but at 200px native). At 640px upscale, ~83% appears to be the competitive range (P03, P07).

---

*Document compiled by DigiSteel-YOLO research team — June 24, 2026*
