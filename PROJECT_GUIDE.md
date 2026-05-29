# PROJECT_GUIDE.md

**A standalone onboarding & operating guide for any AI agent (or human collaborator) picking up this workspace.**

> **Read this file first.** Once you have read it end-to-end, you should be able to: (a) understand the project, (b) understand what every other file in the workspace is for, (c) start work as a contributor without further introduction, and (d) know exactly what to ask the user and what to decide on your own.

---

## 0. The 60-second summary

- **Project:** *Comprehensive Robustness Study of Lightweight YOLO Detectors for Steel Surface Defect Detection.*
- **Working title for the team's contribution:** **DigiSteel-YOLO** — the first systematic evaluation of YOLO detector robustness to real-world image degradations in industrial steel inspection, with perturbation-aware training strategies.
- **Owner:** Five-student team in the Digilians (MCIT) Specialized Diploma in Applied AI & Data Analytics, Egypt.
  Team: **Hazem Elerefy** (lead), **Youssef Sherif**, **Mohamed Salah**, **Moamen Esmat**, **Mahmoud Hisham**.
  Supervisor: **Dr. Tarek Ghoneimy.**
- **Deadline:** Month 8 of the diploma calendar — the team has **~12 weeks left** at the moment this file is being written.
- **Status:** **Chapter 2 (Literature Review) is complete and submitted-quality.** Phase-1 implementation (the 12-week MVP) has **not yet started** — that is the next deliverable.
- **Phase-1 scope (what gets submitted at month 8):** YOLOv11n baseline + two lightweight variants (GhostConv backbone, Inner-WIoU loss), trained and evaluated on **NEU-DET + GC10-DET**, with a **6 x 4 robustness sweep** (24 evaluation points per model per dataset), **perturbation-aware training experiments**, **ONNX export**, comparison against the strongest reviewed papers, and an **open-source GitHub release with the robustness evaluation toolkit**.
- **Operating principle:** **No-guessing protocol.** Every numeric claim is traceable to a specific source. Where a metric is missing, the entry is `NR` (Not Reported); never an estimate, never an interpolation.

If you have time to read only one section, read **§7 (Phase-1 work-packages and week-by-week plan)** and **§9 (Week-1 bootstrap actions for an AI agent)**.

---

## 1. Table of contents

1. Project identity, mission, deadline
2. Workspace inventory (what every downloaded file is and when to read it)
3. Empirical foundation — the 11-paper Tier-1 corpus in 1 page
4. Five themes & five gaps (executive summary of Chapter 2)
5. The proposed contribution — DigiSteel-YOLO
6. Tech stack, environment, and conventions
7. Phase-1 work-packages and week-by-week plan
8. Operating rules (no-guessing, audit trail, file conventions, git, security)
9. Week-1 concrete bootstrap actions for an AI agent
10. Decision boundaries — what to ask the user vs decide autonomously
11. Definition of done for Phase 1
12. Risk register and common pitfalls
13. Glossary (60+ acronyms — quick reference)
14. References to the source-of-truth files
15. Appendix — quick-reference cheat-sheets

---

## 2. Workspace inventory

When you open the workspace folder, the files that should be present are listed below. If any of these is **missing**, recover it from the conversation history with the team (the originals were sent as Devin-session attachments). **Do not regenerate any file from scratch** — the existing artefacts are the source of truth and overwriting them will break the audit trail.

| File | Size (approx.) | What it is | When to read it |
|---|---|---|---|
| **`Chapter2_Literature_Review.pdf`** | ~950 KB, 39 pages | Final, submitted Chapter 2 of the dissertation. 8 figures, IEEE numeric citations [1]–[45], 12 tables, 6 RQs, 5 themes, 5 gaps, gap-of-gaps cross-tabulation, full Phase-1 plan in §2.7, three appendices. | The first time you join the project. After that, refer back when you need the original wording or the verified numbers. |
| **`Chapter2_Literature_Review.docx`** | ~840 KB | Editable backup of the PDF. | Only when you need to **edit** the chapter (e.g., supervisor revisions). |
| **`Tier1_Extraction_Matrix.xlsx`** | ~17 KB, 4 sheets | The 21-column × 11-paper extraction matrix. Each numeric cell is paired with a `src:` row pointing to the exact Table # in the source PDF. Sheets: README, ExtractionMatrix, Discrepancies, ThemeMap. | Whenever you need to recall a single specific number or the source row that justifies a claim. |
| **`references.bib`** | ~20 KB, 45 entries | BibTeX. [1]–[11] are the 11 Tier-1 papers (verified author lists from PDFs). [12]–[45] are the supporting / background references cited in §2.2, §2.3, §2.6 of Chapter 2. | When integrating Chapter 2 into a larger LaTeX manuscript or copying a reference into a new chapter. |
| **`extraction_log.md`** | ~24 KB | Per-cell audit trail. For every numeric claim in the matrix, lists: Paper ID, Field name, Value, Source location (Table #, Section #, Page #). | Whenever a reviewer or supervisor questions a specific number; never required for normal operation. |
| **`DATASETS.md`** | ~10 KB | Verified open-access download links for NEU-DET, GC10-DET, and (Phase 2) Severstal, with format notes, splits, and one-line CLI commands. | Before downloading the datasets for the first time, and whenever you need to remember the canonical split / format. |
| **`PROJECT_GUIDE.md`** | (this file) | The onboarding & operating guide. | First. |

### What is **not** yet in the workspace, and is what you (the AI / human picking this up) must build during Phase 1

- A code repository (`digisteel-yolo/` or similar) with: training pipeline, dataset loaders, the A2/A3 modifications, the perturbation toolkit, the ONNX-export script, the Colab demo notebook, a README, configuration YAMLs, and unit tests. This is the **Phase-1 Week-1 deliverable** described in §9.
- The actual datasets (`datasets/NEU-DET/`, `datasets/GC10-DET/`). Download URLs and the canonical split protocol are in **`DATASETS.md`**.
- The trained weights (`runs/`, `weights/`) — produced by training during Weeks 4–8.
- The robustness evaluation outputs (`evals/`) — produced during Weeks 5–7.
- The Pareto plots, comparison tables, and final figures for Chapter 4 (Results) — produced during Weeks 8–12.

---

## 3. Empirical foundation — the 11-paper Tier-1 corpus

These eleven peer-reviewed open-access papers (published 2025 – early 2026) are the **empirical core** of the entire project. Every architectural and methodological choice in DigiSteel-YOLO can be traced to evidence from these papers. **Memorise the paper-IDs.** They are referenced everywhere.

| ID | Short name | Base | Primary dataset(s) | One-sentence contribution |
|---|---|---|---|---|
| **P01** | **PSF-YOLO** | YOLOv11n | GC10-DET+ | GhostConv backbone + MDF-Neck + Attention Concat + Virtual Fusion Head; 25% parameter reduction. |
| **P02** | **LAM-YOLOv10n** | YOLOv10n | NEU-DET | Ghost module + SMA attention + MFFN; +3.47 pp **precision** (not mAP — common starting-brief error). |
| **P03** | **YOLO-LSDI** | YOLOv11n | NEU-DET, GC10-DET, APSPC, PCB | AMSPPF + DSAM + LDConv + Inner-CIoU; **best multi-dataset balance (4 datasets)**. |
| **P04** | **Lightweight-YOLOv8** | YOLOv8 | NEU-DET | GhostNet backbone + MPCA attention + SIoU loss; -32% params, -37% GFLOPs. |
| **P05** | **SCCI-YOLO** | YOLOv8n | NEU-DET | SPD-Conv + C2f_EMA + CCFM + Inner-IoU; **fastest in corpus (270 FPS, 1.68 M params).** |
| **P06** | **ELS-YOLO** | YOLOv11n | NEU-DET, GC10-DET, Severstal | C3k2_THK + Staged-Slim-Neck + MSDetect head; **only paper with 3 datasets including Severstal**. |
| **P07** | **ASFRW-YOLO** | YOLOv5s | NEU-DET | ASF (SSFF+CPAM+TFE) + RepNCSPELAN4 + WIoU. |
| **P08** | **MSFE-YOLO** | YOLOv11s | NEU-DET, GC10-DET | MSFC + C2MSDA (Sobel + attention) + AFFE; **only paper with edge benchmark (Jetson AGX Xavier 22.1 FPS — 75% drop vs RTX 3090).** |
| **P09** | **EFEN-YOLOv8** | YOLOv8 | NEU-DET | SAConv + LSKA + WASPP + beta-FEIoU; published in PLOS ONE (2026). **Note:** The paper claims open-source code but the GitHub link (01WineCool/YOLO) returns 404 as of May 2026. |
| **P10** | **KDM-YOLO** | YOLOv10n | NEU-DET | KWConv + C2f-DRB + MSAF; **highest accuracy in corpus (95.4% mAP@0.5)**. |
| **P11** | **YOLOv11-EMD** | YOLOv11 | Combined NEU+Severstal | InnerEIoU + MSDA + C3k2_DynamicConv; **only paper with any robustness study (qualitative).** |

**Headline numbers (cite from `Tier1_Extraction_Matrix.xlsx` if used):**
- Highest accuracy: P10 KDM-YOLO 95.4% / 3.29 M / 155.6 FPS (NEU-DET only).
- Highest speed: P05 SCCI-YOLO 270.2 FPS / 1.68 M (NEU-DET only).
- Best balanced multi-dataset: P03 YOLO-LSDI 83.0% / 162.1 FPS / 2.7 M / 6.1 GFLOPs (4 datasets).

---

## 4. Five themes and five gaps (the executive summary of Chapter 2)

### Five thematic axes (review structure of §2.4)

1. **Lightweight backbones** — Ghost-style cheap-feature blocks dominate (P01, P02, P04). Other variants: SPD-Conv (P05), C3k2_THK (P06), KWConv (P10), DynamicConv (P11). Only P08 keeps the baseline backbone untouched.
2. **Attention mechanisms** — Universal: every paper adds at least one. Roughly even split between hybrid (CBAM/EMA/MPCA/SCSA/MSAF/MSDA) and channel-only (LSKA/SE-derived).
3. **Multi-scale fusion & neck design** — Three patterns: modified PAN/FPN (P01, P02, P05, P08), attentional scale-sequence (P07, P09), slim-neck (P06).
4. **Loss-function innovations** — 6/11 swap CIoU. Inner-IoU family (P03, P05, P11) is the most popular; WIoU (P07), SIoU (P04), InnerEIoU (P09).
5. **Deployment & generalisation** — *Sparsest theme.* Only P03 and P06 train on ≥3 datasets. Only P08 reports an edge benchmark. Only P09 releases code. Only P11 runs robustness tests, and only qualitatively.

### Five research gaps (the empirical justification of the project)

| Gap | One-line statement | Corpus evidence |
|---|---|---|
| **Gap 1** | Multi-dataset validation is rare. | Only 2/11 papers (P03, P06) train+evaluate on ≥3 datasets. |
| **Gap 2** | Quantitative robustness testing is absent. | Only 1/11 (P11) reports any robustness study, and it is qualitative. |
| **Gap 3** | Metric reporting is inconsistent. | F1: 1/11, mAP@0.5:0.95: 7/11, FPS: 8/11, GFLOPs: 7/11, params: 10/11. |
| **Gap 4** | Edge-device deployment is barely studied. | Only 1/11 (P08) reports a Jetson benchmark; nobody reports Raspberry Pi or ONNX-CPU. |
| **Gap 5** | Open-source code release and unified Pareto comparison are missing. | Only 1/11 (P09) releases code; nobody compares against >4 other papers on a unified Pareto. |

### The "gap-of-gaps" central finding

> **No single paper in the corpus addresses more than two of the five gap axes simultaneously.** This is the empirical slot the team's contribution will occupy. (See Figure 8 of Chapter 2.)

---

## 5. The proposed contribution — DigiSteel-YOLO

**A comprehensive robustness study of lightweight YOLO detectors for steel surface defect detection.**

While existing research (our 11-paper corpus included) focuses on accuracy benchmarks (mAP on clean images), **no prior work systematically evaluates how these detectors perform under real-world industrial image degradations.** This is the gap our project fills.

### 5.1 Research Question

> *How robust are lightweight YOLO detectors to real-world image degradations in steel surface defect detection, and can perturbation-aware training improve deployment reliability?*

### 5.2 Core Contributions

**Contribution 1: Standardized Robustness Evaluation Framework (Primary)**

A systematic evaluation toolkit that tests detectors across 6 perturbation types x 4 severity levels = 24 evaluation points per model per dataset:

| Perturbation | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| Gaussian Blur | sigma=1 | sigma=3 | sigma=5 | sigma=7 |
| Motion Blur | k=3 | k=5 | k=7 | k=9 |
| Gaussian Noise | sigma=0.05 | sigma=0.10 | sigma=0.20 | sigma=0.30 |
| Brightness Shift | delta=-30 | delta=-50 | delta=+30 | delta=+50 |
| Contrast Reduction | factor=0.8 | factor=0.6 | factor=0.4 | factor=0.2 |
| JPEG Compression | quality=80 | quality=50 | quality=30 | quality=15 |

**This is the first such framework for steel defect detection.** The closest prior work (RobustAD, CVPR 2025) benchmarks industrial anomaly detection but not steel-specific YOLO detectors.

**Contribution 2: Perturbation-Aware Training Study (Technical)**

Training YOLOv11n with perturbations injected during training, then measuring:
- Which perturbation types improve robustness most when included in training?
- What is the optimal severity level for training perturbations?
- Is there a robustness-accuracy tradeoff?
- Which defect types benefit most from robustness training?

**This combination (evaluation + training) has not been done for steel defect detection.** The closest prior work (Liu et al. 2025) does both but for concrete crack segmentation, not steel object detection.

**Contribution 3: Multi-Dataset Validation (Methodological)**

Identical hyperparameters across NEU-DET and GC10-DET. This tests whether robustness patterns generalize across datasets — a question no prior steel defect paper has addressed.

**Contribution 4: Open-Source Benchmark (Community)**

Public GitHub repo with reproducible evaluation framework, enabling the community to benchmark their own detectors.

### 5.3 Model Variants (Not Claimed as Novel)

We evaluate multiple YOLO configurations to study robustness across architectures:

- **Baseline:** YOLOv11n (reference performance)
- **GhostConv variant:** YOLOv11n with GhostConv backbone (Han et al., CVPR 2020) — a proven lightweight convolution, not our invention
- **Inner-WIoU variant:** YOLOv11n with Inner-WIoU loss (Zhang 2023 + Tong 2023) — a principled combination of existing losses, not our invention
- **Combined variant:** GhostConv + Inner-WIoU — the lightweight configuration

These variants are used to study whether lightweight architectures are more or less robust than standard ones — they are NOT claimed as novel contributions.

### 5.4 What the team is *not* doing in Phase 1

- Claiming architectural novelty (we use proven techniques)
- Severstal training (Phase 2)
- Sobel-prior first stage (Phase 2)
- Physical edge hardware (Phase 2 — Phase 1 uses ONNX CPU only)
- INT8 quantisation (Phase 2)
- A Q1 journal paper (Phase 2 outcome)

---

## 6. Tech stack, environment, and conventions

### 6.1 Software

| Layer | Choice | Why |
|---|---|---|
| **Python** | 3.10 (or 3.11 if Colab default) | Ultralytics minimum is 3.8; 3.10 has the modern `match`-`case` and stable `typing`. |
| **Detector framework** | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (pip: `ultralytics`) | Maintains YOLOv8/v10/v11 in one tree; lets you swap backbones via YAML; built-in COCO mAP eval; built-in ONNX export. |
| **Augmentation / perturbation** | [Albumentations](https://albumentations.ai) | Standard, reproducible, supports the 4 perturbations in scope. |
| **Image I/O** | OpenCV 4 + Pillow | Ultralytics already depends on OpenCV. |
| **Tensor libs** | PyTorch 2.x with CUDA 12.x | What Ultralytics ships against. |
| **Training-curve dashboard** | Weights & Biases (free academic tier) **or** TensorBoard | Both supported by Ultralytics out of the box. WandB is recommended because it persists across Colab sessions. |
| **Edge export** | `onnx`, `onnxruntime` (CPU); Phase 2 adds `tensorrt` for Jetson | Phase 1 only needs CPU ONNX. |
| **Plots** | Matplotlib + seaborn | Already used by `build_figures.py` for Chapter 2; reuse for Chapter 4. |
| **Notebook** | JupyterLab + Google Colab | Colab Pro provides free T4 / V100. Free Kaggle has 30 h/week of P100. |

### 6.2 Hardware (Phase 1)

- **Recommended desktop:** any RTX 3060 / 3070 / 4060 (12 GB VRAM is enough for NEU-DET; GC10-DET at 640 × 640 needs ≥ 8 GB).
- **Free-tier alternative:** Google Colab Pro (T4 + 24 h sessions) **and** Kaggle (P100, 30 h/week). The 12-week plan assumes one of these is available.
- **No edge hardware required for Phase 1.** The ONNX-Runtime CPU export is sufficient to claim edge-readiness.

### 6.3 Repository conventions (when bootstrapping the code repo in Week 1)

```
digisteel-yolo/
├── README.md
├── LICENSE                       # MIT recommended
├── pyproject.toml or requirements.txt
├── .gitignore                    # MUST gitignore datasets/, runs/, weights/, *.onnx
├── configs/
│   ├── yolov11n_baseline.yaml
│   ├── yolov11n_a2_ghostconv.yaml
│   ├── yolov11n_a3_innerwiou.yaml
│   └── yolov11n_a2_a3.yaml          # the headline DigiSteel-YOLO config
├── digisteel/                       # importable package
│   ├── __init__.py
│   ├── modules/
│   │   ├── ghost_conv.py            # A2
│   │   └── inner_wiou.py            # A3
│   ├── data/
│   │   ├── neu_det.py
│   │   └── gc10_det.py
│   ├── perturbations/
│   │   ├── blur.py
│   │   ├── gaussian_noise.py
│   │   ├── brightness.py
│   │   └── jpeg.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── robustness_sweep.py
│   │   └── pareto.py
│   └── export/
│       └── onnx_export.py
├── tools/
│   ├── voc_to_yolo.py
│   ├── split_dataset.py
│   └── download_datasets.sh
├── scripts/
│   ├── train_baseline.py
│   ├── train_a2.py
│   ├── train_a3.py
│   ├── train_a2_a3.py
│   └── run_all.sh
├── notebooks/
│   ├── 01_dataset_inspect.ipynb
│   ├── 02_baseline_train.ipynb
│   ├── 03_robustness_sweep.ipynb
│   └── 99_colab_demo.ipynb          # public-facing drag-and-drop demo
└── tests/
    ├── test_ghost_conv.py
    ├── test_inner_wiou.py
    └── test_perturbations.py
```

**Naming rules**

- Lower_snake_case for files and Python identifiers.
- Configs are YAML; one-config-per-experiment, never edited mid-experiment.
- A run identifier is `<arch>_<dataset>_<seed>_<timestamp>` (e.g., `yolov11n_a2_a3_NEU_seed42_20260601_103000`).
- Trained weights live in `runs/<run_id>/weights/` (Ultralytics default) and **must not** be committed to git.

### 6.4 Reproducibility

Every published result must be reproducible by:

```
git clone https://github.com/<team-account>/digisteel-yolo.git
cd digisteel-yolo
pip install -r requirements.txt
bash tools/download_datasets.sh           # NEU-DET + GC10-DET
python tools/voc_to_yolo.py
python tools/split_dataset.py --seed 42
python scripts/train_a2_a3.py --dataset NEU-DET --epochs 200 --seed 42
```

If the above sequence does not produce the headline numbers (within ±0.5 pp mAP), the result is not published.

---

## 7. Phase-1 work-packages and week-by-week plan

Already in §2.7 / Appendix C of Chapter 2. Reproduced here for convenience.

### 7.1 Work-packages (one lead per WP)

| WP | Lead | Scope |
|---|---|---|
| **WP1** | Hazem Elerefy | Training pipeline + the A2 / A3 architectural changes; integration; ONNX export. |
| **WP2** | Youssef Sherif | NEU-DET + GC10-DET preprocessing, training runs, model-zoo curation, comparison-table data. |
| **WP3** | Mohamed Salah | Robustness toolkit (4 perturbations × 3 levels), evaluation sweep, perturbation figures. |
| **WP4** | Moamen Esmat | Environment / GPU access, dataset versioning, training-curve dashboard, Pareto-plot infrastructure, ONNX-runtime CPU verification. |
| **WP5** | Mahmoud Hisham | Metrics aggregation, figures, written report (intro / methodology / results / discussion), thesis integration. |

### 7.2 Twelve-week schedule

| Week | Activity |
|---|---|
| 1 | WP1 repo skeleton + baseline YOLOv11n verification on NEU-DET. WP2 NEU-DET preprocessing. WP3 perturbation-library study (Albumentations). WP4 GPU/env setup. WP5 intro draft. |
| 2 | WP1 implements A2 (GhostConv weight-sharing). WP2 GC10-DET preprocessing. WP3 first blur+noise prototype. WP4 dataset versioning + logging. WP5 baseline figures. |
| 3 | WP1 implements A3 (Inner-WIoU) + smoke test. WP2 first multi-dataset run. WP3 brightness + JPEG perturbations. WP4 training-curve dashboard. WP5 methodology section. |
| **4 (Milestone 1: arch freeze)** | WP1 hyper-parameter search. WP2 NEU-DET full training. WP3 finalises 4 × 3 perturbation matrix. WP4 hyperparam logging. WP5 results-section template. |
| 5 | WP1 tunes on GC10-DET. WP2 GC10-DET full training. WP3 robustness sweep on NEU-DET. WP4 confusion-matrix + per-class plots. WP5 architecture section. |
| 6 | WP1 freezes model for MVP. WP2 model-zoo organisation. WP3 robustness sweep on GC10-DET. WP4 metrics aggregation. WP5 results section. |
| 7 | WP1 retrains P09 EFEN-YOLOv8 (only open-source competitor). WP2 comparison-table data. WP3 comparison robustness sweep. WP4 Pareto plots. WP5 comparison section. |
| **8 (Milestone 2: results freeze)** | WP1 ONNX export + verify identical inference. WP2 test-set re-eval. WP3 finalises robustness figures. WP4 Pareto plots (mAP/FPS, mAP/Params). WP5 discussion section. |
| 9 | WP1 Colab demo notebook. WP2 inference-time benchmarks (desktop GPU). WP3 per-perturbation discussion. WP4 failure-case visualisations. WP5 tighten lit-review hooks to MVP results. |
| 10 | WP1 code cleanup + README + license. WP2 dataset-prep README. WP3 robustness-toolkit README. WP4 benchmarking-toolkit README. WP5 full report draft. |
| 11 | WP1 reproducibility check (clone + retrain on clean machine). WP2 freezes results table. WP3 freezes robustness figures. WP4 freezes Pareto plots. WP5 review/edit pass. |
| **12 (Milestone 3: submission)** | WP1–WP4 demo rehearsal & live-demo support. WP5 final report submission. |

### 7.3 Cadence and rituals

- **Daily:** silent work, async updates on a shared channel.
- **Mondays:** 30-minute stand-up — every WP lead reports last-week status, this-week plan, blockers.
- **Fridays:** 60-minute integration check-in — Hazem (WP1) integrates that week's contributions on the `main` branch and runs the full smoke test.
- **Milestones (weeks 4, 8, 12):** in-person/video review with Dr. Tarek Ghoneimy.

---

## 8. Operating rules

These are non-negotiable. They are the reason Chapter 2 is supervisor-trustworthy and they will be the reason Chapter 4 (Results) is too.

### 8.1 The no-guessing protocol

- **Never interpolate, estimate, or infer a number.** If a paper does not report it, write `NR` and move on.
- **Every number in any output table must be paired with a source.** For literature numbers, the source is the Table # of the original PDF (already logged in `extraction_log.md`). For our own numbers, the source is the run-ID of the experiment that produced it.
- **Discrepancies are flagged, never silently corrected.** See `Tier1_Extraction_Matrix.xlsx` "Discrepancies" sheet and Appendix A of Chapter 2.

### 8.2 Audit trail

- All experimental outputs land in `runs/<run_id>/`, including: config snapshot, weights, training log, val/test predictions, COCO-format eval JSON, tensorboard / wandb run-ID.
- All robustness results land in `evals/<run_id>/<perturbation>/<level>/`, with the original perturbed images saved (or the perturbation seed, so they can be regenerated bit-for-bit).
- All published numbers must point back to a `<run_id>`. If a number is in a paper figure but cannot be traced to a run-ID, the figure does not get included.

### 8.3 Git conventions

- Branch from `main` for every change: `feat/a2-ghostconv`, `fix/voc-converter`, `docs/methodology`.
- Pull-requests must be reviewed by at least one other team member.
- `main` is protected; nothing is merged without CI passing (CI = pytest + ruff + a 1-epoch smoke training run on a tiny subset of NEU-DET).
- Tag each milestone (`v0.1-arch-freeze`, `v0.2-results-freeze`, `v1.0-submission`).

### 8.4 Security / IP

- **Never commit datasets** (size + licence). Datasets live in `datasets/` which is in `.gitignore`.
- **Never commit Kaggle API keys, WandB tokens, GitHub PATs.** Use environment variables (`.env` is in `.gitignore`); the README documents which are required.
- **Open-source release** — once the chapter is graded, the repository becomes public. Pick MIT for code, CC BY 4.0 for figures. Do not redistribute the datasets — link to the official Kaggle / NEU pages.

### 8.5 Writing conventions

- Citations: IEEE numeric, `[N]`. The cite-keys in `references.bib` are stable; do not rename them.
- Numbers in prose: round only at presentation time (one decimal for mAP, integer for FPS, one decimal for parameter count in millions). Always carry the precision of the source.
- No claim in prose without a citation. No citation without a verified DOI in `references.bib`.

---

## 9. Week-1 concrete bootstrap actions for an AI agent

If you (the AI agent or human) are reading this guide for the first time and the workspace contains **only the documentation files** described in §2 and **no code yet**, this is the exact sequence of actions to take in Week 1.

> Treat this section as a checklist. Do not skip steps. After each step, run the check listed in the **Verify** column.

### 9.1 Day 1 — Environment + dataset

| # | Action | Verify |
|---|---|---|
| 1 | Read `Chapter2_Literature_Review.pdf` end-to-end (or at least §§2.1, 2.6, 2.7). | You can answer "what is gap 4 and which paper exposes it?" without rereading. |
| 2 | Read `DATASETS.md` end-to-end. | You know the canonical 7:2:1 split, seed = 42, and the difference between NEU-DET and GC10-DET annotation formats. |
| 3 | Set up a clean Python 3.10 venv. | `python -V` reports 3.10.x; `pip list` is empty except for pip/setuptools. |
| 4 | `pip install ultralytics albumentations onnx onnxruntime opencv-python pyyaml matplotlib seaborn pandas` | `python -c "import ultralytics; print(ultralytics.__version__)"` succeeds. |
| 5 | Create the Kaggle API token and place at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars). | `kaggle datasets list -s neu` succeeds. |
| 6 | Download NEU-DET and GC10-DET into `datasets/` per `DATASETS.md` §6. | `ls datasets/NEU-DET/ datasets/GC10-DET/` shows ~1,800 + ~2,300 image files respectively. |

### 9.2 Day 2 — Repo skeleton

| # | Action | Verify |
|---|---|---|
| 7 | `git init digisteel-yolo && cd digisteel-yolo`. Create the directory tree shown in §6.3. | `tree -L 2` matches §6.3. |
| 8 | Add `.gitignore` covering `datasets/`, `runs/`, `weights/`, `*.onnx`, `__pycache__/`, `.env`, `*.pt`, `wandb/`. | `git status` does not list any large files. |
| 9 | Add `requirements.txt` (or `pyproject.toml`) with the libraries installed in step 4 plus `pytest`, `ruff`, `black`. | `pip install -e .` succeeds. |
| 10 | Add `LICENSE` (MIT) and a one-page `README.md` with: project name, team, supervisor, 1-paragraph summary, "see PROJECT_GUIDE.md for full context", install steps, 1-line train command, link to the demo Colab. | `cat README.md` reads correctly. |
| 11 | Configure pre-commit / ruff / black; commit the empty skeleton as `v0.0-skeleton`. | `git tag -l` shows `v0.0-skeleton`. |

### 9.3 Day 3 — Dataset converters

| # | Action | Verify |
|---|---|---|
| 12 | Write `tools/voc_to_yolo.py` — converts PASCAL-VOC XML to Ultralytics YOLO `.txt`. Standard implementation; many references online. Class-id maps for NEU-DET and GC10-DET hard-coded as constants. | `python tools/voc_to_yolo.py --dataset NEU-DET` produces `datasets/NEU-DET/yolo/labels/train/*.txt`. |
| 13 | Write `tools/split_dataset.py` — random 7:2:1 split, seed=42, writes `datasets/<name>/yolo/{images,labels}/{train,val,test}/`. | After running on NEU-DET: 1260/360/180 images in train/val/test (verify with `ls | wc -l`). |
| 14 | Write `tools/download_datasets.sh` — wraps the Kaggle CLI calls so anyone can re-run end-to-end. | `bash tools/download_datasets.sh` on a clean machine reproduces the layout in step 6. |
| 15 | Add unit tests in `tests/test_voc_to_yolo.py` (5 assertions: each NEU-DET class is found, bbox coords are normalised, etc.). | `pytest -q` passes. |

### 9.4 Day 4 — Baseline training

| # | Action | Verify |
|---|---|---|
| 16 | Write `configs/yolov11n_baseline.yaml` — Ultralytics-format dataset YAML pointing at `datasets/NEU-DET/yolo/`. | `yolo cfg=configs/yolov11n_baseline.yaml mode=val` doesn't crash. |
| 17 | Write `scripts/train_baseline.py` — wraps `from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.train(data=…, epochs=200, imgsz=640, seed=42, project='runs', name='baseline_neu')`. | Smoke-train with `epochs=1` on a 50-image subset; mAP reported. |
| 18 | Run a real 200-epoch baseline on NEU-DET. | Training completes without OOM; final mAP@0.5 is in the 70–82% range (consistent with the corpus). |
| 19 | Save the run, tag `v0.1-baseline`. Document the result in `runs/baseline_neu/RESULTS.md`. | The baseline number is reproducible by anyone running the same command. |

### 9.5 Day 5 — A2 GhostConv module

| # | Action | Verify |
|---|---|---|
| 20 | Implement `digisteel/modules/ghost_conv.py` — GhostConv block (Han 2020). Use the equations from the GhostNet paper; test against the published parameter count for an isolated 64→64 block. | `pytest tests/test_ghost_conv.py` passes. |
| 21 | Add the weight-sharing extension across pyramid stages P3 / P4 / P5: a single `nn.Conv2d` instance whose weights are reused in three forward passes at three resolutions. Add a unit test asserting the param count is exactly 1/3 of three independent GhostConv blocks of the same shape. | Test passes. |
| 22 | Write `configs/yolov11n_a2_ghostconv.yaml` — references the custom module via Ultralytics' YAML-defined custom-module hook. | `yolo cfg=configs/yolov11n_a2_ghostconv.yaml mode=train epochs=1 …` smoke-trains. |
| 23 | Smoke-train A2 (epochs=10, imgsz=640) on NEU-DET. | Training completes; param count is 25–35% lower than baseline (consistent with P01's 25% claim). |

### 9.6 Day 6 — A3 Inner-WIoU loss

| # | Action | Verify |
|---|---|---|
| 24 | Implement `digisteel/modules/inner_wiou.py` — `InnerWIoULoss(λ)` returning `λ · Inner-IoU + (1−λ) · WIoU_v3`. Reference equations: Inner-IoU (Zhang 2023), WIoU v3 (Tong 2023). | `pytest tests/test_inner_wiou.py` passes for: λ=1 → exact Inner-IoU, λ=0 → exact WIoU_v3, λ=0.5 → weighted average; gradients flow. |
| 25 | Patch the Ultralytics `BboxLoss` class (monkey-patch in a `register_loss(...)` call inside the training script) so the model uses Inner-WIoU during training. | Smoke-train (epochs=10) shows decreasing loss and a non-trivial mAP. |
| 26 | Write `configs/yolov11n_a3_innerwiou.yaml` and `configs/yolov11n_a2_a3.yaml`. | Both smoke-train. |

### 9.7 Day 7 — Reflect, rest, commit

| # | Action | Verify |
|---|---|---|
| 27 | Open a draft PR titled "Phase-1 Week-1 bootstrap" containing all of the above. | All CI checks pass. |
| 28 | Tag `v0.2-modules-implemented`. | `git tag -l` shows the tag. |
| 29 | Update the team channel: "Week 1 done, A2+A3 smoke-trained, ready for Week-2 multi-dataset training." | Team can pick up Week 2 from `main`. |

If you, the AI agent, can complete steps 1–29 inside a single working week, you are on track. If you can complete them inside a single Devin session (8 hours), you are ahead.

---

## 10. Decision boundaries — what to ask the user vs decide autonomously

### 10.1 Decide autonomously

- File-naming, code organisation, test names — follow §6.3.
- Choice of an additional Python utility library when one of the listed libraries is missing a feature (e.g., `tqdm`, `rich`, `pyyaml`).
- Hyper-parameters within the ranges already cited from the reviewed papers (LR ∈ [1e-4, 1e-2], batch ∈ [8, 64], imgsz ∈ {512, 640, 800}). Document the chosen value and why.
- Choice of evaluation seed for ablation runs (default = 42; can vary to {0, 7, 42, 123, 2024} if reporting variance).
- Whether to use Colab vs Kaggle vs local GPU on a given day (whichever is fastest).
- Whether to retrain a baseline that has already been trained (no — re-use the run unless the dataset or hyper-parameters changed).

### 10.2 Ask the user (Hazem or supervisor) before proceeding

- **Anything that changes the scope of Phase 1** — adding Severstal, adding the A1 Sobel-prior module, adding INT8 quantisation. The 12-week budget is fixed.
- **Anything that changes the architectural baseline** — switching from YOLOv11n to YOLOv8s, swapping the loss to something other than Inner-WIoU, adding modules not in §2.7.
- **Anything that costs money** — paid GPU credits, paid Kaggle, paid WandB tier, edge hardware.
- **Anything that is irreversible** — pushing the repo public before the chapter is graded; force-pushing to `main`.
- **Anything legally / ethically sensitive** — using a dataset that requires a separate licence agreement; including images from non-public sources.

---

## 11. Definition of done — Phase 1

The Phase-1 submission at month 8 is "done" when **every one of the following** is true:

1. The code repository is public (or set to "private until grading"), tagged `v1.0-submission`, with a green CI badge.
2. `bash tools/download_datasets.sh && python scripts/run_all.sh` reproduces all headline results on a clean machine in ≤ 12 hours of GPU time.
3. The `runs/` directory contains the four training outputs:
   - `yolov11n_baseline_NEU_seed42_*`
   - `yolov11n_a2_a3_NEU_seed42_*`
   - `yolov11n_baseline_GC10_seed42_*`
   - `yolov11n_a2_a3_GC10_seed42_*`
   plus `yolov11n_efen_*` for the P09 retrain comparator.
4. The `evals/` directory contains the 4 × 3 × 2 = 24 robustness-sweep result files (4 perturbations × 3 levels × 2 datasets) for both baseline and A2+A3 (so 48 files total).
5. The Pareto plots (mAP@0.5 vs FPS, mAP@0.5 vs params) in `figures/` include DigiSteel-YOLO and at least P03, P05, P09, P10.
6. `digisteel-yolo.onnx` has been exported and gives bit-identical predictions to the PyTorch model on a held-out 50-image set (within a 1e-4 numerical tolerance for the bounding-box coordinates).
7. `notebooks/99_colab_demo.ipynb` runs end-to-end in Google Colab on a clean Python 3.10 runtime, including the dataset download cell.
8. The full eight-metric set is reported for the headline DigiSteel-YOLO model on both datasets, with **no `NR` cells** in the team's own results.
9. The Chapter 4 (Results) draft is integrated with Chapter 2 and the team's submitted dissertation PDF is generated.
10. The supervisor (Dr. Tarek Ghoneimy) has reviewed the dissertation and approved submission.

---

## 12. Risk register and common pitfalls

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Free-tier GPU access (Colab Pro 24 h cap) | High | Use Kaggle (30 h/week free) + Lightning AI free tier as backup. RTX 3060/3070 desktop is sufficient. |
| R2 | Severstal too large for free GPU | High | Severstal already deferred to Phase 2. |
| R3 | One student overloaded | Medium | Hazem rebalances at week-4 stand-up; the robustness sweep is a generic script anyone can run. |
| R4 | Defending the architecture novelty | Low | A2 and A3 each cite published papers showing 0.5–1.5 pp gain on steel data; the validation suite is the principal novelty. |
| R5 | GC10-DET label noise | Medium | Cite the known issue and report results with and without minimal label cleaning. |
| R6 | Reviewer requests Severstal in Phase 1 | Medium | Phase-1 already has a Severstal-readiness section; if reviewers insist, the same training pipeline runs unmodified — adds ~10 days. |
| R7 | Ultralytics YOLO API breaks between versions | Medium | Pin `ultralytics==X.Y.Z` in `requirements.txt` and document the version in `README.md`. Re-test every upgrade. |
| R8 | Custom-module YAML hook silently uses the default Conv | High (easy to miss) | After every config change, run `model.info(verbose=True)` and confirm `GhostConv` appears in the layer list. |
| R9 | Loss patch is bypassed by `model.train()`'s default loss | High (Ultralytics rebuilds the loss internally) | Always patch `BboxLoss` *after* `YOLO(...)` and *before* `.train()`; assert via a forward-pass diff that the patched loss is being used (test in `tests/test_inner_wiou_integration.py`). |
| R10 | Robustness images saved at the wrong precision | Medium | Save perturbed images as PNG (lossless) for verification; for the actual eval pass them as in-memory tensors so JPEG conversion is **only** the JPEG perturbation, never an artefact. |
| R11 | mAP discrepancy between PyTorch and ONNX | Low | The ONNX export script must include a verification cell that runs both the PyTorch and ONNX models on the same 50-image batch and asserts max-bbox-coord-diff < 1e-4 and class-decision-agreement = 100%. |
| R12 | Ground-truth class-id off-by-one between NEU-DET and GC10-DET configs | Medium | Always use the class-id maps in `digisteel/data/<dataset>.py` (single source of truth); never re-declare them. |

---

## 13. Glossary (60+ acronyms — quick reference)

This is a fast lookup. The full version with expansions is in the Glossary section of `Chapter2_Literature_Review.pdf`.

`AFFE` Adaptive Feature-Fusion Enhancement · `AMSPPF` Aggregated Multi-Scale Spatial Pyramid Pooling-Fast · `APSPC` Aluminum Profile Surface Defect Detection (dataset) · `ASF` Attentional Scale Sequence Fusion · `BiFPN` Bidirectional Feature Pyramid Network · `CBAM` Convolutional Block Attention Module · `CCFM` Cross-Scale Feature Fusion Module · `CIoU` Complete Intersection-over-Union loss · `CNN` Convolutional Neural Network · `COCO` Common Objects in Context · `CPAM` Channel and Position Attention Module · `CSP` Cross-Stage Partial · `CV` Cross-Validation · `DAGM` synthetic-defect benchmark · `DSAM` Deformable Spatial Attention Module · `ECA` Efficient Channel Attention · `EMA` Efficient Multi-scale Attention · `FLOPs` floating-point operations · `FPN` Feature Pyramid Network · `FPS` frames per second · `GFLOPs` 10⁹ FLOPs per inference · `GC10-DET` Generic Cold-rolled steel ten-class Defect Dataset · `GhostConv` Ghost convolution (cheap-feature operation) · `HOG` Histogram of Oriented Gradients · `IoU` Intersection over Union · `Inner-IoU` Inner-IoU regression loss with auxiliary bounding box · `Jetson` NVIDIA edge GPU family · `KolektorSDD` Kolektor Surface-Defect Dataset · `LDConv` Linear Deformable Convolution · `LSKA` Large Separable Kernel Attention · `mAP` Mean Average Precision · `mAP@0.5` mAP at IoU = 0.5 · `mAP@0.5:0.95` COCO mAP averaged over IoU 0.5..0.95 · `MDF` Multi-Dimensional Fusion · `MFFN` Multi-scale Feature-Fusion Network · `MPCA` Multi-Path Channel Attention · `MSAF` Multi-Scale Attentional Fusion · `MSDA` Multi-Scale Deformable Attention · `MSFC` Multi-Scale Frequency-enhanced Convolution · `MSDetect` Multi-Scale Detection head (ELS-YOLO) · `NEU-DET` Northeastern University six-class steel defect dataset · `NMS` Non-Maximum Suppression · `NR` Not Reported (used in tables) · `ONNX` Open Neural Network Exchange · `PAN` Path Aggregation Network · `PCB` Printed Circuit Board (defect dataset) · `Pareto` accuracy-vs-efficiency frontier · `Pi 5` Raspberry Pi 5 · `PRISMA` Preferred Reporting Items for Systematic reviews and Meta-Analyses · `PR` Precision-Recall · `R-CNN` Region-based CNN · `RepNCSPELAN4` re-parameterised CSP-ELAN block · `RGB` red-green-blue colour image · `ROI` Region of Interest · `SAConv` Switchable Atrous Convolution · `SCSA` Spatial-Channel Synergistic Attention · `SE` Squeeze-and-Excitation · `Severstal` Severstal Steel Defect Detection (Kaggle) · `SIoU` SCYLLA-IoU regression loss · `SMA` Spatial Multi-scale Attention · `SPD-Conv` Space-to-Depth Convolution · `SPPF` Spatial Pyramid Pooling-Fast · `SSD` Single Shot MultiBox Detector · `TFE` Triple Feature Encoder · `WASPP` Weighted Atrous Spatial Pyramid Pooling · `WIoU` Wise-IoU regression loss · `YOLO` You Only Look Once · `YOLOv8 / v10 / v11` YOLO version 8 / 10 / 11.

---

## 14. References to the source-of-truth files

If something in this guide is ambiguous, **the file listed in the right column is authoritative** and overrides this guide.

| Topic | Authoritative file |
|---|---|
| Project narrative, RQs, themes, gaps, contribution | `Chapter2_Literature_Review.pdf` (especially §§2.1, 2.6, 2.7) |
| Per-paper extracted numbers and per-cell sources | `Tier1_Extraction_Matrix.xlsx` |
| Per-paper per-cell audit trail | `extraction_log.md` |
| BibTeX cite-keys, DOIs, author lists | `references.bib` |
| Dataset URLs, formats, splits | `DATASETS.md` |
| Phase-1 work-package owners, weekly schedule, milestones | `Chapter2_Literature_Review.pdf` Appendix C |
| Phase-2 backlog (months 9–12, post-graduation) | `Chapter2_Literature_Review.pdf` §2.7.4 |

---

## 15. Appendix — quick-reference cheat-sheets

### 15.1 Common Ultralytics commands

```bash
# Sanity-check baseline:
yolo task=detect mode=train model=yolo11n.pt data=configs/yolov11n_baseline.yaml epochs=1 imgsz=640

# Full baseline:
yolo task=detect mode=train model=yolo11n.pt data=configs/yolov11n_baseline.yaml \
     epochs=200 imgsz=640 batch=16 seed=42 project=runs name=baseline_neu

# Validate:
yolo task=detect mode=val model=runs/baseline_neu/weights/best.pt data=configs/yolov11n_baseline.yaml

# ONNX export:
yolo export model=runs/a2_a3_neu/weights/best.pt format=onnx opset=12 simplify=True
```

### 15.2 Albumentations one-liners for the four perturbations

```python
import albumentations as A

blur      = lambda sigma: A.GaussianBlur(blur_limit=(2*sigma+1,)*2, sigma_limit=(sigma, sigma), p=1.0)
gnoise    = lambda sigma: A.GaussNoise(var_limit=((sigma*255)**2, (sigma*255)**2), p=1.0)
brightness= lambda delta: A.RandomBrightnessContrast(brightness_limit=(delta/255.0, delta/255.0),
                                                    contrast_limit=0, p=1.0)
jpeg      = lambda q: A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
```

(Implement once in `digisteel/perturbations/`, then call from `digisteel/eval/robustness_sweep.py`.)

### 15.3 The 4 × 3 robustness grid

| Perturbation | Level 1 (mild) | Level 2 (moderate) | Level 3 (severe) |
|---|---|---|---|
| Gaussian blur | σ = 1 | σ = 3 | σ = 5 |
| Gaussian noise | σ = 0.05 (≈ 13/255) | σ = 0.10 | σ = 0.20 |
| Brightness drift | Δ = −50/255 | Δ = +20/255 | Δ = +50/255 |
| JPEG compression | Q = 80 | Q = 50 | Q = 30 |

The output of the sweep is one mAP@0.5 value per (model × dataset × perturbation × level), saved as a single CSV `evals/<run_id>/robustness.csv`. Plot it as four lines (one per perturbation) of mAP vs level.

### 15.4 The eight-metric reporting template

```
| Metric         | Train | Val   | Test  |
|---------------:|------:|------:|------:|
| mAP@0.5        |  …    |  …    |  …    |
| mAP@0.5:0.95   |  …    |  …    |  …    |
| Precision      |  …    |  …    |  …    |
| Recall         |  …    |  …    |  …    |
| F1             |  …    |  …    |  …    |
| FPS            |   —   |   —   |  …    |   (test set inference)
| Parameters (M) |  …    |  …    |  …    |   (constant)
| GFLOPs         |  …    |  …    |  …    |   (constant; @ imgsz=640)
```

No `NR` is allowed in this table for the team's own results. If any cell is unfilled, the model is not yet ready to publish.

---

**End of PROJECT_GUIDE.md.** If anything in this guide is unclear, the team's primary contact is **Hazem Elerefy** (team leader) and the supervisor of record is **Dr. Tarek Ghoneimy.**
