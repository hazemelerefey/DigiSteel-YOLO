<div align="center">

# DAFEGate-YOLO
### Dual-Branch Defect-Aware Feature Enhancement for Real-Time Surface Defect Detection in Hot-Rolled Flat Steel Production

<br>

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-API%20Image-2496ED?style=for-the-badge&logo=docker)](https://hub.docker.com/r/hazemelerefy/digisteel-api)
[![mAP@0.5](https://img.shields.io/badge/mAP%400.5-81.98%25-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO)
[![Params](https://img.shields.io/badge/Parameters-2.69M-blue?style=for-the-badge)]()
[![FPS](https://img.shields.io/badge/Inference-145%20FPS-yellow?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

<br>

**Product:** DigiSteel &nbsp;|&nbsp; **Team:** DigiSteel Team &nbsp;|&nbsp; **Model:** DAFEGate-YOLO

</div>

---

## 🔬 Overview

**DAFEGate-YOLO** is a novel real-time defect detection system purpose-built for **hot-rolled flat steel sheet production lines**. It introduces the **DAFEGate** (Defect-Aware Feature Enhancement Gate) module — a lightweight, plug-in backbone enhancement for YOLOv11n that explicitly addresses the core challenge of industrial steel inspection: the **morphological duality** between two fundamentally different defect categories.

> Steel surface defects split into two visual families that require entirely different detection strategies:
> - **Linear defects** (*Crazing, Scratches*): High spatial-frequency, thin cracks detectable by **edge features**
> - **Surface anomalies** (*Inclusions, Patches, Pitted Surface, Rolled-in Scale*): Low spatial-frequency irregularities detectable by **texture variance**

Standard convolutions apply the same generic filters uniformly to both. **DAFEGate-YOLO** is the first model to solve this with **morphology-specialized dual branches** fused via channel attention and an additive residual highway.

---

## ✨ Key Results

Evaluated on the **NEU-DET benchmark** (6-class steel surface defect dataset, 1,800 images) under a rigorous clean **70/20/10 split protocol** — no data leakage, no cross-dataset augmentation.

| Metric | Baseline (YOLOv11n) | **DAFEGate-YOLO** | Δ |
|:---|:---:|:---:|:---:|
| **mAP@0.5** | 79.35% | **81.98%** | **+2.63pp** |
| **mAP@0.5:0.95** | 45.83% | **46.80%** | **+0.97pp** |
| **Recall** | 76.24% | **79.79%** | **+3.55pp** |
| **Crazing AP** | 43.60% | **49.10%** | **+5.50pp** |
| **Parameters** | 2.59M | **2.69M** | **+3.7%** |
| **Inference Speed** | — | **145 FPS** | Real-time ✅ |

> **Crazing** — the hardest defect class (thin sub-pixel cracks) — improved by **+5.5pp**, directly validating the Sobel-initialized edge branch hypothesis.

### Comparison with State-of-the-Art (Fair Protocol Only)

| Model | mAP@0.5 | mAP@0.5:0.95 | Params | Speed |
|:---|:---:|:---:|:---:|:---:|
| ASFRW-YOLO | 83.2% | 46.4% | 6.20M | ~125 FPS |
| YOLO-LSDI | 83.0% | — | 2.70M | 162.1 FPS |
| EFEN-YOLOv8 | 80.4% | — | — | — |
| MSFE-YOLO | 79.8% | — | 11.69M | 89.3 FPS |
| ELS-YOLO | 79.5% | 43.2% | 2.36M | — |
| **DAFEGate-YOLO (Ours)** | **81.98%** | **46.80%** | **2.69M** | **145 FPS** |

> ✅ Highest **mAP@0.5:0.95** (localization precision) among all comparable models.  
> ✅ Outperforms all **YOLOv11n-family** models on NEU-DET under clean protocol.

---

## 🏗️ Architecture: The DAFEGate Module

DAFEGate is a lightweight plug-in module inserted at the **P3 stage** (80×80 feature maps, 256 channels) of the YOLOv11n backbone.

```
Input: x ∈ ℝ^(B × C × H × W)

┌────────────────────────────────────────────────────┐
│                  DAFEGate v4                       │
│                                                    │
│  x ──┬── EdgeAwareConv [Sobel-X/Y init] → E (C/2) │
│      │                                             │
│      └── TextureBranch [Local Variance] → T (C/2) │
│                                                    │
│  Concat(E, T) ─→ SE Channel Attention              │
│                                                    │
│  y = x + sigmoid(α) · h    [Additive Residual]    │
│       (α initialized ≈ 0.10, fully learnable)     │
└────────────────────────────────────────────────────┘

Output: y ∈ ℝ^(B × C × H × W)
```

### Why Additive > Multiplicative?

```
Additive residual (ours):  ∂y/∂x = 1.0         → gradient always flows ✅
Multiplicative gate (v3):  ∂y/∂x = σ(g) ≤ 1.0  → gradient suppressed  ❌
```

The additive residual is the decisive design choice: the skip connection **guarantees gradient flow** through all training epochs, eliminating the instability observed in multiplicative gating (DAFEGate v3).

---

## 🚀 Live Demos

### 🌐 Web Interface (Hugging Face Space)
Try the model directly in your browser — no setup required:

**👉 [https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO](https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO)**

Upload any steel surface image and receive:
- Annotated bounding boxes with class labels and confidence scores
- Semantic rejection message if the input is not a steel surface (powered by CLIP Zero-Shot Domain Guard)

### 🐳 Docker REST API
Deploy the full FastAPI server locally or on any cloud infrastructure:

```bash
docker run -p 8000:8000 hazemelerefy/digisteel-api:latest
```

**API Endpoints:**
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service health check |
| `POST` | `/predict` | Upload image → returns JSON defect list |
| `GET` | `/docs` | Swagger UI / interactive API documentation |

**Example Request:**
```bash
curl -F "file=@steel_surface.jpg" http://localhost:8000/predict
```

**Example Response:**
```json
{
  "message": "Input passed the semantic steel-surface domain check. Detected 3 defects.",
  "defects": [
    { "defect_name": "crazing", "confidence": 0.8743, "box": [12.1, 30.5, 187.2, 195.8] },
    { "defect_name": "patches", "confidence": 0.6231, "box": [50.0, 70.2, 120.4, 140.1] }
  ]
}
```

### 🛡️ Zero-Shot CLIP Semantic Domain Guard
Both deployments integrate a **semantic domain guard** using `openai/clip-vit-base-patch32` to reject out-of-domain inputs *before* running inference:

```
Non-steel input → "Domain Rejection: Input appears to be a 'text document', not a steel surface."
```

---

## 📁 Repository Structure

```
DigiSteel-YOLO/
│
├── submission_package/               ← Official artifact package for the team
│   ├── 00_Master_Reference/          ← DigiSteel Master Research Document
│   ├── 01_Publication_Figures/       ← All 11 publication figures (sequentially numbered)
│   ├── 02_Research_Reports/          ← Ablation, forensic analysis, reference papers
│   ├── 03_Model_Architecture_and_Configs/  ← dafe.py, model YAML, dataset config
│   ├── 04_Deployment_and_Demos/      ← Gradio Space + Docker FastAPI code
│   └── 05_Weights/                   ← best.pt (DAFEGate v4, mAP@0.5 = 81.98%)
│
├── digisteel/modules/dafe.py         ← DAFEGate v4 PyTorch source code
├── dafegate/modules/dafe.py          ← HF-compatible module bridge
├── huggingface_space/                ← Gradio web application source
├── Deployment/                       ← FastAPI + Docker production API
├── configs/                          ← Model & training configuration YAMLs
├── figures/                          ← Full publication figure suite
├── docs/                             ← Research reports & ablation studies
├── scripts/figure_generation/        ← Reproducible figure generation scripts
└── evals/                            ← Raw experiment results & evaluation logs
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/hazemelerefey/DigiSteel-YOLO.git
cd DigiSteel-YOLO
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r huggingface_space/requirements.txt
```

**Run the Gradio demo locally:**
```bash
python huggingface_space/app.py
```

**Run the FastAPI server locally:**
```bash
cd Deployment
uvicorn Main:app --reload
```

---

## 📊 Defect Classes (NEU-DET Benchmark)

| Class | Category | AP@0.5 (Baseline) | AP@0.5 (DAFEGate-YOLO) | Δ |
|:---|:---|:---:|:---:|:---:|
| **Crazing** | Linear / Edge | 43.6% | **49.1%** | **+5.5pp** |
| **Inclusion** | Surface / Texture | 85.2% | **88.3%** | **+3.1pp** |
| **Patches** | Surface / Texture | 91.0% | **91.7%** | +0.7pp |
| **Pitted Surface** | Surface / Texture | 79.3% | **85.0%** | **+5.7pp** |
| **Rolled-in Scale** | Surface / Texture | 77.9% | **78.8%** | +0.9pp |
| **Scratches** | Linear / Edge | 99.0% | 98.9% | −0.1pp |

---

## 🔬 Ablation Study Summary

| Version | Key Design | mAP@0.5 | vs Baseline |
|:---|:---|:---:|:---:|
| DAFE v1 | C/2 splitting, additive residual, SE attention @ P2+P3 | 78.91% | −0.44pp |
| DAFE v2 | Simplified texture branch | 78.64% | −0.71pp |
| DAFEGate v3 | Full-C channels, multiplicative gate, no SE @ P2+P3 | 80.16% | +0.81pp |
| **DAFEGate v4 (Final)** | **C/2 splitting, additive residual, SE @ P3 only** | **81.98%** | **+2.63pp** |

---

## 📄 Citation

If you use DAFEGate-YOLO or this repository in your research, please cite:

```bibtex
@article{digisteel2026dafegate,
  title     = {DAFEGate-YOLO: Dual-Branch Defect-Aware Feature Enhancement for
               Real-Time Surface Defect Detection in Hot-Rolled Flat Steel Production},
  author    = {Elerefy, Hazem and Sherif, Youssef and Salah, Mohamed and
               Esmat, Moamen and Hisham, Mahmoud},
  journal   = {arXiv preprint},
  year      = {2026},
  note      = {Supervised by Dr. Tarek Ghoneimy.
               Code: https://github.com/hazemelerefey/DigiSteel-YOLO.
               Demo: https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO}
}
```

---

## 👥 DigiSteel Team

| Role | Name |
|:---|:---|
| Lead Researcher & Architecture | Hazem Elerefy |
| Experimental Research | Youssef Sherif |
| Experimental Research | Mohamed Salah |
| Experimental Research | Moamen Esmat |
| Experimental Research | Mahmoud Hisham |
| Supervisor | Dr. Tarek Ghoneimy |

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**DAFEGate-YOLO** — Built by the **DigiSteel Team** for real-world hot-rolled flat steel production inspection.

[![HF Space](https://img.shields.io/badge/🤗_Try_it_Live-Hugging_Face-orange?style=flat-square)](https://huggingface.co/spaces/hazemelerefy/DigiSteel-YOLO)
[![Docker](https://img.shields.io/badge/Docker_API-hazemelerefy%2Fdigisteel--api-2496ED?style=flat-square&logo=docker)](https://hub.docker.com/r/hazemelerefy/digisteel-api)

</div>
