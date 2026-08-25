# DAFESteel

**Real-Time Automated Steel-Surface Inspection and Defect Detection for Hot-Rolled Flat Steel**

## Key Metrics
- **82.0%** — mAP@0.5
- **46.80%** — mAP@0.5:0.95
- **2.69M** — parameters
- **79.79%** — Recall
- **145 FPS** — Inference
- **6** — defect classes

## ABSTRACT
DAFESteel is an end-to-end intelligent inspection service for automated detection of surface defects in hot-rolled flat steel. It combines real-time computer vision, physics-aware visualization, and decision engine to support fast and consistent industrial quality inspection.

At its core, DAFESteel uses DAFEGate-YOLO, a lightweight YOLOv11n-based detection engine designed for the visual quality of steel defects. The time damage and fine-grained surface anomalies.

## AIM OF THE WORK
Build a fast, accurate, and deployable automated inspection service that detects and localizes hot-rolled steel surface defects while reducing dependence on manual visual inspection.

## SYSTEM OVERVIEW
### Pipeline
1. **Steel Surface Input**
2. **Domain Guard Validation**
3. **DAFEGate-YOLO Inference**
4. **Defect Output Visualization**
5. **QC Interface / REST API**

### Example detection JSON shown on poster
```json
{
  "detections": [
    {
      "name": "Scratches",
      "conf": 0.91,
      "area": "medium",
      "count": 6
    },
    {
      "name": "Patches",
      "conf": 0.76,
      "area": "large",
      "count": 3
    },
    {
      "name": "Pitted Surface",
      "conf": 0.86,
      "area": "small",
      "count": 5
    }
  ]
}
```

## RESULTS
### Summary
- **82.0%** — mAP@0.5 (**±23.82 pp**)
- **46.80%** — mAP@0.5:0.95 (**±22.2 pp**)
- **79.79%** — Recall (**±3.18 pp**)
- **145 FPS** — Inference (**real-time**)

### Per-Class AP of detection engine (mAP@0.5)
- **Cr — Crazing:** 49.1
- **In — Inclusion:** 88.3
- **Pa — Patches:** 91.7
- **Ps — Pitted Surface:** 85.0
- **Rs — Rolled-in Scale:** 78.8
- **Sc — Scratches:** 98.9

## DEMOS AND PRESENTATION
- **LIVE DEMO**
- **SOURCE CODE**

Web interface for image upload and inspection (**REST API**) / real-time detection logs for production use.

## DATASET
NEU-DET hot-rolled steel surface benchmark, **1,800 images**, **six defect classes**.

### Defect classes shown
- Crazing
- Scratches
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale

## METHODOLOGY
DAFEGate-YOLOv11n with a defect-aware feature module optimized at FPS scale.

### Diagram labels shown
- **Feature map** — P7/C2f-C5
- **Edge branch** — + Soft Buffers
- **Texture branch** — LG / Gabor
- **SE attention** — & Gating
- **Defect** — Correction

### Notes / bullets
- Edge-aware branch captures fine cracks, crazing, and scratches.
- Texture-aware branch captures pits, pockmarks, and rolled-in scale.
- SE attention stabilizes features under variable conditions.
- Adaptive resolution scaling ensures real-time latency across line speeds.

### Deployment Methods
Choose one: **Dockerized FastAPI REST service** with predicts and `/logs` endpoints.

## CONCLUSION
DAFESteel moves steel-surface inspection beyond a standalone research model toward a practical, quality-inspection tool. It combines a defect-aware validation, real-time defect detection, and deployable services to bring maintain a lightweight, 2.69M-parameter detection engine.

## FUTURE WORK
- Extend data expansion across mills, steel grades, lighting conditions, and camera setups.
- Production-line integration with high-speed cameras and edge-optimized hardware.
- Defect severity grading and real-time measurement, and automated quality reports.
- Explainability, long-term drift detection, and process correction dashboard.

## USED TOOLS
- VS Code
- Manus
- ChatGPT
- Perplexity
- Docker
- Hugging Face
- GitHub

## ROADMAP FOCUS
From predictable benchmarking validation to plant-floor continuous monitoring, reporting, and predictive quality intelligence.

## Footer Text
**DAFESteel | Real-Time AI-Automated Steel Surface Inspection**

**Contact:** MADE  
**Dataset engine:** DAFEGate-YOLO  
**Server:** Web + REST API
