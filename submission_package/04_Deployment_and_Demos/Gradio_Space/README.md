---
title: DigiSteel YOLO DAFEGate v4
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# DigiSteel YOLO DAFEGate v4

Professional Hugging Face Space package for the best recorded DigiSteel-YOLO model.

## Model

- Architecture: YOLOv11n-style detector with the custom DAFEGate v4 module at P3.
- Task: steel surface defect detection.
- Classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- Input size: 640 px.

## Test Performance

| Metric | Value |
| --- | ---: |
| mAP@0.5 | 81.98% |
| mAP@0.5:0.95 | 46.80% |
| Precision | 72.55% |
| Recall | 79.79% |
| Training time | 1.5 h |

## Files

- `app.py`: Gradio inference app.
- `models/best.pt`: best DAFEGate v4 checkpoint.
- `dafegate/modules/dafe.py`: custom module required to load the checkpoint.
- `requirements.txt`: runtime dependencies for the Space.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

## Notes

This model was trained and evaluated on the NEU-DET defect classes. The Space includes an integrated **Zero-Shot Semantic Domain Guard (CLIP)** that automatically detects and rejects out-of-distribution inputs (e.g., natural photos, human faces, UI screenshots, graphic banners) before running the YOLO detector.

The semantic gate provides a robust deployment safeguard. For production and research extensions, negative/background samples can also be incorporated into model training.

