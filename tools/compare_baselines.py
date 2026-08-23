"""Compare pretrained YOLO baseline model sizes."""
from ultralytics import YOLO

models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
for m in models:
    try:
        model = YOLO(m)
        p = sum(x.numel() for x in model.model.parameters())
        print(f"{m}: params={p/1e6:.2f}M")
    except Exception as e:
        print(f"{m}: ERROR {e}")
