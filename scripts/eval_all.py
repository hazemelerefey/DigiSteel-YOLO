import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\DigiSteel-Yolo\DigiSteel-YOLO")
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

# Register custom modules so ultralytics can find them when loading checkpoints
import ultralytics.nn.tasks as tasks
from digisteel.modules.dafe import DAFE
from digisteel.modules.ema import EMA
from digisteel.modules.ghost_conv import GhostConv
from digisteel.modules.inner_wiou import InnerWIoULoss
tasks.DAFE = DAFE
tasks.EMA = EMA
tasks.GhostConv = GhostConv
tasks.InnerWIoULoss = InnerWIoULoss

models = [
    ("Baseline v1", "runs/baseline_seed42/weights/best.pt"),
    ("Baseline v2", "runs/baseline_optimized/weights/best.pt"),
    ("DigiSteel DAFE", "runs/digisteel_dafe/weights/best.pt"),
    ("DigiSteel v2", "runs/digisteel_v2/weights/best.pt"),
    ("DigiSteel v2 seed42", "runs/digisteel_v2_seed42/weights/best.pt"),
]

print("=" * 65)
print("  FRESH EVALUATION vs TRAINING CSV")
print("=" * 65)
print(f"  {'Model':<25} {'Fresh Eval':>12} {'Train CSV':>12} {'Diff':>8}")
print("  " + "-" * 59)

for name, path in models:
    p = Path(path)
    if not p.exists():
        print(f"  {name:<25} (no weights)")
        continue

    # Fresh eval
    m = YOLO(path)
    r = m.val(data="configs/data/neu_det.yaml", imgsz=800, verbose=False, workers=0)
    fresh = r.box.map50 * 100

    # Training CSV best
    csv_path = p.parent.parent / "results.csv"
    csv_best = 0
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        map_col = [c for c in df.columns if 'mAP50' in c and 'mAP50-95' not in c]
        if map_col:
            csv_best = df[map_col[0]].max() * 100

    diff = fresh - csv_best
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<25} {fresh:>8.1f}% {csv_best:>8.1f}% {sign}{diff:>6.1f}%")

print("  " + "-" * 59)
print()
print("  Fresh Eval = model.val() right now (most accurate)")
print("  Train CSV  = best validation score during training")
print("  Diff       = Fresh - Train (negative = train was inflated)")
