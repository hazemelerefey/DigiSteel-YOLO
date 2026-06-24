import os
from collections import Counter

label_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\labels\train"
class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
counts = Counter()
img_counts = Counter()

for f in os.listdir(label_dir):
    if not f.endswith(".txt"):
        continue
    img_counts["total"] += 1
    with open(os.path.join(label_dir, f)) as fh:
        classes_in_img = set()
        for line in fh:
            if line.strip():
                cls = int(line.strip().split()[0])
                counts[cls] += 1
                classes_in_img.add(cls)
        for c in classes_in_img:
            img_counts[c] += 1

print("=== NEU-DET Train Set ===")
print("Total images:", img_counts["total"])
print("Total annotations:", sum(counts.values()))
print()
print(f"{'Class':<20} {'Images':<10} {'Annotations':<15} {'Avg/img':<10}")
print("-"*55)
for i, name in enumerate(class_names):
    imgs = img_counts.get(i, 0)
    anns = counts.get(i, 0)
    avg = anns/imgs if imgs > 0 else 0
    print(f"{name:<20} {imgs:<10} {anns:<15} {avg:.1f}")

# Check annotation box sizes for crazing
print("\n=== Crazing Box Size Analysis ===")
crazing_boxes = []
for f in os.listdir(label_dir):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, f)) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 0:  # crazing = class 0
                w, h = float(parts[3]), float(parts[4])
                crazing_boxes.append((w, h, w*h))

if crazing_boxes:
    areas = [b[2] for b in crazing_boxes]
    ws = [b[0] for b in crazing_boxes]
    hs = [b[1] for b in crazing_boxes]
    print(f"Total crazing boxes: {len(crazing_boxes)}")
    print(f"Width  - min: {min(ws):.4f}, max: {max(ws):.4f}, mean: {sum(ws)/len(ws):.4f}")
    print(f"Height - min: {min(hs):.4f}, max: {max(hs):.4f}, mean: {sum(hs)/len(hs):.4f}")
    print(f"Area   - min: {min(areas):.6f}, max: {max(areas):.6f}, mean: {sum(areas)/len(areas):.6f}")
    
    # Size distribution
    tiny = sum(1 for a in areas if a < 0.005)
    small = sum(1 for a in areas if 0.005 <= a < 0.02)
    medium = sum(1 for a in areas if 0.02 <= a < 0.1)
    large = sum(1 for a in areas if a >= 0.1)
    print(f"\nSize distribution:")
    print(f"  Tiny   (<0.5% img):  {tiny} ({tiny/len(areas)*100:.1f}%)")
    print(f"  Small  (0.5-2%):     {small} ({small/len(areas)*100:.1f}%)")
    print(f"  Medium (2-10%):      {medium} ({medium/len(areas)*100:.1f}%)")
    print(f"  Large  (>10%):       {large} ({large/len(areas)*100:.1f}%)")

# Also check for multi-label images
print("\n=== Multi-label Images ===")
multi_label = 0
for f in os.listdir(label_dir):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, f)) as fh:
        classes_in_img = set()
        for line in fh:
            if line.strip():
                classes_in_img.add(int(line.strip().split()[0]))
        if len(classes_in_img) > 1:
            multi_label += 1
print(f"Images with multiple defect types: {multi_label}")

# Check val/test
for split in ["val", "test"]:
    label_dir_split = label_dir.replace("train", split)
    if os.path.exists(label_dir_split):
        split_counts = Counter()
        split_imgs = 0
        for f in os.listdir(label_dir_split):
            if not f.endswith(".txt"):
                continue
            split_imgs += 1
            with open(os.path.join(label_dir_split, f)) as fh:
                for line in fh:
                    if line.strip():
                        cls = int(line.strip().split()[0])
                        split_counts[cls] += 1
        print(f"\n{split}: {split_imgs} images, {sum(split_counts.values())} annotations")
        for i, name in enumerate(class_names):
            print(f"  {name}: {split_counts.get(i, 0)}")
