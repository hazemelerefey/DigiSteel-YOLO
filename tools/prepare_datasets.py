#!/usr/bin/env python3
"""
Prepare datasets for DigiSteel-YOLO training.

This script converts NEU-DET (VOC XML) and GC10-DET (JSON) annotations
to YOLO format and splits them into train/val/test sets.

Usage:
    python tools/prepare_datasets.py
"""

import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Class mappings
NEU_DET_CLASSES = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}

GC10_DET_CLASSES = {
    "punching": 0,
    "weld_line": 1,
    "crescent_gap": 2,
    "water_spot": 3,
    "oil_spot": 4,
    "silk_spot": 5,
    "inclusion": 6,
    "rolled_pit": 7,
    "crease": 8,
    "waist_folding": 9,
}


def convert_neu_det_voc_to_yolo(xml_path: Path, img_width: int = 200, img_height: int = 200) -> str:
    """Convert NEU-DET VOC XML to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in NEU_DET_CLASSES:
            continue

        class_id = NEU_DET_CLASSES[class_name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        center_x = ((xmin + xmax) / 2) / img_width
        center_y = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return "\n".join(yolo_lines)


def convert_gc10_det_json_to_yolo(json_path: Path, img_width: int = 2048, img_height: int = 1000) -> str:
    """Convert GC10-DET JSON to YOLO format."""
    with open(json_path) as f:
        data = json.load(f)

    yolo_lines = []
    for obj in data.get("objects", []):
        class_name = obj.get("classTitle", "")
        if class_name not in GC10_DET_CLASSES:
            continue

        class_id = GC10_DET_CLASSES[class_name]
        points = obj.get("points", {}).get("exterior", [])

        if len(points) < 2:
            continue

        # Get bounding box from exterior points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)

        # Convert to YOLO format
        center_x = ((xmin + xmax) / 2) / img_width
        center_y = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return "\n".join(yolo_lines)


def prepare_neu_det(data_dir: Path, seed: int = 42) -> None:
    """Prepare NEU-DET dataset for YOLO training."""
    print("=" * 60)
    print("  Preparing NEU-DET Dataset")
    print("=" * 60)

    # Source directories
    train_images = data_dir / "train_images"
    train_annotations = data_dir / "train_annotations"
    valid_images = data_dir / "valid_images"
    valid_annotations = data_dir / "valid_annotations"

    # Output directories
    output_dir = data_dir / "yolo"
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process training images
    print("\n  Processing training images...")
    train_files = list(train_images.glob("*.jpg"))
    print(f"  Found {len(train_files)} training images")

    # Convert annotations
    converted = 0
    for img_path in train_files:
        xml_path = train_annotations / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue

        # Convert to YOLO format
        yolo_text = convert_neu_det_voc_to_yolo(xml_path)

        # Split: 70% train, 20% val, 10% test
        random.seed(seed + hash(img_path.name) % 10000)
        r = random.random()
        if r < 0.7:
            split = "train"
        elif r < 0.9:
            split = "val"
        else:
            split = "test"

        # Copy image
        dst_img = output_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write label
        dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")
        dst_label.write_text(yolo_text)

        converted += 1

    # Process validation images (add to train set)
    if valid_images.exists():
        print("  Processing validation images...")
        valid_files = list(valid_images.glob("*.jpg"))
        print(f"  Found {len(valid_files)} validation images")

        for img_path in valid_files:
            xml_path = valid_annotations / (img_path.stem + ".xml")
            if not xml_path.exists():
                continue

            yolo_text = convert_neu_det_voc_to_yolo(xml_path)

            # Add to train set
            dst_img = output_dir / "images" / "train" / img_path.name
            shutil.copy2(img_path, dst_img)

            dst_label = output_dir / "labels" / "train" / (img_path.stem + ".txt")
            dst_label.write_text(yolo_text)

            converted += 1

    # Count files per split
    for split in ["train", "val", "test"]:
        n_images = len(list((output_dir / "images" / split).glob("*.jpg")))
        n_labels = len(list((output_dir / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_images} images, {n_labels} labels")

    print(f"\n  ✓ Converted {converted} files")
    print(f"  Output: {output_dir}")


def prepare_gc10_det(data_dir: Path, seed: int = 42) -> None:
    """Prepare GC10-DET dataset for YOLO training."""
    print("\n" + "=" * 60)
    print("  Preparing GC10-DET Dataset")
    print("=" * 60)

    # Source directories
    gc10_dir = data_dir / "GC10-DET"
    images_dir = gc10_dir / "images"
    annotations_dir = gc10_dir / "ann"

    if not images_dir.exists() or not annotations_dir.exists():
        print(f"  ERROR: GC10-DET directories not found in {gc10_dir}")
        return

    # Output directories
    output_dir = data_dir / "yolo"
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"  Found {len(image_files)} images")

    # Convert annotations
    converted = 0
    for img_path in image_files:
        json_path = annotations_dir / (img_path.name + ".json")
        if not json_path.exists():
            continue

        # Convert to YOLO format
        yolo_text = convert_gc10_det_json_to_yolo(json_path)

        # Split: 70% train, 20% val, 10% test
        random.seed(seed + hash(img_path.name) % 10000)
        r = random.random()
        if r < 0.7:
            split = "train"
        elif r < 0.9:
            split = "val"
        else:
            split = "test"

        # Copy image
        dst_img = output_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write label
        dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")
        dst_label.write_text(yolo_text)

        converted += 1

    # Count files per split
    for split in ["train", "val", "test"]:
        n_images = len(list((output_dir / "images" / split).glob("*.jpg")))
        n_labels = len(list((output_dir / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_images} images, {n_labels} labels")

    print(f"\n  ✓ Converted {converted} files")
    print(f"  Output: {output_dir}")


def main():
    print("=" * 60)
    print("  DigiSteel-YOLO Dataset Preparation")
    print("=" * 60)

    data_dir = Path("datasets")
    seed = 42

    # Prepare NEU-DET
    neu_dir = data_dir / "NEU-DET"
    if neu_dir.exists():
        prepare_neu_det(neu_dir, seed)
    else:
        print(f"  NEU-DET not found at {neu_dir}")

    # Prepare GC10-DET
    gc10_dir = data_dir / "GC10-DET"
    if gc10_dir.exists():
        prepare_gc10_det(gc10_dir, seed)
    else:
        print(f"  GC10-DET not found at {gc10_dir}")

    print("\n" + "=" * 60)
    print("  Dataset Preparation Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train baseline: python scripts/train_baseline.py --dataset NEU-DET")
    print("  2. Train DigiSteel: python scripts/train_digisteel.py --dataset NEU-DET")


if __name__ == "__main__":
    main()
