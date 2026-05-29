#!/usr/bin/env python3
"""
Convert PASCAL VOC XML annotations to YOLO format.

This script converts the VOC-style XML annotations used by NEU-DET and GC10-DET
to the YOLO txt format required by Ultralytics.

Usage:
    python tools/voc_to_yolo.py --dataset NEU-DET
    python tools/voc_to_yolo.py --dataset GC10-DET
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Class mappings for each dataset
CLASS_MAPPINGS = {
    "NEU-DET": {
        "crazing": 0,
        "inclusion": 1,
        "patches": 2,
        "pitted_surface": 3,
        "rolled-in_scale": 4,
        "scratches": 5,
    },
    "GC10-DET": {
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
    },
}


def voc_to_yolo(
    xml_path: Path,
    img_width: int,
    img_height: int,
    class_mapping: dict,
) -> str:
    """
    Convert a single VOC XML annotation to YOLO format.

    Args:
        xml_path: Path to XML file
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_mapping: Dictionary mapping class names to IDs

    Returns:
        YOLO format string (one line per object)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_mapping:
            print(f"  WARNING: Unknown class '{class_name}' in {xml_path}")
            continue

        class_id = class_mapping[class_name]

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


def convert_dataset(dataset: str, data_dir: str = "datasets") -> None:
    """
    Convert entire dataset from VOC to YOLO format.

    Args:
        dataset: Dataset name ("NEU-DET" or "GC10-DET")
        data_dir: Base data directory
    """
    if dataset not in CLASS_MAPPINGS:
        print(f"ERROR: Unknown dataset '{dataset}'. Use: {list(CLASS_MAPPINGS.keys())}")
        sys.exit(1)

    dataset_dir = Path(data_dir) / dataset
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    class_mapping = CLASS_MAPPINGS[dataset]

    print("=" * 60)
    print(f"  Converting {dataset} to YOLO format")
    print(f"  Classes: {len(class_mapping)}")
    print("=" * 60)

    # Find XML files
    xml_dir = dataset_dir / "annotations"
    if not xml_dir.exists():
        # Try alternative directory names
        for alt_name in ["XML", "xml", "Annotations"]:
            alt_dir = dataset_dir / alt_name
            if alt_dir.exists():
                xml_dir = alt_dir
                break

    if not xml_dir.exists():
        print(f"ERROR: Annotations directory not found in {dataset_dir}")
        print("Expected: annotations/, XML/, or Annotations/")
        sys.exit(1)

    # Create output directory
    labels_dir = dataset_dir / "yolo" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find image dimensions (assume 200x200 for NEU-DET, 2048x1000 for GC10-DET)
    if dataset == "NEU-DET":
        img_width, img_height = 200, 200
    elif dataset == "GC10-DET":
        img_width, img_height = 2048, 1000
    else:
        img_width, img_height = 200, 200  # Default

    # Convert each XML
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    converted = 0
    for xml_path in xml_files:
        yolo_text = voc_to_yolo(xml_path, img_width, img_height, class_mapping)

        # Write YOLO label
        label_path = labels_dir / (xml_path.stem + ".txt")
        label_path.write_text(yolo_text)

        converted += 1

    print(f"  ✓ Converted {converted} files")
    print(f"  Output: {labels_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert VOC to YOLO format")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (NEU-DET or GC10-DET)")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Base data directory")

    args = parser.parse_args()

    convert_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
