#!/usr/bin/env python3
"""
Split dataset into train/val/test sets.

This script splits the dataset into train/val/test sets with a fixed random seed
for reproducibility. Default split ratio is 7:2:1.

Usage:
    python tools/split_dataset.py --dataset NEU-DET --seed 42
    python tools/split_dataset.py --dataset GC10-DET --seed 42
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


def split_dataset(
    dataset: str,
    data_dir: str = "datasets",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: Dataset name
        data_dir: Base data directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    dataset_dir = Path(data_dir) / dataset
    yolo_dir = dataset_dir / "yolo"

    if not yolo_dir.exists():
        print(f"ERROR: YOLO directory not found: {yolo_dir}")
        print("Run tools/voc_to_yolo.py first")
        sys.exit(1)

    # Find images and labels
    images_dir = dataset_dir / "images"
    labels_dir = yolo_dir / "labels"

    if not images_dir.exists():
        # Try to find images in the dataset directory
        for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
            images = list(dataset_dir.glob(ext))
            if images:
                images_dir = dataset_dir
                break

    if not images_dir.exists():
        print(f"ERROR: Images directory not found in {dataset_dir}")
        sys.exit(1)

    # Get all image files
    image_files = []
    for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
        image_files.extend(images_dir.glob(ext))

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Splitting {dataset} dataset")
    print(f"  Total images: {len(image_files)}")
    print(f"  Train: {train_ratio:.0%}")
    print(f"  Val: {val_ratio:.0%}")
    print(f"  Test: {test_ratio:.0%}")
    print(f"  Seed: {seed}")
    print("=" * 60)

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split indices
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")

    # Create output directories
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in splits.items():
        print(f"\n  Copying {split_name} files...")
        for img_path in files:
            # Copy image
            dst_img = yolo_dir / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy corresponding label
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dst_label = yolo_dir / "labels" / split_name / (img_path.stem + ".txt")
                shutil.copy2(label_path, dst_label)

    print("\n" + "=" * 60)
    print("  Split Complete!")
    print(f"  Output: {yolo_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Base data directory")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    split_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
