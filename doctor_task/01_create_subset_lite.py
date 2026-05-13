"""
Step 1.2-LITE: Create 10-15% LIGHTWEIGHT subset of NEU-DET dataset
Purpose: CPU-optimized task - runs in 1-2 hours instead of 4-6
Author: DigiSteel Team
Date: May 13, 2026

Optimizations:
- Reduced from 40% to 10-15% (180-270 images instead of 720)
- Smaller training epochs (20 instead of 50)
- Smaller batch size (16 instead of 32)
- Smaller model (1.2M params instead of 2.5M)
- Runs in ~1-2 hours on CPU
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.train_test_split import train_test_split
import xml.etree.ElementTree as ET

# Configuration - LIGHTWEIGHT FOR CPU
NEU_DET_PATH = Path("datasets/NEU-DET")
OUTPUT_PATH = Path("doctor_task/data/NEU-DET-subset-lite")
SUBSET_RATIO = 0.15  # 15% subset (MUCH smaller than 40%)
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def setup_directories():
    """Create output directory structure"""
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'annotations', 'labels']:
            path = OUTPUT_PATH / split / subdir
            path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created at {OUTPUT_PATH}")

def get_all_images():
    """Get list of all NEU-DET images"""
    images_dir = NEU_DET_PATH / "images"
    images = list(images_dir.glob("*.jpg"))
    print(f"Total NEU-DET images available: {len(images)}")
    return sorted([img.stem for img in images])

def create_subset(all_images):
    """Create random subset"""
    random.seed(RANDOM_SEED)
    subset_size = int(len(all_images) * SUBSET_RATIO)
    subset = random.sample(all_images, subset_size)
    print(f"✓ Selected {len(subset)} images ({SUBSET_RATIO*100}% subset)")
    print(f"  This is LIGHTWEIGHT for CPU: {len(subset)} images")
    return sorted(subset)

def split_dataset(subset):
    """Split subset into train/val/test"""
    train_val, test = train_test_split(
        subset, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO), 
        random_state=RANDOM_SEED
    )
    
    splits = {
        'train': train,
        'val': val,
        'test': test
    }
    
    print("\n=== Dataset Split ===")
    for split_name, images in splits.items():
        print(f"  {split_name}: {len(images)} images ({len(images)/len(subset)*100:.1f}%)")
    
    return splits

def copy_images_and_annotations(splits):
    """Copy images and XML annotations to subset"""
    images_src = NEU_DET_PATH / "images"
    annotations_src = NEU_DET_PATH / "annotations_voc"
    
    for split_name, image_names in splits.items():
        for img_name in image_names:
            # Copy image
            src_img = images_src / f"{img_name}.jpg"
            dst_img = OUTPUT_PATH / split_name / "images" / f"{img_name}.jpg"
            shutil.copy2(src_img, dst_img)
            
            # Copy annotation XML
            src_xml = annotations_src / f"{img_name}.xml"
            dst_xml = OUTPUT_PATH / split_name / "annotations" / f"{img_name}.xml"
            shutil.copy2(src_xml, dst_xml)
        
        print(f"✓ Copied {len(image_names)} images for {split_name}")

def convert_voc_to_yolo(xml_file, img_width, img_height, classes):
    """Convert single VOC XML to YOLO format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        
        class_id = classes[class_name]
        
        # Get bounding box
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return yolo_lines

def convert_annotations(splits):
    """Convert VOC to YOLO format"""
    classes = {
        'crazing': 0,
        'inclusion': 1,
        'patches': 2,
        'pitted_surface': 3,
        'rolled-in_scale': 4,
        'scratches': 5
    }
    
    for split_name in ['train', 'val', 'test']:
        split_dir = OUTPUT_PATH / split_name
        annotations_dir = split_dir / 'annotations'
        labels_dir = split_dir / 'labels'
        
        xml_files = list(annotations_dir.glob('*.xml'))
        
        for xml_file in xml_files:
            img_width = 200
            img_height = 200
            
            yolo_lines = convert_voc_to_yolo(xml_file, img_width, img_height, classes)
            
            label_file = labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
        
        print(f"✓ {split_name}: Converted {len(xml_files)} annotations")

def verify_subset():
    """Verify subset was created correctly"""
    print("\n=== Verification ===")
    total = 0
    for split in ['train', 'val', 'test']:
        img_count = len(list((OUTPUT_PATH / split / "images").glob("*.jpg")))
        xml_count = len(list((OUTPUT_PATH / split / "annotations").glob("*.xml")))
        txt_count = len(list((OUTPUT_PATH / split / "labels").glob("*.txt")))
        total += img_count
        
        status = "✓" if (img_count == xml_count == txt_count) else "✗"
        print(f"{status} {split}: {img_count} images, {xml_count} annotations, {txt_count} labels")
    
    print(f"\nLIGHTWEIGHT subset size: {total} images")
    print(f"Total size: {total/1800*100:.1f}% of NEU-DET")
    print(f"Estimated CPU training time: 1-2 hours (much faster!)")
    return total

if __name__ == "__main__":
    print("=" * 70)
    print("Step 1.2-LITE: Create LIGHTWEIGHT 10-15% NEU-DET Subset (CPU Optimized)")
    print("=" * 70)
    print()
    
    setup_directories()
    all_images = get_all_images()
    subset = create_subset(all_images)
    splits = split_dataset(subset)
    copy_images_and_annotations(splits)
    convert_annotations(splits)
    total = verify_subset()
    
    print("\n✓ Lightweight subset creation COMPLETE")
    print(f"  Ready for fast CPU training!")
    print(f"  Output: {OUTPUT_PATH}")
