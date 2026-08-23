"""
NEU-DET Preprocessing Pipeline
- Apply CLAHE to all images
- Upscale to 320×320
- Copy labels unchanged
"""
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def preprocess_dataset(
    src_dir: Path,
    dst_dir: Path,
    target_size: int = 320,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8
):
    """Preprocess all images in dataset."""
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (dst_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Initialize CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    
    # Process each split
    for split in ['train', 'val', 'test']:
        img_dir = src_dir / 'images' / split
        lbl_dir = src_dir / 'labels' / split
        
        img_files = list(img_dir.glob('*.jpg'))
        print(f"\nProcessing {split}: {len(img_files)} images")
        
        for img_path in tqdm(img_files):
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # Apply CLAHE
            enhanced = clahe.apply(img)
            
            # Upscale
            upscaled = cv2.resize(
                enhanced, 
                (target_size, target_size),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Save
            output_path = dst_dir / 'images' / split / img_path.name
            cv2.imwrite(str(output_path), upscaled)
            
            # Copy label (unchanged)
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy(lbl_path, dst_dir / 'labels' / split / lbl_path.name)
    
    print(f"\n✅ Preprocessing complete: {dst_dir}")

if __name__ == '__main__':
    src = Path('datasets/NEU-DET/yolo')
    dst = Path('datasets/NEU-DET/yolo_preprocessed')
    
    preprocess_dataset(
        src_dir=src,
        dst_dir=dst,
        target_size=320,
        clahe_clip_limit=2.0,
        clahe_tile_size=8
    )
