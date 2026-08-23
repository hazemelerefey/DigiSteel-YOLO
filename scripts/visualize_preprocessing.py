"""
Visual comparison: Original vs Preprocessed (CLAHE + Upscale)
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
orig_dir = Path('datasets/NEU-DET/yolo/images/train')
prep_dir = Path('datasets/NEU-DET/yolo_preprocessed/images/train')

# Select 4 sample images (one from each class if possible)
sample_names = ['crazing_1.jpg', 'inclusion_1.jpg', 'patches_1.jpg', 'pitted_surface_1.jpg']

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle('Original vs Preprocessed (CLAHE + 320x320)', fontsize=16, fontweight='bold')

for i, name in enumerate(sample_names):
    orig_path = orig_dir / name
    prep_path = prep_dir / name
    
    if not orig_path.exists() or not prep_path.exists():
        continue
    
    # Load images
    orig = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
    prep = cv2.imread(str(prep_path), cv2.IMREAD_GRAYSCALE)
    
    # Display original
    axes[i, 0].imshow(orig, cmap='gray')
    axes[i, 0].set_title(f'Original: {name}\n{orig.shape[1]}x{orig.shape[0]}', fontsize=10)
    axes[i, 0].axis('off')
    
    # Display preprocessed
    axes[i, 1].imshow(prep, cmap='gray')
    axes[i, 1].set_title(f'Preprocessed: {name}\n{prep.shape[1]}x{prep.shape[0]}', fontsize=10)
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved: preprocessing_comparison.png')
plt.close()

# Also show histogram comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Pixel Intensity Distribution: Original vs Preprocessed', fontsize=14, fontweight='bold')

for i, name in enumerate(sample_names[:2]):  # Just 2 samples for histograms
    orig_path = orig_dir / name
    prep_path = prep_dir / name
    
    if not orig_path.exists() or not prep_path.exists():
        continue
    
    orig = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
    prep = cv2.imread(str(prep_path), cv2.IMREAD_GRAYSCALE)
    
    # Histogram for original
    axes[i, 0].hist(orig.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[i, 0].set_title(f'Original: {name}\nMean={orig.mean():.1f}, Std={orig.std():.1f}', fontsize=10)
    axes[i, 0].set_xlabel('Pixel Intensity')
    axes[i, 0].set_ylabel('Frequency')
    
    # Histogram for preprocessed
    axes[i, 1].hist(prep.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes[i, 1].set_title(f'Preprocessed: {name}\nMean={prep.mean():.1f}, Std={prep.std():.1f}', fontsize=10)
    axes[i, 1].set_xlabel('Pixel Intensity')
    axes[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histogram_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved: histogram_comparison.png')
plt.close()

print('\n✓ Visual comparison complete!')
