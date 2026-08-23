from pathlib import Path

base = Path('datasets/NEU-DET/yolo_preprocessed')
print('Preprocessed Dataset Structure:')
print('=' * 60)

for split in ['train', 'val', 'test']:
    img_dir = base / 'images' / split
    lbl_dir = base / 'labels' / split
    
    if img_dir.exists():
        imgs = list(img_dir.glob('*.jpg'))
        print(f'\n{split.upper()}:')
        print(f'  Images: {len(imgs)}')
        
        if lbl_dir.exists():
            lbls = list(lbl_dir.glob('*.txt'))
            print(f'  Labels: {len(lbls)}')
            
            # Check if counts match
            if len(imgs) == len(lbls):
                print(f'  ✓ Match: Yes')
            else:
                print(f'  ✗ Match: No (mismatch!)')
        
        # Sample image info
        if imgs:
            import cv2
            sample = cv2.imread(str(imgs[0]))
            print(f'  Sample: {imgs[0].name} ({sample.shape[1]}x{sample.shape[0]})')

print('\n' + '=' * 60)
print('✓ Preprocessing complete!')
