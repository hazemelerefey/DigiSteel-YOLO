import os
import matplotlib.pyplot as plt
from PIL import Image
import glob

def create_figure1():
    base_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\images\train"
    
    # Define classes for morphological duality
    classes = {
        'crazing': 'Linear Defect: Crazing',
        'scratches': 'Linear Defect: Scratches',
        'pitted_surface': 'Surface Anomaly: Pitted Surface',
        'rolled-in_scale': 'Surface Anomaly: Rolled-in Scale'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, (cls, title) in enumerate(classes.items()):
        # Find an image for this class
        search_pattern = os.path.join(base_dir, f"{cls}_*.jpg")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Warning: No images found for {cls} at {search_pattern}")
            continue
            
        img_path = files[0] # Just take the first one
        img = Image.open(img_path)
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"({chr(97+idx)}) {title}", fontsize=14, pad=10)
        axes[idx].axis('off')
        
    plt.tight_layout()
    
    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Figure1_Morphological_Duality.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    create_figure1()
