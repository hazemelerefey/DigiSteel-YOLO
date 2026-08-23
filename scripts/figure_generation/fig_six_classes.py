import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_six_classes_figure():
    base_img_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\images\train"
    base_lbl_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\labels\train"
    
    # 6 classes with clean display names and descriptions
    classes_info = [
        ("crazing", "Crazing (Cr)", "Linear / Edge"),
        ("inclusion", "Inclusion (In)", "Surface / Texture"),
        ("patches", "Patches (Pa)", "Surface / Texture"),
        ("pitted_surface", "Pitted Surface (PS)", "Surface / Texture"),
        ("rolled-in_scale", "Rolled-in Scale (RS)", "Surface / Texture"),
        ("scratches", "Scratches (Sc)", "Linear / Edge"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), dpi=300)
    axes = axes.flatten()
    
    # Elegant styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    for idx, (cls_name, display_title, defect_type) in enumerate(classes_info):
        img_files = sorted(glob.glob(os.path.join(base_img_dir, f"{cls_name}_*.jpg")))
        if not img_files:
            continue
            
        # Select a visually clear sample
        sample_img_path = img_files[0]
        img = Image.open(sample_img_path).convert("RGB")
        w, h = img.size
        
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        
        # Load ground truth bounding boxes if available
        base_name = os.path.splitext(os.path.basename(sample_img_path))[0]
        lbl_path = os.path.join(base_lbl_dir, f"{base_name}.txt")
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, bw, bh = map(float, parts[:5])
                        # Denormalize coordinates
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h
                        
                        rect = patches.Rectangle(
                            (x1, y1), box_w, box_h,
                            linewidth=2,
                            edgecolor='#00FF66',
                            facecolor='none',
                            linestyle='-'
                        )
                        ax.add_patch(rect)
        
        # Title with panel letter and defect category
        panel_letter = chr(97 + idx)
        ax.set_title(f"({panel_letter}) {display_title}\n[{defect_type}]", fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')
        
    plt.suptitle("The Six Steel Surface Defect Categories in the NEU-DET Benchmark", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.25)
    
    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "01b_Figure1b_NEU_DET_Six_Classes.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    generate_six_classes_figure()
