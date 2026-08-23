import os
import sys
import argparse

# Add project root to sys.path so PyTorch can find the custom 'digisteel' module
sys.path.append(r"D:\DigiSteel-Yolo\DigiSteel-YOLO")

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob

def run_inference_and_plot(baseline_weight, v4_weight, image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model_base = YOLO(baseline_weight)
        model_v4 = YOLO(v4_weight)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    for idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # Run inference
        res_base = model_base(img_path, verbose=False)[0]
        res_v4 = model_v4(img_path, verbose=False)[0]
        
        # Plotting
        img_base_plotted = res_base.plot(line_width=2, font_size=1)
        img_v4_plotted = res_v4.plot(line_width=2, font_size=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Convert BGR to RGB for matplotlib
        axes[0].imshow(cv2.cvtColor(img_base_plotted, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Baseline (YOLOv11n)", fontsize=14, pad=10)
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_v4_plotted, cv2.COLOR_BGR2RGB))
        axes[1].set_title("DAFEGate v4", fontsize=14, pad=10)
        axes[1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"Figure5_Comparison_{os.path.basename(img_path)}")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Qualitative Detection Examples.')
    parser.add_argument('--baseline', type=str, default=r"D:\DigiSteel-Yolo\DigiSteel-YOLO\runs\detect\week4_4a_fresh_baseline\weights\best.pt")
    parser.add_argument('--v4', type=str, default=r"D:\DigiSteel-Yolo\DigiSteel-YOLO\runs\detect\exp_5a_digisteel_v4_dafe\weights\best.pt")
    
    args = parser.parse_args()
    
    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    
    # Pick a few specific challenging images (crazing and pitted surface)
    base_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\images\test"
    test_images = []
    
    for cls in ['crazing', 'pitted_surface', 'scratches']:
        pattern = os.path.join(base_dir, f"{cls}_*.jpg")
        files = glob.glob(pattern)
        if files:
            test_images.append(files[0]) # take the first one
            
    if not test_images:
        print(f"No test images found in {base_dir}")
    else:
        run_inference_and_plot(args.baseline, args.v4, test_images, output_dir)
