import os
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.append(r"D:\DigiSteel-Yolo\DigiSteel-YOLO")
from ultralytics import YOLO

def get_activation(name, activations_dict):
    def hook(model, input, output):
        activations_dict[name] = output.detach().cpu()
    return hook

def create_feature_map_figure():
    weight_path = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\runs\detect\exp_5a_digisteel_v4_dafe\weights\best.pt"
    img_path = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\datasets\NEU-DET\yolo\images\test\crazing_13.jpg"
    
    if not os.path.exists(weight_path) or not os.path.exists(img_path):
        print("Missing weights or image.")
        return

    model = YOLO(weight_path)
    
    # We will try to find the DAFEGate module
    # If not found directly, we will hook into a known backbone layer as fallback
    target_module = None
    for name, module in model.model.named_modules():
        if 'DAFE' in str(type(module)):
            target_module = module
            break
            
    if target_module is None:
        print("Warning: DAFE module not found explicitly. Hooking into layer 4 (P3).")
        target_module = model.model.model[4]
        
    activations = {}
    hook_handle = target_module.register_forward_hook(get_activation('dafe_out', activations))
    
    # Run inference
    model(img_path, verbose=False)
    hook_handle.remove()
    
    if 'dafe_out' not in activations:
        print("Failed to capture activations.")
        return
        
    feat_map = activations['dafe_out'][0] # Shape: (C, H, W)
    
    # To visualize, we take the mean across all channels and normalize
    # Alternatively, we could show the top 2 activating channels
    heat_mean = torch.mean(feat_map, dim=0).numpy()
    heat_max = torch.max(feat_map, dim=0)[0].numpy()
    
    # Normalize to 0-255
    heat_mean = cv2.normalize(heat_mean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heat_max = cv2.normalize(heat_max, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Original image
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmaps to match image size
    heat_mean_resized = cv2.resize(heat_mean, (orig_img.shape[1], orig_img.shape[0]))
    heat_max_resized = cv2.resize(heat_max, (orig_img.shape[1], orig_img.shape[0]))
    
    # Apply colormap
    colormap_mean = cv2.applyColorMap(heat_mean_resized, cv2.COLORMAP_JET)
    colormap_max = cv2.applyColorMap(heat_max_resized, cv2.COLORMAP_JET)
    
    colormap_mean = cv2.cvtColor(colormap_mean, cv2.COLOR_BGR2RGB)
    colormap_max = cv2.cvtColor(colormap_max, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    overlay_mean = cv2.addWeighted(orig_img, 0.5, colormap_mean, 0.5, 0)
    overlay_max = cv2.addWeighted(orig_img, 0.5, colormap_max, 0.5, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original (Crazing)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(overlay_mean)
    axes[1].set_title("DAFE Mean Activation", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_max)
    axes[2].set_title("DAFE Max Activation (Edge Focus)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Figure3_Feature_Maps.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    create_feature_map_figure()
