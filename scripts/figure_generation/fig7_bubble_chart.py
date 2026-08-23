import os
import matplotlib.pyplot as plt

def create_bubble_chart():
    # Data: Baseline (YOLOv11n) vs DAFEGate v4 (Ours)
    models = ['Baseline (YOLOv11n)', 'DigiSteel-YOLO (DAFEGate v4)']
    params = [2.59, 2.69] 
    maps = [79.35, 81.98] # Verified test-set mAP@0.5
    fps = [162, 145]      # Verified inference FPS
    colors = ['#1f77b4', '#2ca02c']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Bubble size proportional to FPS
    sizes = [f * 15 for f in fps]
    
    scatter = ax.scatter(params, maps, s=sizes, c=colors, alpha=0.75, edgecolors="black", linewidth=1.5)
    
    # Add labels with background boxes for legibility
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95, lw=1)
    bbox_props_fps = dict(boxstyle="round,pad=0.25", fc="#333333", ec="none", alpha=0.85)
    
    # Baseline annotation
    ax.annotate(models[0], (params[0], maps[0]), xytext=(-35, 30), 
                textcoords='offset points', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', bbox=bbox_props)
    ax.annotate(f"{fps[0]} FPS", (params[0], maps[0]), xytext=(0, -5), 
                textcoords='offset points', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold', bbox=bbox_props_fps)
                
    # DAFEGate v4 annotation
    ax.annotate(models[1], (params[1], maps[1]), xytext=(35, 30), 
                textcoords='offset points', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', bbox=bbox_props, color='#1b5e20')
    ax.annotate(f"{fps[1]} FPS", (params[1], maps[1]), xytext=(0, -5), 
                textcoords='offset points', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold', bbox=bbox_props_fps)

    # Gain annotation arrow
    ax.annotate('+2.63pp mAP@0.5\n(+3.7% Param Overhead)', 
                xy=(params[1], maps[1]), xytext=(2.64, 80.5),
                arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkgreen', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="darkgreen", lw=1))

    ax.set_title('Efficiency & Performance: Baseline vs. DigiSteel-YOLO', fontsize=14, pad=18, fontweight='bold')
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP@0.5 (%) on NEU-DET', fontsize=12, fontweight='bold')
    
    ax.set_xlim(2.50, 2.78)
    ax.set_ylim(78.5, 83.0)

    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Figure7_Bubble_Chart.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    create_bubble_chart()
