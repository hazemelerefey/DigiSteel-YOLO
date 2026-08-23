import os
import matplotlib.pyplot as plt
import numpy as np

def create_bar_chart():
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # Data from DAFE-ABLATION-REPORT.md
    baseline_ap = [43.6, 85.2, 91.0, 79.3, 77.9, 99.0]
    v4_ap = [49.1, 88.3, 91.7, 85.0, 78.8, 98.9]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width/2, baseline_ap, width, label='Baseline (YOLOv11n)', color='#8c564b', alpha=0.8)
    rects2 = ax.bar(x + width/2, v4_ap, width, label='DAFEGate v4', color='#2ca02c', alpha=0.9)
    
    ax.set_ylabel('mAP@0.5 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance: Baseline vs DAFEGate v4', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    
    ax.set_ylim(40, 105)
    
    # Attach a text label above each bar displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
            
    autolabel(rects1)
    autolabel(rects2)
    
    # Highlight crazing jump
    ax.annotate('+5.5% Jump\n(Sobel Edge Power)', xy=(0 + width/2, 49.1), xytext=(0.5, 60),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=11, fontweight='bold', color='darkred', ha='center')

    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Figure6_PerClass_AP.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    create_bar_chart()
