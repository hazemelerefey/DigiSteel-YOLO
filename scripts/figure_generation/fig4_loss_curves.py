import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_loss_curves(baseline_csv, v4_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to read and plot a specific line
    def plot_line(csv_path, label, color, linestyle='-'):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [col.strip() for col in df.columns]
            if 'train/box_loss' in df.columns:
                ax.plot(df['epoch'], df['train/box_loss'], label=label, color=color, linestyle=linestyle, linewidth=2.2, alpha=0.9)
            else:
                print(f"Warning: 'train/box_loss' not found in {csv_path}")
        else:
            print(f"Warning: CSV not found at {csv_path}")

    # Plot Baseline and DAFEGate v4 from full continuous CSV logs
    plot_line(baseline_csv, 'Baseline (YOLOv11n)', '#1f77b4')                   # Blue
    plot_line(v4_csv, 'DigiSteel-YOLO (DAFEGate v4)', '#2ca02c')              # Green
    
    ax.set_title('Training Dynamics: Bounding Box Loss (Baseline vs. DigiSteel-YOLO)', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Box Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    
    ax.set_xlim(0, 400)
    ax.set_ylim(0.8, 2.6)
    
    out_path = os.path.join(output_dir, "FigureS1_Training_Loss_Dynamics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Training Dynamics Loss Curves.')
    parser.add_argument('--baseline', type=str, default=r"D:\DigiSteel-Yolo\DigiSteel-YOLO\runs\detect\week4_4a_fresh_baseline\results.csv", help='Path to baseline results.csv')
    parser.add_argument('--v4', type=str, default=r"D:\DigiSteel-Yolo\DigiSteel-YOLO\runs\detect\exp_5a_digisteel_v4_dafe\results.csv", help='Path to v4 results.csv')
    
    args = parser.parse_args()
    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    
    plot_loss_curves(args.baseline, args.v4, output_dir)
