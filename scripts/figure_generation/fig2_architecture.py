import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Color palette
    c_input = '#E8F1FF'    # Light blue
    c_branch = '#F2FBF6'   # Light green
    c_attn = '#FFF7E6'     # Light orange
    c_fusion = '#F3E5F5'   # Light purple
    c_output = '#F7ECFF'   # Light violet
    border_color = '#2F3542'

    # Box styling helper
    def draw_box(x, y, w, h, text, subtext="", color='#FFFFFF', edge_color=border_color, lw=1.5):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor=color, edgecolor=edge_color, linewidth=lw)
        ax.add_patch(box)
        if subtext:
            ax.text(x + w/2, y + h*0.62, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#1E272C')
            ax.text(x + w/2, y + h*0.28, subtext, ha='center', va='center', fontsize=9, color='#57606F', style='italic')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#1E272C')
        return (x + w/2, y, x + w/2, y + h)

    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, label="", dashed=False, color='#2F3542'):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, ls=ls, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label, fontsize=9, fontweight='bold', color='#D84315')

    # Title
    ax.text(7, 10.5, 'DAFEGate v4: Defect-Aware Feature Enhancement Module', ha='center', va='center', fontsize=14, fontweight='bold', color='#2F6FED')

    # 1. Input Node
    draw_box(4.5, 9.2, 5.0, 0.8, "Input Feature Map x", "Shape: (B × C × H × W) [at P3: C=256, 80×80]", c_input)

    # 2. Edge Branch
    draw_arrow(7.0, 9.2, 7.0, 8.2)
    draw_box(3.5, 7.4, 7.0, 0.8, "EdgeAwareConv (Sobel-X/Y Initialized)", "3×3 Conv + BN + SiLU | Output: C/2 channels (128)", c_branch)

    # Split from Edge to Texture and Concat
    draw_arrow(7.0, 7.4, 7.0, 6.4)
    draw_arrow(7.0, 7.4, 3.5, 5.0) # Direct to concat (left)

    # 3. Texture Branch (analyzes edge map variance)
    draw_box(5.5, 5.6, 6.5, 0.8, "TextureBranch (Local Variance Extraction)", "Var(E) = AvgPool(E²) - (AvgPool(E))² -> 1×1 Conv -> SiLU | C/2 channels", c_branch)

    # Connect Texture to Concat
    draw_arrow(8.75, 5.6, 8.75, 4.8)

    # 4. Concat
    draw_box(3.5, 4.0, 7.0, 0.8, "Channel Concatenation [Edge + Texture]", "Reconstructed Dimension: (B × C × H × W)", c_branch)

    # 5. Channel Attention (SE)
    draw_arrow(7.0, 4.0, 7.0, 3.1)
    draw_box(3.5, 2.3, 7.0, 0.8, "Channel Attention & Feature Reweighting", "GAP -> FC(C/8) -> ReLU -> FC(C) -> Sigmoid -> Multiply (fused ⊙ att)", c_attn)

    # 6. Fusion & Residual Strength
    draw_arrow(7.0, 2.3, 7.0, 1.4)
    draw_box(4.0, 0.8, 6.0, 0.6, "Fusion & Learnable Scale", "1×1 Conv + BN + SiLU -> Multiply by α = sigmoid(α_raw)", c_fusion)

    # 7. Additive Residual & Output
    # Skip highway on the left
    ax.plot([4.5, 1.2, 1.2, 5.5], [9.6, 9.6, -0.1, -0.1], color='#2F6FED', lw=2.0, ls='--')
    ax.annotate('', xy=(5.5, -0.1), xytext=(5.3, -0.1),
                arrowprops=dict(arrowstyle="-|>", color='#2F6FED', lw=2.0, mutation_scale=15))
    ax.text(1.4, 5.0, "Identity Skip Connection (Gradient Highway: ∂y/∂x = 1.0)", 
            rotation=90, va='center', ha='center', fontsize=9.5, fontweight='bold', color='#2F6FED')

    draw_arrow(7.0, 0.8, 7.0, 0.1)
    draw_box(4.0, -0.5, 6.0, 0.6, "Additive Residual Output: y = x + α · enhanced", "Preserves baseline features while injecting defect-specific enhancements", c_output)

    output_dir = r"D:\DigiSteel-Yolo\DigiSteel-YOLO\figures"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Figure2_DAFEGate_Architecture.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    draw_architecture_diagram()
