#!/usr/bin/env python3
"""Generate PDF summary of 11 reference papers for DigiSteel-YOLO."""

import subprocess
import sys

# Install fpdf2 if not available
try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF


class PaperPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "DigiSteel-YOLO: Reference Papers Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 80, 130)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Data
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


def main():
    pdf = PaperPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === TITLE PAGE ===
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "DigiSteel-YOLO", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Reference Papers Comprehensive Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Literature Review: 11 Reference Papers", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Steel Surface Defect Detection using YOLO Architectures", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, "Project: Comprehensive Robustness Study of Lightweight YOLO Detectors", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Team: Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat, Mahmoud Hisham", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Supervisor: Dr. Tarek Ghoneimy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Compiled: June 24, 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # === MASTER COMPARISON TABLE ===
    pdf.add_page()
    pdf.section_title("1. Executive Summary - Master Comparison Table")

    headers = ["ID", "Paper", "Base", "mAP@0.5", "Params(M)", "FPS", "Year", "Journal"]
    col_w = [12, 35, 22, 20, 20, 18, 14, 49]
    data = [
        ["P01", "PSF-YOLO", "YOLOv11n", "82.2%*", "1.82", "N/A", "2025", "Nat. Sci. Reports"],
        ["P02", "LAM-YOLOv10n", "YOLOv10n", "94.39%", "N/A", "154", "2025", "Nat. Sci. Reports"],
        ["P03", "YOLO-LSDI", "YOLOv11n", "83.0%", "2.7", "162.1", "2025", "MDPI Electronics"],
        ["P04", "Lightweight-YOLOv8", "YOLOv8n", "78.6%", "2.04", "171.5", "2025", "Nat. Sci. Reports"],
        ["P05", "SCCI-YOLO", "YOLOv8n", "78.6%", "1.68", "270.2", "2025", "Nat. Sci. Reports"],
        ["P06", "ELS-YOLO", "YOLOv11n", "79.5%", "2.36", "N/A", "2025", "MDPI Electronics"],
        ["P07", "ASFRW-YOLO", "YOLOv5s", "83.2%", "6.20", "125", "2025", "Nat. Sci. Reports"],
        ["P08", "MSFE-YOLO", "YOLOv11s", "79.8%", "11.69", "89.3", "2026", "MDPI Sensors"],
        ["P09", "EFEN-YOLOv8", "YOLOv8n", "80.4%", "N/A", "N/A", "2026", "PLOS ONE"],
        ["P10", "KDM-YOLO", "YOLOv10n", "95.4%", "3.29", "155.6", "2026", "MDPI Sensors"],
        ["P11", "YOLOv11-EMD", "YOLOv11", "94.9%", "N/A", "N/A", "2025", "MDPI Math."],
    ]
    pdf.add_table(headers, data, col_w)
    pdf.body_text("* P01 evaluates on GC10-DET+ (not NEU-DET). P10 trains at native 200px (not 640px upscale).")

    # === NOVEL MODULES TABLE ===
    pdf.sub_title("Novel Modules Summary")
    headers2 = ["Paper", "Novel Modules"]
    col_w2 = [35, 155]
    data2 = [
        ["P01 PSF-YOLO", "MDF-Neck, Virtual Fusion Head, Attention Concat, GhostConv"],
        ["P02 LAM-YOLOv10n", "GhostConv, SMA (Spatial Multi-Scale Attention), MFFN"],
        ["P03 YOLO-LSDI", "AMSPPF, DSAM, LDConv, Inner-CIoU"],
        ["P04 Lightweight-YOLOv8", "GhostNet backbone, MPCA, SIoU loss"],
        ["P05 SCCI-YOLO", "SPD-Conv, C2f_EMA, CCFM, Inner-IoU"],
        ["P06 ELS-YOLO", "C3k2_THK, Staged-Slim-Neck, MSDetect"],
        ["P07 ASFRW-YOLO", "ASF (SSFF+TFE+CPAM), RepNCSPELAN4, WIoU v3"],
        ["P08 MSFE-YOLO", "MSFC, C2MSDA, AFFE"],
        ["P09 EFEN-YOLOv8", "SAConv, LSKA, WASPP, gamma-FEIoU"],
        ["P10 KDM-YOLO", "KWConv, C2f-DRB, MSAF"],
        ["P11 YOLOv11-EMD", "InnerEIoU, MSDA, C3k2_DynamicConv, Transfer Learning"],
    ]
    pdf.add_table(headers2, data2, col_w2)

    # === INDIVIDUAL PAPERS ===
    papers = [
        {
            "id": "P01", "name": "PSF-YOLO",
            "title": "A lightweight YOLOv11-based framework for small steel defect detection with a newly enhanced feature fusion module",
            "authors": "Yongyao Wang, Haiyang Sun, Kai Luo, Quanfu Zhu, Haofei Li, Yuyang Sun, Zhenjie Wu, Gang Wang",
            "journal": "Scientific Reports (Nature Portfolio), 2025, Vol. 15, Article 34322",
            "doi": "10.1038/s41598-025-16619-9",
            "base": "YOLOv11n", "params": "1.82M", "map50": "82.2% (GC10-DET+)", "map5095": "45.8% (GC10-DET+)", "fps": "N/A",
            "modules": "MDF-Neck (Multi-Dimensional-Fusion Neck integrating P1 layer), Virtual Fusion Head (resolution-aware pooling + alignment), Attention Concat Module (spatial-channel attention outperforming SE/CBAM), GhostConv (parameter reduction)",
            "training": "Optimizer: SGD; Dataset: GC10-DET+ (enhanced); Split: 7:1:2; Augmentation: flipping, rotation, mosaic",
            "strengths": "1) 25% parameter reduction with +3.2% mAP gain. 2) Strong cross-domain generalization (3 datasets). 3) Principled design with individual ablation.",
            "weaknesses": "1) No FPS/latency benchmarks. 2) No evaluation on NEU-DET. 3) GC10-DET+ dataset not publicly available.",
        },
        {
            "id": "P02", "name": "LAM-YOLOv10n",
            "title": "Steel surface defect detection algorithm based on improved YOLOv10",
            "authors": "Laomo Zhang, Zhike Wang, Ying Ma, Guowei Li",
            "journal": "Scientific Reports (Nature Publishing Group), 2025, Vol. 15, Article 32827",
            "doi": "10.1038/s41598-025-16725-8",
            "base": "YOLOv10n", "params": "N/A", "map50": "94.39%", "map5095": "N/A", "fps": "154",
            "modules": "GhostConv (lightweight ghost module), SMA (Spatial Multi-Scale Attention - 3 parallel branches with multi-scale pooling and cross-spatial attention), MFFN (Multi-Branch Feature Fusion Network - GAP+conv and local pooling branches)",
            "training": "Hardware: GTX 1060Ti; Epochs: 100; Batch: 16; LR: 0.001; Optimizer: Adam; Dataset: NEU-DET (augmented as PRO-DataSet)",
            "strengths": "1) 96.96% precision as nano model (outperforms YOLOv10b). 2) Well-designed modular architecture. 3) Competitive speed (154 FPS).",
            "weaknesses": "1) Missing param count/FLOPs/mAP@0.5:0.95. 2) Only NEU-DET (no cross-dataset). 3) Outdated hardware (GTX 1060Ti).",
        },
        {
            "id": "P03", "name": "YOLO-LSDI",
            "title": "YOLO-LSDI: An Enhanced Algorithm for Steel Surface Defect Detection Using a YOLOv11 Network",
            "authors": "Fuqiang Wang, Xinbin Jiang, Yizhou Han, Lei Wu",
            "journal": "Electronics (MDPI), 2025, Vol. 14(13), Article 2576",
            "doi": "10.3390/electronics14132576",
            "base": "YOLOv11n", "params": "2.7M", "map50": "83.0%", "map5095": "~52-57%", "fps": "162.1",
            "modules": "AMSPPF (Adaptive Multi-Scale Pooling-Fast), DSAM (Deformable Spatial Attention Module), LDConv (Linear Deformable Convolution), Inner-CIoU loss",
            "training": "Cross-dataset validation on NEU-DET, GC10-DET, APSPC PCB datasets",
            "strengths": "1) +5.8% mAP with reduced GFLOPs. 2) Real-time at 162.1 FPS. 3) Cross-dataset validation on 3 datasets.",
            "weaknesses": "1) MDPI Electronics (not top-tier). 2) Incremental contributions. 3) Limited ablation clarity.",
        },
        {
            "id": "P04", "name": "Lightweight-YOLOv8",
            "title": "A lightweight algorithm for steel surface defect detection using improved YOLOv8",
            "authors": "Shuangbao Ma, Xin Zhao, Li Wan, Yapeng Zhang, Hongliang Gao",
            "journal": "Scientific Reports (Nature Publishing Group), 2025, Vol. 15, Article 8966",
            "doi": "10.1038/s41598-025-93469-5",
            "base": "YOLOv8n", "params": "2.04M", "map50": "78.6%", "map5095": "44.5%", "fps": "171.5",
            "modules": "GhostNet backbone (replaces CSPDarknet), MPCA (Multi-Path Coordinate Attention - max/avg pooling along W and H), SIoU loss (angle+distance+shape+IoU costs)",
            "training": "GPU: RTX 4090; Epochs: 300; Batch: 16; Optimizer: SGD (lr=0.01, cosine); Image: 640x640; Split: 90/10",
            "strengths": "1) Extreme lightweight (2.04M, 5.1 GFLOPs). 2) ALL metrics improved simultaneously. 3) Edge deployment validated (Raspberry Pi: 115.7 FPS, Jetson: 440.3 FPS). 4) Highly cited (37 citations).",
            "weaknesses": "1) Only NEU-DET. 2) Poor on crazing (21.4% AP). 3) No comparison with 2024 SOTA.",
        },
        {
            "id": "P05", "name": "SCCI-YOLO",
            "title": "An efficient and lightweight algorithm for detecting surface defects of steel based on SCCI-YOLO",
            "authors": "Huixiang Zhou, Hong Zou, Gaojun Hu",
            "journal": "Scientific Reports (Nature Portfolio), 2025, Vol. 15, Article 36276",
            "doi": "10.1038/s41598-025-20154-y",
            "base": "YOLOv8n", "params": "1.68M", "map50": "78.6%", "map5095": "N/A", "fps": "270.2",
            "modules": "SPD-Conv (Space-to-Depth for small defect detection), C2f_EMA (C2f + EMA attention), CCFM (Cross-scale Convolutional Feature Module), Inner-IoU loss",
            "training": "Standard YOLOv8 training protocol on NEU-DET",
            "strengths": "1) Ultra-lightweight (1.68M params, 43.9% reduction). 2) Fastest inference (270.2 FPS). 3) Effective small defect detection via SPD-Conv.",
            "weaknesses": "1) Modest absolute mAP (78.6%). 2) Single dataset (NEU-DET only). 3) Lightweight-accuracy trade-off.",
        },
        {
            "id": "P06", "name": "ELS-YOLO",
            "title": "ELS-YOLO: Efficient Lightweight YOLO for Steel Surface Defect Detection",
            "authors": "Zhiheng Zhang, Guoyun Zhong, Peng Ding, Jianfeng He, Jun Zhang, Chongyang Zhu",
            "journal": "Electronics (MDPI), 2025, Vol. 14(19), Article 3877",
            "doi": "10.3390/electronics14193877",
            "base": "YOLOv11n", "params": "2.36M", "map50": "79.5%", "map5095": "43.2%", "fps": "N/A",
            "modules": "C3k2_THK (T-shaped conv + Heterogeneous Kernel Selection + SCSA attention), Staged-Slim-Neck (DGSConv-L/H + GMLCA attention), MSDetect (MRFB multi-scale detection head)",
            "training": "GPU: RTX 4070Ti Super; Epochs: 400; Batch: 16; Optimizer: AdamW (lr=0.001); Image: 640x640",
            "strengths": "1) Exceptional param efficiency (2.36M, 5.6G FLOPs). 2) Strong multi-scale detection. 3) Cross-dataset generalization (NEU-DET, GC10-DET, Severstal).",
            "weaknesses": "1) mAP@50:0.95 gap (43.2% vs 44.1% baseline). 2) No FPS reported. 3) Insufficient small-scale defect detection.",
        },
        {
            "id": "P07", "name": "ASFRW-YOLO",
            "title": "A high precision and lightweight method for steel surface defect detection based on improved YOLOv5",
            "authors": "Mudan Zhou, Haoyu Wang, Yuhao Wang",
            "journal": "Scientific Reports (Nature Publishing Group), 2025",
            "doi": "10.1038/s41598-025-28022-5",
            "base": "YOLOv5s", "params": "6.20M", "map50": "83.2%", "map5095": "46.4%", "fps": "~125",
            "modules": "ASF (Attentional Scale Sequence Fusion: SSFF 3D conv + TFE + CPAM), RepNCSPELAN4 (from YOLOv9, replaces C3), WIoU v3 loss (dynamic non-monotonic focusing)",
            "training": "GPU: RTX 4060 Laptop; Epochs: 300; Batch: 16; Optimizer: SGD; Image: 640x640",
            "strengths": "1) Strong mAP improvement (+7.0%). 2) Best mAP@0.5:0.95 (46.4%). 3) Parameter efficiency (fewer than YOLOv5s baseline). 4) Excellent on small defects (crazing +14.4%).",
            "weaknesses": "1) Struggles with crack-type defects. 2) Lab-only evaluation. 3) Limited dataset scope (NEU-DET only).",
        },
        {
            "id": "P08", "name": "MSFE-YOLO",
            "title": "MSFE-YOLO: A Steel Surface Defect Detection Algorithm Integrating Multi-Scale Frequency Domain and Defect-Aware Attention",
            "authors": "Siqi Su, Jiale Shen, P. Lin, Wanhe Tang, Weijie Zhang, Zhen Chen",
            "journal": "Sensors (MDPI), 2026, Vol. 26(8), Article 2311",
            "doi": "10.3390/s26082311",
            "base": "YOLOv11s", "params": "11.69M", "map50": "79.8%", "map5095": "N/A", "fps": "89.3",
            "modules": "MSFC (Multi-Scale Frequency-Enhanced Conv with Laplacian + depth-adaptive dilation), C2MSDA (Cross-Stage Defect-Aware Attention: Sobel edge + spatial + channel), AFFE (Adaptive Feature Fusion Enhancement)",
            "training": "GPU: RTX 3090; Epochs: 300; Batch: 16; Optimizer: SGD; Image: 640x640; GC10-DET: 34.4% mAP@0.5:0.95",
            "strengths": "1) Effective frequency-domain fusion (Laplacian + Sobel). 2) Comprehensive defect-aware attention. 3) Real-time capable (89.3 FPS).",
            "weaknesses": "1) Modest mAP gain (+1.7%) at high cost (+30% params, -35% FPS). 2) Low mAP@0.5:0.95 on GC10-DET. 3) Recall regression (72.6% vs 75.5%).",
        },
        {
            "id": "P09", "name": "EFEN-YOLOv8",
            "title": "EFEN-YOLOv8: Surface defect detection network based on spatial feature capture and multi-level weighted attention",
            "authors": "Meishun Wu, Jinmin Peng, Xinyi Yu, Heng Xu, Haotian Sun",
            "journal": "PLOS ONE, 2026",
            "doi": "10.1371/journal.pone.0339617",
            "base": "YOLOv8n", "params": "N/A", "map50": "80.4%", "map5095": "N/A", "fps": "N/A",
            "modules": "SAConv (Shallow Attention Conv - multi-scale heterogeneous kernels), LSKA (Large Separable Kernel Attention - cascaded 1D), WASPP (Weighted Atrous Spatial Pyramid Pooling), gamma-FEIoU loss (EIoU + Focal + adaptive class weighting)",
            "training": "GPU: RTX 3060; Epochs: 350; Batch: 16; Optimizer: SGD; 5 random seeds for statistical validation; GC10-DET: 72.1%",
            "strengths": "1) +7.4% mAP with rigorous statistical validation (CIs, t-tests). 2) Modular ablation (WASPP +3.5%, gamma-FEIoU +3.0%, LSKA +2.2%). 3) Cross-dataset (NEU-DET + GC10-DET). 4) Open access with code.",
            "weaknesses": "1) Increased computational cost (no FPS). 2) Poor on cracking (44.4%). 3) SAConv marginal alone (+0.2%).",
        },
        {
            "id": "P10", "name": "KDM-YOLO",
            "title": "Lightweight Visual Localization of Steel Surface Defects for Autonomous Inspection Robots Based on Improved YOLOv10n",
            "authors": "Jinwu Tong, Xin Zhang, Xinyun Lu, Han Cao, Lengtao Yao, Bingbing Gao",
            "journal": "Sensors (MDPI), 2026, Vol. 26(7), Article 2132",
            "doi": "10.3390/s26072132",
            "base": "YOLOv10n", "params": "3.29M", "map50": "95.4%", "map5095": "N/A", "fps": "155.6",
            "modules": "KWConv (KernelWarehouse dynamic kernel sharing), C2f-DRB (Dilated Residual Block with re-parameterization), MSAF (Multi-Scale Attention Fusion: Region + Pixel attention with gated fusion)",
            "training": "GPU: RTX 4060; Epochs: 200; Batch: 16; Optimizer: SGD; Image: 200x200 (native); Cross-domain: bearing defect +3.2%",
            "strengths": "1) Highest mAP (95.4%) with lightweight design (3.29M, 155.6 FPS). 2) Excellent ablation (cumulative gains). 3) Cross-domain generalization. 4) Crazing: +24.8% AP improvement.",
            "weaknesses": "1) No mAP@0.5:0.95 reported. 2) Trained at native 200px (protocol difference). 3) Small dataset (1,800 images).",
        },
        {
            "id": "P11", "name": "YOLOv11-EMD",
            "title": "YOLOv11-EMD: An Enhanced Object Detection Algorithm Assisted by Multi-Stage Transfer Learning for Industrial Steel Surface Defect Detection",
            "authors": "Weipeng Shi, Junlin Dai, Changhe Li, Na Niu",
            "journal": "Mathematics (MDPI), 2025, Vol. 13(17), Article 2769",
            "doi": "10.3390/math13172769",
            "base": "YOLOv11", "params": "N/A", "map50": "94.9%", "map5095": "N/A", "fps": "N/A",
            "modules": "InnerEIoU loss (improved EIoU), MSDA (Multi-Scale Dilated Attention), C3k2_DynamicConv (attention-weighted kernel combinations), Multi-stage Transfer Learning (source pre-train + target fine-tune)",
            "training": "Multi-stage transfer learning; Datasets: NEU-DET + Severstal (combined), NEU-DET + GC10-DET (cross-scenario); Transfer learning: -3.2% time, +8.8% mAP",
            "strengths": "1) Practical transfer learning framework (+8.8% mAP). 2) Multi-dataset evaluation. 3) Targeted improvements for specific challenges.",
            "weaknesses": "1) Limited metrics (no FPS, params, mAP@0.5:0.95). 2) Short paper (6 pages). 3) Cross-scenario gap (94.9% to 79.9%).",
        },
    ]

    for p in papers:
        pdf.add_page()
        pdf.section_title(f"{p['id']} - {p['name']}")
        pdf.sub_title("Full Title")
        pdf.body_text(p["title"])
        pdf.sub_title("Authors")
        pdf.body_text(p["authors"])
        pdf.sub_title("Publication")
        pdf.body_text(f"{p['journal']}")
        pdf.body_text(f"DOI: {p['doi']}")
        pdf.sub_title("Base Model & Metrics")

        headers_m = ["Base Model", "Params", "mAP@0.5", "mAP@0.5:0.95", "FPS"]
        data_m = [[p["base"], p["params"], p["map50"], p["map5095"], p["fps"]]]
        pdf.add_table(headers_m, data_m, [38, 38, 38, 38, 38])

        pdf.sub_title("Novel Modules")
        pdf.body_text(p["modules"])
        pdf.sub_title("Training Details")
        pdf.body_text(p["training"])
        pdf.sub_title("Strengths")
        pdf.body_text(p["strengths"])
        pdf.sub_title("Weaknesses")
        pdf.body_text(p["weaknesses"])

    # === COMPARATIVE ANALYSIS ===
    pdf.add_page()
    pdf.section_title("13. Comparative Analysis")

    pdf.sub_title("Accuracy vs. Efficiency Trade-off")
    headers3 = ["Paper", "mAP@0.5", "Params(M)", "FPS", "Rating"]
    col_w3 = [40, 30, 30, 30, 60]
    data3 = [
        ["P10 KDM-YOLO", "95.4%", "3.29", "155.6", "Best accuracy-efficiency"],
        ["P02 LAM-YOLOv10n", "94.39%", "N/A", "154", "High accuracy"],
        ["P11 YOLOv11-EMD", "94.9%", "N/A", "N/A", "Transfer learning"],
        ["P07 ASFRW-YOLO", "83.2%", "6.20", "125", "Best localization"],
        ["P03 YOLO-LSDI", "83.0%", "2.7", "162.1", "Balanced"],
        ["P09 EFEN-YOLOv8", "80.4%", "N/A", "N/A", "Statistical rigor"],
        ["P08 MSFE-YOLO", "79.8%", "11.69", "89.3", "Frequency domain"],
        ["P06 ELS-YOLO", "79.5%", "2.36", "N/A", "Most efficient"],
        ["P04 Lightweight-YOLOv8", "78.6%", "2.04", "171.5", "Edge deployment"],
        ["P05 SCCI-YOLO", "78.6%", "1.68", "270.2", "Fastest inference"],
    ]
    pdf.add_table(headers3, data3, col_w3)

    pdf.sub_title("Key Observations for DigiSteel-YOLO")
    observations = [
        "1. NO paper reports robustness evaluation - DigiSteel's perturbation framework (6 types x 4 levels) is unique.",
        "2. Training protocol inconsistency - P10 trains at 200px native; others at 640px. Direct comparison is unfair.",
        "3. GhostConv is popular (P01, P02, P04) - validates our use in DigiSteel v2.",
        "4. Inner-IoU variants trending (P03, P05, P11) - validates our Inner-WIoU choice.",
        "5. EMA attention used by P05 - same as DigiSteel v2. Common technique, not novel.",
        "6. Best achievable on NEU-DET: ~95% (P10 at 200px). At 640px: ~83% is competitive range.",
        "7. Most papers use NEU-DET (10/11) and SGD optimizer with 200-300 epochs.",
        "8. Lightweight models (P04, P05) achieve 78.6% with <2.1M params - efficiency frontier.",
    ]
    for obs in observations:
        pdf.body_text(obs)

    # Save
    output_path = "docs/Reference_Papers_Summary.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    main()
