"""
Defect-Aware Feature Enhancement Gate (DAFEGate) v4 — Final Model.

Novel contribution of DAFEGate-YOLO (DigiSteel Team) for hot-rolled flat steel
surface defect detection. Combines Sobel-initialized edge detection with local
variance texture analysis in a dual-branch architecture with channel attention
and additive residual highway.

Architecture:
    Edge branch (Sobel-initialized) → C//2 channels
    Texture branch (local variance) → C//2 channels
    → Concat → SE Channel Attention → Fusion → Additive Residual

Dataset: NEU-DET (6 defect classes, 70/20/10 clean split)
Result:  mAP@0.5 = 81.98% (+2.63pp over YOLOv11n baseline) | 145 FPS

Classes exported:
    EdgeAwareConv  — Sobel-initialized convolution for linear defect detection
    TextureBranch  — Local variance for surface anomaly detection
    DAFEGate       — Full dual-branch module (Ultralytics-compatible)

Team:       Hazem Elerefy, Youssef Sherif, Mohamed Salah, Moamen Esmat,
            Mahmoud Hisham, Mohamed Awni
Supervisor: Dr. Tarek Ghoneimy
Product:    DigiSteel (Digilians / MCIT)
"""

import torch
import torch.nn as nn


class EdgeAwareConv(nn.Module):
    """
    Convolution initialized with Sobel filters for edge detection.

    First two filters are initialized as Sobel-X and Sobel-Y operators.
    Remaining filters use Kaiming initialization. All weights are learnable,
    so the network can adapt the edge detectors during training.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self._init_sobel()

    def _init_sobel(self):
        """Initialize first filters with Sobel, rest with Kaiming."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            if self.conv.weight.shape[0] >= 1:
                for c in range(self.conv.weight.shape[1]):
                    self.conv.weight[0, c] = sobel_x
            if self.conv.weight.shape[0] >= 2:
                for c in range(self.conv.weight.shape[1]):
                    self.conv.weight[1, c] = sobel_y
            if self.conv.weight.shape[0] > 2:
                nn.init.kaiming_normal_(self.conv.weight[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TextureBranch(nn.Module):
    """
    Texture branch using local variance.

    Computes local variance to capture texture irregularities like pitting,
    scale, and inclusions. These defects manifest as local texture changes
    rather than edges.
    """

    def __init__(self, channels: int, pool_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=pool_size, stride=1,
            padding=pool_size // 2, count_include_pad=False,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = self.avg_pool(x)
        local_var = self.avg_pool(x * x) - local_mean * local_mean
        local_var = torch.clamp(local_var, min=0.0)
        return self.conv(local_var)


class DAFEGate(nn.Module):
    """
    Defect-Aware Feature Enhancement Gate (v4) for Ultralytics YOLO.

    Combines proven components from DAFE v2 with Ultralytics-compatible
    (c1, c2) signature for seamless YAML integration.

    Architecture (per forward pass):
        input x: (B, C, H, W)
          |-> EdgeAwareConv  -> (B, C//2, H, W)   [Sobel-initialized edge features]
          |-> TextureBranch  -> (B, C//2, H, W)   [local variance texture features]
          |-> concat         -> (B, C, H, W)
          |-> channel_att    -> (B, C, 1, 1)       [squeeze-excite over C]
          |-> fusion conv    -> (B, C, H, W)       [1x1 conv + BN + SiLU]
          |-> alpha * enhanced
        output: x + alpha * enhanced               [additive residual, always]

    Key design choices:
    - Additive residual (not multiplicative gate): gradient=1.0 through skip
      path regardless of alpha. Multiplicative gates halve gradients when
      gate~0.5, causing slow learning death spiral over long training.
    - C//2 channel split: forces edge and texture branches to specialize.
      Full-C branches learn redundant features (confirmed in v3 experiments).
    - Channel attention: lets the model decide which channels to enhance.
      Missing in v3 (replaced by simpler gate), restored here.
    - alpha_raw=-2.2 -> sigmoid(-2.2) ~ 0.1 at init: near-identity at epoch 0,
      preserving pretrained COCO backbone features. Learns freely during training.
    - Ultralytics YAML API: (c1, c2) signature. Lazy init when c1=0.
    """

    def __init__(self, c1: int = 0, c2: int = None):
        super().__init__()
        self._channels = c1
        self.reduction = 8
        if c1 > 0:
            self._build(c1)

    def _build(self, channels: int):
        if channels % 2 != 0:
            raise ValueError(f"DAFEGate requires even channel count, got {channels}")

        self._channels = channels
        branch_ch = channels // 2

        self.edge_branch = EdgeAwareConv(channels, branch_ch)
        self.texture_branch = TextureBranch(branch_ch)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid(),
        )

        if not hasattr(self, "alpha_raw"):
            self.alpha_raw = nn.Parameter(torch.tensor(-2.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self._channels:
            self._build(x.shape[1])
            self.to(x.device)

        edge_feat = self.edge_branch(x)
        texture_feat = self.texture_branch(edge_feat)

        fused = torch.cat([edge_feat, texture_feat], dim=1)
        att = self.channel_att(fused).unsqueeze(-1).unsqueeze(-1)
        enhanced = self.fusion(fused * att)

        alpha = torch.sigmoid(self.alpha_raw)
        return x + alpha * enhanced
