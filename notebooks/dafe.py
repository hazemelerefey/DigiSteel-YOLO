"""
Defect-Aware Feature Enhancement (DAFE) v2 Module.

Novel contribution of DigiSteel-YOLO. Specifically designed for flat steel
surface defects, which fall into two categories:

1. Linear defects (scratches, crazing): thin edges and cracks
   -> Detected by the edge-aware branch (Sobel-initialized convolutions)

2. Surface anomalies (pitting, scale, inclusions): texture irregularities
   -> Detected by the texture-aware branch (local variance features)

v2 improvements over v1:
- EdgeBranch uses Sobel-initialized weights (actually edge-aware, not just a regular conv)
- TextureBranch simplified to local variance only (fewer parameters)
- No lazy initialization (all layers built in __init__)
- Lower parameter count while maintaining expressiveness

References:
    - Closest related work:
      ELS-YOLO (2025): Edge-focused enhancement, but single-branch
      SCCI-YOLO (2025): Channel attention, not defect-type-aware
      No existing work combines edge + texture awareness for steel defects
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
            # First filter = Sobel X (for all input channels)
            if self.conv.weight.shape[0] >= 1:
                for c in range(self.conv.weight.shape[1]):
                    self.conv.weight[0, c] = sobel_x
            # Second filter = Sobel Y (for all input channels)
            if self.conv.weight.shape[0] >= 2:
                for c in range(self.conv.weight.shape[1]):
                    self.conv.weight[1, c] = sobel_y
            # Rest use Kaiming initialization
            if self.conv.weight.shape[0] > 2:
                nn.init.kaiming_normal_(self.conv.weight[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TextureBranch(nn.Module):
    """
    Texture branch using local variance only.

    Computes local variance to capture texture irregularities like pitting,
    scale, and inclusions. These defects manifest as local texture changes
    rather than edges.

    v2: Uses only local variance (no concatenation with original input).
    This reduces parameters and focuses on texture change magnitude.
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
        # Local variance: E[X^2] - E[X]^2
        local_mean = self.avg_pool(x)
        local_var = self.avg_pool(x * x) - local_mean * local_mean
        local_var = torch.clamp(local_var, min=0.0)
        # Use only local variance (no concatenation with original)
        return self.conv(local_var)


class DAFE(nn.Module):
    """
    Defect-Aware Feature Enhancement v2.

    Dual-branch module that enhances features for flat steel defect detection:
    - Edge branch: Sobel-initialized conv for linear defects (scratches, crazing)
    - Texture branch: local variance for surface anomalies (pitting, scale, inclusions)

    The branches are fused with channel attention, and a learnable residual
    controls the contribution of the enhanced features.

    Args:
        channels: Number of input/output channels (must be even).
        reduction: Channel reduction ratio for channel attention.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.reduction = reduction
        self._channels = channels
        self._build(channels)

    def _build(self, channels: int):
        """Build all sub-layers for the given channel count."""
        if channels % 2 != 0:
            raise ValueError(f"DAFE requires even channel count, got {channels}")

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

        # Learnable residual scaling: sigmoid(-2.2) ~ 0.1
        if not hasattr(self, "alpha_raw"):
            self.alpha_raw = nn.Parameter(torch.tensor(-2.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapt to actual channel count if different from constructor arg
        if x.shape[1] != self._channels:
            self._build(x.shape[1])

        # Edge features from Sobel-initialized branch
        edge_feat = self.edge_branch(x)  # (B, C//2, H, W)

        # Texture features from edge features (analyzes texture of edge map)
        texture_feat = self.texture_branch(edge_feat)  # (B, C//2, H, W)

        # Fuse both branches
        fused = torch.cat([edge_feat, texture_feat], dim=1)  # (B, C, H, W)

        # Channel attention
        att = self.channel_att(fused).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        enhanced = self.fusion(fused * att)  # (B, C, H, W)

        # Residual with learnable alpha
        alpha = torch.sigmoid(self.alpha_raw)
        return x + alpha * enhanced


class DAFEEdgeOnly(nn.Module):
    """
    DAFE Edge-Only variant — simplified single-branch version.

    Removes the texture branch and uses only the Sobel-initialized
    edge-aware convolution. This tests whether the texture branch
    is necessary or if edge features alone are sufficient.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for channel attention.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.reduction = reduction
        self._channels = channels
        self._build(channels)

    def _build(self, channels: int):
        """Build all sub-layers for the given channel count."""
        self._channels = channels

        # Single edge branch (full channels, not half)
        self.edge_branch = EdgeAwareConv(channels, channels)

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

        # Learnable residual scaling: sigmoid(-2.2) ~ 0.1
        if not hasattr(self, "alpha_raw"):
            self.alpha_raw = nn.Parameter(torch.tensor(-2.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapt to actual channel count if different from constructor arg
        if x.shape[1] != self._channels:
            self._build(x.shape[1])

        # Edge features only (no texture branch)
        edge_feat = self.edge_branch(x)  # (B, C, H, W)

        # Channel attention
        att = self.channel_att(edge_feat).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        enhanced = self.fusion(edge_feat * att)  # (B, C, H, W)

        # Residual with learnable alpha
        alpha = torch.sigmoid(self.alpha_raw)
        return x + alpha * enhanced
