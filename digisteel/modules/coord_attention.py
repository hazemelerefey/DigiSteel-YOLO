"""
Coordinate Attention Module.

Reference:
    Hou et al., "Coordinate Attention for Efficient Mobile Network Design," CVPR 2021.

Coordinate Attention decomposes channel attention into two 1D feature encoding
processes along orthogonal spatial directions (horizontal and vertical). This
preserves long-range dependencies along one direction while maintaining precise
positional information along the other — critical for localizing fine defects
like crazing and scratches at 200×200 resolution.

Unlike WFCA which loses spatial info through GAP + DWT, CA encodes spatial
coordinates explicitly through directional pooling.

Lazy initialization: conv layers are built on first forward pass to match the
actual input channels (handles YOLO scale factor multiplication correctly).
"""

import torch
import torch.nn as nn


class CoordAttention(nn.Module):
    """
    Coordinate Attention module.

    Architecture:
        Input (B, C, H, W)
        ├─ X Avg Pool (H, 1) → permute → concat
        ├─ Y Avg Pool (1, W) → ──────────┘
        ├─ 1×1 Conv (C/r) → BN → h_swish
        ├─ Split → f_h (C/r, H), f_w (C/r, W)
        ├─ Conv_h → C channels → sigmoid → (C, H, 1)
        ├─ Conv_w → C channels → sigmoid → (C, 1, W)
        └─ Multiply × input → output

    Args:
        inp: Input channels (used if oup not specified).
        oup: Output channels (defaults to inp).
        reduction: Channel reduction ratio (default 32).
    """

    def __init__(self, inp: int, oup: int = None, reduction: int = 32):
        super().__init__()
        self.inp = inp
        self.oup = oup or inp
        self.reduction = reduction
        self._built = False

        # Horizontal and vertical pooling (no learned params)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def _build(self, actual_channels: int):
        """Build conv layers lazily to match actual input channels."""
        self.inp = actual_channels
        self.oup = actual_channels  # output = input channels (identity residual)
        mid_channels = max(8, actual_channels // self.reduction)
        device = next(self.pool_h.parameters()).device \
            if any(p is not None for p in [next(self.pool_h.parameters(), None)]) \
            else None

        # Shared 1×1 conv after directional concat
        self.conv1 = nn.Conv2d(actual_channels, mid_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)

        # Separate 1×1 projections per direction → sigmoid attention
        self.conv_h = nn.Conv2d(mid_channels, actual_channels, 1, 1, 0)
        self.conv_w = nn.Conv2d(mid_channels, actual_channels, 1, 1, 0)

        if device is not None:
            self.to(device)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1])

        identity = x
        B, C, H, W = x.shape

        # Directional pooling
        x_h = self.pool_h(x)   # (B, C, H, 1)
        x_w = self.pool_w(x)   # (B, C, 1, W)

        # Concatenate along spatial axis: (B, C, H+W, 1)
        x_hw = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)

        # Shared transform
        y = self.conv1(x_hw)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into directions
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)

        # Directional attention weights
        a_h = torch.sigmoid(self.conv_h(x_h))  # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))  # (B, C, 1, W)

        # Apply attention
        return identity * a_h * a_w