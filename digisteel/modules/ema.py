"""
Efficient Multi-scale Attention (EMA) Module.

Applies parallel multi-scale attention using grouped 1D convolutions across
spatial dimensions, enabling the network to attend to features at multiple
scales simultaneously with minimal overhead.

NOT a novel contribution — this is an established attention mechanism applied
to the DigiSteel-YOLO neck for multi-scale feature refinement.

Reference:
    Ouyang et al. 2023, Efficient Multi-Scale Attention Module with
    Cross-Spatial Learning, ICASSP 2023.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention.

    Splits channels into groups, applies parallel 1D strip pooling (horizontal
    and vertical) per group to capture multi-scale spatial context, then
    recombines via cross-spatial attention.

    Args:
        channels: Number of input/output channels.
        groups: Number of channel groups for multi-scale processing.
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self._built = False

        # Dummy parameter for device detection (EMA has no learnable params in __init__)
        self._device_anchor = nn.Parameter(torch.empty(0))

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

    def _build(self, actual_channels: int):
        """Lazily build layers based on actual input channels."""
        assert actual_channels % self.groups == 0, (
            f"channels ({actual_channels}) must be divisible by groups ({self.groups})"
        )
        group_ch = actual_channels // self.groups

        self.gn = nn.GroupNorm(self.groups, actual_channels)
        self.conv_hw = nn.Conv2d(group_ch, group_ch, kernel_size=1)
        self.conv_pool_h = nn.Conv1d(group_ch, group_ch, kernel_size=3, padding=1)
        self.conv_pool_w = nn.Conv1d(group_ch, group_ch, kernel_size=3, padding=1)

        # Move to same device as anchor parameter
        device = self._device_anchor.device
        self.gn = self.gn.to(device)
        self.conv_hw = self.conv_hw.to(device)
        self.conv_pool_h = self.conv_pool_h.to(device)
        self.conv_pool_w = self.conv_pool_w.to(device)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Lazy initialization on first forward pass
        if not self._built:
            self._build(C)

        # Group normalization
        x_gn = self.gn(x)

        # Split into groups: (B*groups, C//groups, H, W)
        group_ch = C // self.groups
        x_groups = x_gn.reshape(B * self.groups, group_ch, H, W)

        # Spatial attention via strip pooling
        pool_h = self.avg_pool_h(x_groups)  # (B*G, C//G, H, 1)
        pool_w = self.avg_pool_w(x_groups)  # (B*G, C//G, 1, W)

        # 1D conv along each strip
        h_att = self.conv_pool_h(pool_h.squeeze(-1))  # (B*G, C//G, H)
        w_att = self.conv_pool_w(pool_w.squeeze(-2))  # (B*G, C//G, W)

        # Cross-spatial: combine H and W attention
        h_att = h_att.unsqueeze(-1)  # (B*G, C//G, H, 1)
        w_att = w_att.unsqueeze(-2)  # (B*G, C//G, 1, W)

        # Spatial attention map
        hw_att = torch.sigmoid(self.conv_hw(x_groups))  # (B*G, C//G, H, W)

        # Combine: element-wise product of all three attention paths
        att = hw_att * torch.sigmoid(h_att) * torch.sigmoid(w_att)

        # Reshape back and apply as multiplicative attention
        att = att.reshape(B, C, H, W)
        return x * att
