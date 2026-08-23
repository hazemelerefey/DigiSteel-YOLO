from __future__ import annotations

import torch
import torch.nn as nn


class EdgeAwareConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self._init_sobel()

    def _init_sobel(self) -> None:
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
                for channel in range(self.conv.weight.shape[1]):
                    self.conv.weight[0, channel] = sobel_x
            if self.conv.weight.shape[0] >= 2:
                for channel in range(self.conv.weight.shape[1]):
                    self.conv.weight[1, channel] = sobel_y
            if self.conv.weight.shape[0] > 2:
                nn.init.kaiming_normal_(self.conv.weight[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TextureBranch(nn.Module):
    def __init__(self, channels: int, pool_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
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
    def __init__(self, c1: int = 0, c2: int | None = None):
        super().__init__()
        self._channels = c1
        self.reduction = 8
        if c1 > 0:
            self._build(c1)

    def _build(self, channels: int) -> None:
        if channels % 2 != 0:
            raise ValueError(f"DAFEGate requires even channel count, got {channels}")

        self._channels = channels
        branch_channels = channels // 2

        self.edge_branch = EdgeAwareConv(channels, branch_channels)
        self.texture_branch = TextureBranch(branch_channels)
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

        edge_features = self.edge_branch(x)
        texture_features = self.texture_branch(edge_features)
        fused = torch.cat([edge_features, texture_features], dim=1)
        attention = self.channel_att(fused).unsqueeze(-1).unsqueeze(-1)
        enhanced = self.fusion(fused * attention)
        alpha = torch.sigmoid(self.alpha_raw)
        return x + alpha * enhanced
