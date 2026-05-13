"""
A2: GhostConv Weight-Sharing Module

Implementation of Ghost convolution (Han et al., 2020, GhostNet)
with weight-sharing across pyramid stages P3, P4, P5.

Novelty: Single weight tensor reused across three pyramid stages,
reducing parameters by ~33% compared to independent GhostConv blocks.

Reference: Han et al. 2020, GhostNet (https://arxiv.org/abs/2005.04675)
Applied to steel defect detection by: P01 PSF-YOLO, P02 LAM-YOLOv10n, P04 Lightweight-YOLOv8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule(nn.Module):
    """
    Ghost Module: cheap operation to generate more features from feature maps.
    
    Instead of learning all filters independently, learn a subset of "primary" features
    and generate "ghost" features via cheap transformations (e.g., depthwise convs).
    
    Args:
        in_channels: Input channels
        out_channels: Output channels (must be even for ghost generation)
        kernel_size: Kernel size for primary convolution
        ratio: Ratio of ghost features to primary features (default: 2, so ~2x expansion)
        dw_size: Kernel size for cheap depthwise convolution
        stride: Stride of the primary convolution
        use_bn: Whether to use batch normalization (default: True)
        act: Activation function (default: ReLU)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        use_bn: bool = True,
        act: str = "relu",
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=not use_bn,
            ),
            nn.BatchNorm2d(init_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act == "relu" else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=not use_bn,
            ),
            nn.BatchNorm2d(new_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act == "relu" else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]


class GhostConv(nn.Module):
    """
    GhostConv: Conv + GhostModule with optional stride.
    
    Standard entry point for using Ghost convolutions as a drop-in replacement
    for standard Conv2d blocks in neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = GhostModule(in_channels, out_channels, kernel_size, ratio=2, dw_size=3, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GhostConvWeightSharing(nn.Module):
    """
    A2 Novelty: Weight-Sharing GhostConv across pyramid stages.
    
    A single GhostModule is instantiated once and its forward pass is called
    three times on P3, P4, and P5 feature maps, reusing the learned weights.
    
    This reduces the parameter count by 2/3 for a backbone with three pyramid stages.
    
    Example:
        >>> shared_ghost = GhostConvWeightSharing(64, 64)
        >>> p3 = torch.randn(1, 64, 80, 80)  # imagenet 640 -> P3
        >>> p4 = torch.randn(1, 64, 40, 40)  # imagenet 640 -> P4
        >>> p5 = torch.randn(1, 64, 20, 20)  # imagenet 640 -> P5
        >>> out_p3 = shared_ghost(p3)
        >>> out_p4 = shared_ghost(p4)
        >>> out_p5 = shared_ghost(p5)
        >>> # Total params = 1 × GhostModule, not 3 × GhostModule
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio: int = 2,
        dw_size: int = 3,
    ):
        super().__init__()
        self.shared_ghost = GhostModule(
            in_channels, out_channels, kernel_size=kernel_size, ratio=ratio, dw_size=dw_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reuses the shared ghost module."""
        return self.shared_ghost(x)

    def param_count(self) -> int:
        """Return the number of parameters in the shared module."""
        return sum(p.numel() for p in self.shared_ghost.parameters() if p.requires_grad)
