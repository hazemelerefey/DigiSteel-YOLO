"""
GhostConv Module for Lightweight Steel Defect Detection.

Implementation of Ghost convolution (Han et al., 2020, GhostNet) as a
drop-in replacement for standard Conv2d blocks in YOLO backbones.

GhostConv generates "primary" feature maps via a standard convolution,
then produces "ghost" feature maps via cheap linear transformations
(depthwise convolution), reducing parameters while maintaining accuracy.

This module is used as a lightweight backbone variant in the DigiSteel-YOLO
robustness study. It is NOT claimed as a novel contribution — it is a
well-established technique (CVPR 2020) applied to steel defect detection.

References:
    - Han et al. 2020, GhostNet: More Features from Cheap Operations (CVPR 2020)
      arXiv:1911.11907
    - Applied to steel defect detection by:
      P01 PSF-YOLO (Scientific Reports 2025, DOI: 10.1038/s41598-025-16619-9)
      P02 LAM-YOLOv10n (Scientific Reports 2025, DOI: 10.1038/s41598-025-16725-8)
      P04 Lightweight-YOLOv8 (Scientific Reports 2025, DOI: 10.1038/s41598-025-93469-5)
"""

import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """
    Ghost Module: generates more features from cheap operations.

    Instead of learning all filters independently, learn a subset of "primary"
    features and generate "ghost" features via cheap depthwise convolutions.

    This reduces parameters by ~50% compared to a standard convolution while
    maintaining similar representational capacity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the primary convolution.
        ratio: Expansion ratio (2 means ~2x features from primary + ghost).
        dw_size: Kernel size for the cheap depthwise convolution.
        stride: Stride of the primary convolution.
        use_bn: Whether to use batch normalization.
        act: Activation function ("relu" or "identity").

    Example:
        >>> module = GhostModule(64, 128)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> y = module(x)  # shape: (1, 128, 32, 32)
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
    GhostConv: Drop-in replacement for Conv2d using Ghost modules.

    Replaces a standard convolution with a Ghost module that generates
    features more efficiently. Can be used anywhere Conv2d is used in
    a YOLO backbone for parameter reduction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        groups: Number of groups (unused, kept for interface compatibility).
        dilation: Dilation rate (unused, kept for interface compatibility).
        bias: Whether to use bias (unused, controlled by use_bn in GhostModule).

    Example:
        >>> conv = GhostConv(64, 128, kernel_size=3, stride=1)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> y = conv(x)  # shape: (1, 128, 32, 32)
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
        self.conv = GhostModule(
            in_channels,
            out_channels,
            kernel_size,
            ratio=2,
            dw_size=3,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
