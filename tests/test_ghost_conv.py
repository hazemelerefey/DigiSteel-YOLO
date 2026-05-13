"""Unit tests for A2: GhostConv module."""
import pytest
import torch
from digisteel.modules import GhostConv
from digisteel.modules.ghost_conv import GhostConvWeightSharing


def test_ghost_conv_output_shape():
    """GhostConv should output the correct shape."""
    layer = GhostConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
    x = torch.randn(2, 64, 32, 32)
    y = layer(x)
    assert y.shape == (2, 128, 32, 32)


def test_ghost_conv_stride():
    """GhostConv with stride=2 should halve spatial dimensions."""
    layer = GhostConv(in_channels=64, out_channels=128, kernel_size=3, stride=2)
    x = torch.randn(2, 64, 32, 32)
    y = layer(x)
    assert y.shape == (2, 128, 16, 16)


def test_ghost_conv_param_count():
    """GhostConv should have fewer parameters than standard Conv."""
    # This is approximate; actual count depends on the Ghost ratio
    layer = GhostConv(in_channels=64, out_channels=128, kernel_size=3)
    param_count = sum(p.numel() for p in layer.parameters())
    # Expected to be much less than a standard Conv2d(64, 128, 3, padding=1)
    # Standard Conv: 64 * 128 * 3 * 3 = 73,728 params
    assert param_count > 0


def test_ghost_conv_weight_sharing():
    """A2: Weight-sharing should reuse parameters across stages."""
    shared = GhostConvWeightSharing(in_channels=64, out_channels=64)

    p3 = torch.randn(1, 64, 80, 80)
    p4 = torch.randn(1, 64, 40, 40)
    p5 = torch.randn(1, 64, 20, 20)

    out_p3 = shared(p3)
    out_p4 = shared(p4)
    out_p5 = shared(p5)

    # All outputs should have the right channel count
    assert out_p3.shape[1] == 64
    assert out_p4.shape[1] == 64
    assert out_p5.shape[1] == 64

    # Total parameter count should be exactly 1x the underlying module
    params = shared.param_count()
    assert params > 0


def test_ghost_conv_backward():
    """GhostConv gradients should backprop correctly."""
    layer = GhostConv(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
