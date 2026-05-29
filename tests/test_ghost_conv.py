"""Unit tests for GhostConv module."""
import pytest
import torch
from digisteel.modules import GhostConv, GhostModule


def test_ghost_module_output_shape():
    """GhostModule should output the correct shape."""
    module = GhostModule(64, 128, kernel_size=1, ratio=2)
    x = torch.randn(2, 64, 32, 32)
    y = module(x)
    assert y.shape == (2, 128, 32, 32)


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


def test_ghost_conv_fewer_params():
    """GhostConv should have fewer parameters than standard Conv2d."""
    ghost = GhostConv(in_channels=64, out_channels=128, kernel_size=3)
    standard = torch.nn.Conv2d(64, 128, 3, padding=1)

    ghost_params = sum(p.numel() for p in ghost.parameters())
    standard_params = sum(p.numel() for p in standard.parameters())

    assert ghost_params < standard_params, (
        f"GhostConv ({ghost_params}) should have fewer params than Conv2d ({standard_params})"
    )


def test_ghost_conv_param_ratio():
    """GhostConv should have roughly half the parameters of standard Conv2d."""
    ghost = GhostConv(in_channels=64, out_channels=128, kernel_size=3)
    standard = torch.nn.Conv2d(64, 128, 3, padding=1)

    ghost_params = sum(p.numel() for p in ghost.parameters())
    standard_params = sum(p.numel() for p in standard.parameters())
    ratio = ghost_params / standard_params

    # GhostConv should be roughly 50-70% of standard Conv2d
    assert 0.3 < ratio < 0.8, f"Param ratio {ratio:.2f} outside expected range [0.3, 0.8]"


def test_ghost_conv_backward():
    """GhostConv gradients should backprop correctly."""
    layer = GhostConv(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_ghost_conv_different_channels():
    """GhostConv should work with different channel configurations."""
    for in_ch, out_ch in [(32, 64), (64, 64), (128, 256)]:
        layer = GhostConv(in_channels=in_ch, out_channels=out_ch)
        x = torch.randn(1, in_ch, 16, 16)
        y = layer(x)
        assert y.shape == (1, out_ch, 16, 16)
