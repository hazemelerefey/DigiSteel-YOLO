"""Unit tests for EMA (Efficient Multi-scale Attention) module."""
import pytest
import torch


def test_ema_output_shape():
    """EMA should preserve input shape."""
    from digisteel.modules.ema import EMA

    module = EMA(channels=256)
    x = torch.randn(2, 256, 80, 80)
    y = module(x)
    assert y.shape == (2, 256, 80, 80)


def test_ema_different_channels():
    """EMA should work with different channel counts."""
    from digisteel.modules.ema import EMA

    for ch in [128, 256, 512, 1024]:
        module = EMA(channels=ch)
        x = torch.randn(1, ch, 20, 20)
        y = module(x)
        assert y.shape == (1, ch, 20, 20)


def test_ema_backward():
    """Gradients should flow through EMA."""
    from digisteel.modules.ema import EMA

    module = EMA(channels=128)
    x = torch.randn(2, 128, 40, 40, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_ema_small_spatial():
    """EMA should handle small spatial dimensions."""
    from digisteel.modules.ema import EMA

    module = EMA(channels=512)
    x = torch.randn(1, 512, 5, 5)
    y = module(x)
    assert y.shape == (1, 512, 5, 5)


def test_ema_param_count():
    """EMA should be lightweight (<50K params for C=256)."""
    from digisteel.modules.ema import EMA

    module = EMA(channels=256)
    total = sum(p.numel() for p in module.parameters())
    assert total < 50_000, f"EMA should be lightweight, got {total} params"
