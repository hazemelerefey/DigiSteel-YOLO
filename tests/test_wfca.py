"""Unit tests for WFCA (Wavelet Frequency Channel Attention) module."""
import pytest
import torch


def test_wfca_1level_output_shape():
    """WFCA with 1-level DWT should preserve input shape."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=256, reduction=8, dwt_levels=1)
    x = torch.randn(2, 256, 80, 80)
    y = module(x)
    assert y.shape == (2, 256, 80, 80)


def test_wfca_2level_output_shape():
    """WFCA with 2-level DWT should preserve input shape."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=128, reduction=4, dwt_levels=2)
    x = torch.randn(2, 128, 160, 160)
    y = module(x)
    assert y.shape == (2, 128, 160, 160)


def test_wfca_odd_dimensions():
    """WFCA should handle odd spatial dimensions via padding."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=64, reduction=4, dwt_levels=1)
    x = torch.randn(1, 64, 37, 41)
    y = module(x)
    assert y.shape == (1, 64, 37, 41)


def test_wfca_2level_odd_dimensions():
    """2-level WFCA should handle odd dimensions."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=128, reduction=4, dwt_levels=2)
    x = torch.randn(1, 128, 77, 83)
    y = module(x)
    assert y.shape == (1, 128, 77, 83)


def test_wfca_backward():
    """Gradients should flow through WFCA correctly."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=64, reduction=4, dwt_levels=1)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_wfca_2level_backward():
    """Gradients should flow through 2-level WFCA."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=128, reduction=4, dwt_levels=2)
    x = torch.randn(1, 128, 40, 40, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_wfca_param_count_1level():
    """1-level WFCA (C=256, r=8) should have ~65K params per spec."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=256, reduction=8, dwt_levels=1)
    total = sum(p.numel() for p in module.parameters())
    # shared_fc: 4*256 * 32 = 32768, + bias 32
    # 4 subband FCs: 4 * (32*256 + 256) = 33792
    # alpha: 1
    assert 50_000 < total < 80_000, f"Expected ~65K params, got {total}"


def test_wfca_param_count_2level():
    """2-level WFCA (C=128, r=4) should have ~58K params."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=128, reduction=4, dwt_levels=2)
    total = sum(p.numel() for p in module.parameters())
    # 7 subbands: shared_fc: 7*128*32 = 28672, + bias 32
    # 7 FCs: 7*(32*128 + 128) = 29568
    # alpha: 1
    assert 40_000 < total < 75_000, f"Expected ~58K params, got {total}"


def test_wfca_alpha_initialization():
    """Alpha should initialize at 0.1."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=64, reduction=4, dwt_levels=1)
    assert abs(module.alpha.item() - 0.1) < 1e-6


def test_wfca_residual_connection():
    """With alpha=0, output should equal input (pure residual)."""
    from digisteel.modules.wfca import WFCA

    module = WFCA(channels=64, reduction=4, dwt_levels=1)
    with torch.no_grad():
        module.alpha.fill_(0.0)
    x = torch.randn(1, 64, 32, 32)
    y = module(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_haar_dwt_idwt_roundtrip():
    """DWT -> IDWT should perfectly reconstruct the input (even dims)."""
    from digisteel.modules.wfca import haar_dwt_2d, haar_idwt_2d

    x = torch.randn(2, 32, 64, 64)
    ll, lh, hl, hh = haar_dwt_2d(x)
    recon = haar_idwt_2d(ll, lh, hl, hh)
    assert torch.allclose(x, recon, atol=1e-5)
