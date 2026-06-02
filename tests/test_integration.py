"""Integration tests: model builds, forwards, and has expected structure."""
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration

# Skip all tests if ultralytics is not installed
try:
    import ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

skip_without_ultralytics = pytest.mark.skipif(
    not HAS_ULTRALYTICS, reason="ultralytics not installed"
)


@skip_without_ultralytics
def test_digisteel_v2_builds():
    """DigiSteel v2 model should build from YAML without errors."""
    from digisteel.engine.trainer import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))
    assert model is not None


@skip_without_ultralytics
def test_digisteel_v2_param_count():
    """DigiSteel v2 should have ~2.0M parameters (lighter than 2.58M baseline)."""
    from digisteel.engine.trainer import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))

    info = model.info(verbose=False)
    if isinstance(info, (list, tuple)) and len(info) >= 1:
        params = info[0]
    else:
        params = sum(p.numel() for p in model.model.parameters())

    params_m = params / 1e6 if params > 1e4 else params
    print(f"DigiSteel v2 params: {params_m:.2f}M")

    assert 1.0 < params_m < 3.5, f"Expected ~2.0M params, got {params_m:.2f}M"


@skip_without_ultralytics
def test_digisteel_v2_forward():
    """DigiSteel v2 should forward a dummy batch without errors."""
    import torch
    from digisteel.engine.trainer import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(dummy)

    assert output is not None


@skip_without_ultralytics
def test_digisteel_v2_has_wfca():
    """Model should contain WFCA modules."""
    from digisteel.engine.trainer import register_custom_modules
    from digisteel.modules.wfca import WFCA
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))

    wfca_count = sum(1 for m in model.model.modules() if isinstance(m, WFCA))
    assert wfca_count == 2, f"Expected 2 WFCA modules, found {wfca_count}"


@skip_without_ultralytics
def test_digisteel_v2_has_ema():
    """Model should contain EMA modules."""
    from digisteel.engine.trainer import register_custom_modules
    from digisteel.modules.ema import EMA
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))

    ema_count = sum(1 for m in model.model.modules() if isinstance(m, EMA))
    assert ema_count == 4, f"Expected 4 EMA modules, found {ema_count}"


@skip_without_ultralytics
def test_digisteel_v2_has_ghost_conv():
    """Model should contain GhostConv modules."""
    from digisteel.engine.trainer import register_custom_modules
    from digisteel.modules.ghost_conv import GhostConv
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(PROJECT_ROOT / "configs" / "models" / "digisteel_v2.yaml"))

    gc_count = sum(1 for m in model.model.modules() if isinstance(m, GhostConv))
    assert gc_count >= 4, f"Expected >=4 GhostConv modules, found {gc_count}"
