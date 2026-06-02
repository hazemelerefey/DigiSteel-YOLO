"""Tests for DigiSteel custom trainer."""
import pytest
import torch


def test_register_modules_available():
    """register_custom_modules function should be importable."""
    from digisteel.engine.trainer import register_custom_modules
    assert callable(register_custom_modules)


def test_inner_wiou_adapter_output_shape():
    """InnerWIoUAdapter should return per-box IoU values."""
    from digisteel.engine.trainer import InnerWIoUAdapter

    adapter = InnerWIoUAdapter(lambda_weight=0.5)
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]])
    target = torch.tensor([[10.0, 10.0, 50.0, 50.0], [25.0, 25.0, 65.0, 65.0]])
    result = adapter(pred, target)
    assert result.shape == (2,)
    assert (result >= 0).all()
    assert (result <= 1).all()


def test_inner_wiou_adapter_perfect_match():
    """InnerWIoUAdapter should give high IoU for perfect match."""
    from digisteel.engine.trainer import InnerWIoUAdapter

    adapter = InnerWIoUAdapter(lambda_weight=0.5)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    result = adapter(boxes, boxes)
    assert result[0].item() > 0.9


def test_patch_model_for_digisteel_importable():
    """patch_model_for_digisteel should be importable."""
    from digisteel.engine.trainer import patch_model_for_digisteel
    assert callable(patch_model_for_digisteel)


def test_digisteel_trainer_class_exists():
    """DigiSteelTrainer class should be importable."""
    from digisteel.engine.trainer import DigiSteelTrainer
    assert DigiSteelTrainer is not None
