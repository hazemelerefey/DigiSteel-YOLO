"""Unit tests for A3: Inner-WIoU loss module."""
import pytest
import torch
from digisteel.modules.inner_wiou import InnerWIoULoss, inner_iou_loss, wiou_v3_loss


def test_inner_iou_loss_perfect_boxes():
    """Perfect overlap should give loss close to 0."""
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    target = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    loss = inner_iou_loss(pred, target)
    assert loss.item() < 0.1  # Should be very small


def test_wiou_v3_loss_perfect_boxes():
    """Perfect overlap should give loss close to 0."""
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    target = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    loss = wiou_v3_loss(pred, target)
    assert loss.item() < 0.1


def test_inner_wiou_loss_default():
    """InnerWIoULoss with default lambda=0.5 should work."""
    loss_fn = InnerWIoULoss(lambda_weight=0.5)
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]] * 4)
    target = torch.tensor([[10.0, 10.0, 50.0, 50.0]] * 4)
    loss = loss_fn(pred, target)
    assert loss.item() < 0.1


def test_inner_wiou_loss_lambda_zero():
    """lambda=0 should give only WIoU v3 loss."""
    loss_fn = InnerWIoULoss(lambda_weight=0.0)
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    target = torch.tensor([[15.0, 15.0, 55.0, 55.0]])
    loss = loss_fn(pred, target)
    # Should be roughly the WIoU loss
    assert loss.item() > 0.0


def test_inner_wiou_loss_lambda_one():
    """lambda=1 should give only Inner-IoU loss."""
    loss_fn = InnerWIoULoss(lambda_weight=1.0)
    pred = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    target = torch.tensor([[15.0, 15.0, 55.0, 55.0]])
    loss = loss_fn(pred, target)
    # Should be roughly the Inner-IoU loss
    assert loss.item() > 0.0


def test_inner_wiou_loss_batch():
    """Should handle batch inputs correctly."""
    loss_fn = InnerWIoULoss()
    batch_size = 32
    pred = torch.randn(batch_size, 4) * 100  # Random boxes
    target = torch.randn(batch_size, 4) * 100
    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([])  # Scalar


def test_inner_wiou_loss_backward():
    """Gradients should flow correctly."""
    loss_fn = InnerWIoULoss()
    pred = torch.randn(8, 4, requires_grad=True)
    target = torch.randn(8, 4)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None


def test_inner_wiou_loss_repr():
    """String representation should be informative."""
    loss_fn = InnerWIoULoss(lambda_weight=0.5)
    s = str(loss_fn)
    assert "InnerWIoULoss" in s
    assert "lambda" in s.lower()
