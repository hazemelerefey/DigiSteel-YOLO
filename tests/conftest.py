"""Test configuration and fixtures."""
import pytest


@pytest.fixture
def sample_boxes():
    """Sample bounding boxes for testing."""
    import torch
    # (N, 4) in [x1, y1, x2, y2] format
    return torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],  # box 1
            [20.0, 20.0, 60.0, 60.0],  # box 2
        ]
    )
