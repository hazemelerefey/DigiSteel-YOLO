"""
Brightness and contrast perturbations for robustness evaluation.

Simulates lighting variations and environmental interference in industrial settings.
"""

import numpy as np


class BrightnessShift:
    """
    Brightness shift perturbation simulating lighting variation.

    Adds a constant value to all pixels. Positive values simulate
    overexposure (bright lights), negative values simulate underexposure
    (shadows, dim lighting).

    Severity levels:
        1 (mild):    delta=-30 — slight darkening
        2 (moderate): delta=-50 — noticeable darkening
        3 (severe):  delta=+30 — noticeable brightening
        4 (extreme): delta=+50 — significant brightening

    Args:
        level: Severity level (1-4).

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: -30, 2: -50, 3: 30, 4: 50}

    def __init__(self, level: int):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.delta = self.LEVELS[level]
        self.level = level

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness shift to image."""
        shifted = image.astype(np.int16) + self.delta
        return np.clip(shifted, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        return f"BrightnessShift(level={self.level}, delta={self.delta})"


class ContrastReduction:
    """
    Contrast reduction perturbation simulating fog, dust, or environmental haze.

    Reduces image contrast by scaling pixel values toward the mean.
    Formula: output = mean + factor * (input - mean)

    Severity levels:
        1 (mild):    factor=0.8 — slight contrast loss
        2 (moderate): factor=0.6 — noticeable contrast loss
        3 (severe):  factor=0.4 — significant contrast loss
        4 (extreme): factor=0.2 — critical contrast loss (washed out)

    Args:
        level: Severity level (1-4).

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}

    def __init__(self, level: int):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.factor = self.LEVELS[level]
        self.level = level

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast reduction to image."""
        mean_val = np.mean(image)
        reduced = mean_val + self.factor * (image.astype(np.float64) - mean_val)
        return np.clip(reduced, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        return f"ContrastReduction(level={self.level}, factor={self.factor})"
