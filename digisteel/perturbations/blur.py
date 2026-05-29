"""
Blur perturbations for robustness evaluation.

Simulates camera defocus and motion artifacts common in industrial settings.
"""

import numpy as np
import cv2


class GaussianBlur:
    """
    Gaussian blur perturbation simulating camera defocus.

    Severity levels:
        1 (mild):    sigma=1  — slight softness
        2 (moderate): sigma=3  — noticeable blur
        3 (severe):  sigma=5  — significant detail loss
        4 (extreme): sigma=7  — critical detail loss

    Args:
        level: Severity level (1-4).

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: 1, 2: 3, 3: 5, 4: 7}

    def __init__(self, level: int):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.sigma = self.LEVELS[level]
        self.level = level

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        ksize = int(2 * round(3 * self.sigma) + 1)
        return cv2.GaussianBlur(image, (ksize, ksize), self.sigma)

    def __repr__(self) -> str:
        return f"GaussianBlur(level={self.level}, sigma={self.sigma})"


class MotionBlur:
    """
    Motion blur perturbation simulating conveyor belt vibration or camera shake.

    Uses a linear kernel to simulate directional motion blur.

    Severity levels:
        1 (mild):    kernel_size=3  — slight streaking
        2 (moderate): kernel_size=5  — visible motion
        3 (severe):  kernel_size=7  — significant streaking
        4 (extreme): kernel_size=9  — critical motion artifact

    Args:
        level: Severity level (1-4).

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: 3, 2: 5, 3: 7, 4: 9}

    def __init__(self, level: int):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.kernel_size = self.LEVELS[level]
        self.level = level

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply motion blur to image."""
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[self.kernel_size // 2, :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        return cv2.filter2D(image, -1, kernel)

    def __repr__(self) -> str:
        return f"MotionBlur(level={self.level}, kernel_size={self.kernel_size})"
