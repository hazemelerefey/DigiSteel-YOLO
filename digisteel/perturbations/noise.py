"""
Noise perturbation for robustness evaluation.

Simulates sensor noise and electrical interference common in industrial cameras.
"""

import numpy as np


class GaussianNoise:
    """
    Gaussian noise perturbation simulating sensor noise.

    Adds zero-mean Gaussian noise with varying standard deviation.
    Common in industrial cameras due to electrical interference,
    low-light conditions, or sensor degradation.

    Severity levels:
        1 (mild):    sigma=0.05 (≈13/255) — slight grain
        2 (moderate): sigma=0.10 (≈26/255) — visible noise
        3 (severe):  sigma=0.20 (≈51/255) — significant noise
        4 (extreme): sigma=0.30 (≈77/255) — critical noise

    Args:
        level: Severity level (1-4).
        seed: Random seed for reproducibility. Default: None.

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30}

    def __init__(self, level: int, seed: int = None):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.sigma = self.LEVELS[level]
        self.level = level
        self.seed = seed

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to image.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W), dtype uint8.

        Returns:
            Noisy image as numpy array, dtype uint8.
        """
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, self.sigma, image.shape)
        noisy = image.astype(np.float64) / 255.0 + noise
        noisy = np.clip(noisy, 0.0, 1.0)
        return (noisy * 255).astype(np.uint8)

    def __repr__(self) -> str:
        return f"GaussianNoise(level={self.level}, sigma={self.sigma})"
