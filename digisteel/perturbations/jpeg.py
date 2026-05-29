"""
JPEG compression perturbation for robustness evaluation.

Simulates transmission and storage artifacts common in industrial
image acquisition systems.
"""

import numpy as np
import cv2


class JPEGCompression:
    """
    JPEG compression perturbation simulating transmission artifacts.

    Industrial systems often compress images before transmission or storage.
    Lower quality factors introduce more compression artifacts (blocking,
    ringing, color bleeding).

    Severity levels:
        1 (mild):    quality=80 — slight artifacts
        2 (moderate): quality=50 — visible blocking
        3 (severe):  quality=30 — significant artifacts
        4 (extreme): quality=15 — critical artifacts

    Args:
        level: Severity level (1-4).

    Returns:
        Perturbation function that takes a numpy image and returns degraded image.
    """

    LEVELS = {1: 80, 2: 50, 3: 30, 4: 15}

    def __init__(self, level: int):
        if level not in self.LEVELS:
            raise ValueError(f"Level must be 1-4, got {level}")
        self.quality = self.LEVELS[level]
        self.level = level

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply JPEG compression to image.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W), dtype uint8.

        Returns:
            Compressed image as numpy array, dtype uint8.
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, encoded = cv2.imencode(".jpg", image, encode_params)
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    def __repr__(self) -> str:
        return f"JPEGCompression(level={self.level}, quality={self.quality})"
