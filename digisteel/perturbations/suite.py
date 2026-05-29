"""
Perturbation Suite: Unified interface for all perturbation types.

Provides a single entry point for applying any perturbation at any severity
level, plus utilities for batch processing and metadata tracking.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from digisteel.perturbations.blur import GaussianBlur, MotionBlur
from digisteel.perturbations.brightness import BrightnessShift, ContrastReduction
from digisteel.perturbations.jpeg import JPEGCompression
from digisteel.perturbations.noise import GaussianNoise


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation evaluation point."""

    name: str
    level: int
    params: Dict[str, float]

    def __repr__(self) -> str:
        return f"{self.name}_L{self.level}"


# Registry of all perturbation types and their classes
PERTURBATION_REGISTRY: Dict[str, type] = {
    "gaussian_blur": GaussianBlur,
    "motion_blur": MotionBlur,
    "gaussian_noise": GaussianNoise,
    "brightness_shift": BrightnessShift,
    "contrast_reduction": ContrastReduction,
    "jpeg_compression": JPEGCompression,
}

# Default severity levels (1-4)
DEFAULT_LEVELS = [1, 2, 3, 4]


class PerturbationSuite:
    """
    Unified interface for applying image perturbations.

    Supports 6 perturbation types, each with 4 severity levels,
    for a total of 24 evaluation points.

    Example:
        >>> suite = PerturbationSuite()
        >>> degraded = suite.apply(image, "gaussian_blur", level=2)
        >>> configs = suite.all_configs()  # 24 configs
    """

    def __init__(self, levels: List[int] = None):
        """
        Initialize the perturbation suite.

        Args:
            levels: List of severity levels to use. Default: [1, 2, 3, 4].
        """
        self.levels = levels or DEFAULT_LEVELS

    @property
    def perturbation_names(self) -> List[str]:
        """List of available perturbation names."""
        return list(PERTURBATION_REGISTRY.keys())

    def all_configs(self) -> List[PerturbationConfig]:
        """
        Generate all perturbation configurations.

        Returns:
            List of PerturbationConfig objects (6 types x 4 levels = 24 configs).
        """
        configs = []
        for name in self.perturbation_names:
            for level in self.levels:
                cls = PERTURBATION_REGISTRY[name]
                params = self._get_params(cls, level)
                configs.append(PerturbationConfig(name=name, level=level, params=params))
        return configs

    def apply(
        self, image: np.ndarray, perturbation: str, level: int
    ) -> np.ndarray:
        """
        Apply a perturbation to an image.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W), dtype uint8.
            perturbation: Perturbation name (e.g., "gaussian_blur").
            level: Severity level (1-4).

        Returns:
            Degraded image as numpy array, dtype uint8.

        Raises:
            ValueError: If perturbation name is unknown or level is invalid.
        """
        if perturbation not in PERTURBATION_REGISTRY:
            raise ValueError(
                f"Unknown perturbation: {perturbation}. "
                f"Available: {list(PERTURBATION_REGISTRY.keys())}"
            )

        cls = PERTURBATION_REGISTRY[perturbation]
        perturber = cls(level)
        return perturber.apply(image)

    def apply_all(
        self, image: np.ndarray
    ) -> List[Tuple[PerturbationConfig, np.ndarray]]:
        """
        Apply all perturbations at all levels to an image.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W), dtype uint8.

        Returns:
            List of (config, degraded_image) tuples.
        """
        results = []
        for config in self.all_configs():
            degraded = self.apply(image, config.name, config.level)
            results.append((config, degraded))
        return results

    @staticmethod
    def _get_params(cls: type, level: int) -> Dict[str, float]:
        """Extract perturbation parameters for a given level."""
        if hasattr(cls, "LEVELS"):
            level_map = cls.LEVELS
            if level in level_map:
                # Return the main parameter
                key = [k for k in level_map.keys() if k == level][0]
                value = level_map[key]
                # Determine parameter name from class
                if cls == GaussianBlur:
                    return {"sigma": value}
                elif cls == MotionBlur:
                    return {"kernel_size": value}
                elif cls == GaussianNoise:
                    return {"sigma": value}
                elif cls == BrightnessShift:
                    return {"delta": value}
                elif cls == ContrastReduction:
                    return {"factor": value}
                elif cls == JPEGCompression:
                    return {"quality": value}
        return {}

    def summary(self) -> str:
        """Return a human-readable summary of the perturbation matrix."""
        lines = ["Perturbation Suite Summary", "=" * 40]
        for config in self.all_configs():
            lines.append(f"  {config}")
        lines.append(f"\nTotal evaluation points: {len(self.all_configs())}")
        return "\n".join(lines)
