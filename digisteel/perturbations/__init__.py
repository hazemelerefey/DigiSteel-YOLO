"""
Perturbation toolkit for steel defect detector robustness evaluation.

Provides standardized image degradations that simulate real-world industrial
camera conditions. Each perturbation type has multiple severity levels.

Perturbation types:
    - Gaussian Blur (defocus, lens softness)
    - Gaussian Noise (sensor noise, electrical interference)
    - Brightness Shift (lighting variation, over/underexposure)
    - JPEG Compression (transmission/storage artifacts)
    - Contrast Reduction (fog, dust, environmental interference)
    - Motion Blur (conveyor belt vibration, camera shake)

Usage:
    from digisteel.perturbations import PerturbationSuite

    suite = PerturbationSuite()
    degraded = suite.apply(image, perturbation="gaussian_blur", level=2)
"""

from digisteel.perturbations.blur import GaussianBlur, MotionBlur
from digisteel.perturbations.brightness import BrightnessShift, ContrastReduction
from digisteel.perturbations.jpeg import JPEGCompression
from digisteel.perturbations.noise import GaussianNoise
from digisteel.perturbations.suite import PerturbationSuite

__all__ = [
    "GaussianBlur",
    "MotionBlur",
    "GaussianNoise",
    "BrightnessShift",
    "ContrastReduction",
    "JPEGCompression",
    "PerturbationSuite",
]
