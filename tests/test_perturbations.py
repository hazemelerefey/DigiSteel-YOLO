"""Unit tests for the perturbation toolkit."""
import numpy as np
import pytest

from digisteel.perturbations import (
    BrightnessShift,
    ContrastReduction,
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    MotionBlur,
    PerturbationSuite,
)


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_color_image():
    """Create a sample color image for testing."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


class TestGaussianBlur:
    def test_output_shape(self, sample_image):
        blur = GaussianBlur(level=1)
        result = blur.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        blur = GaussianBlur(level=2)
        result = blur.apply(sample_image)
        assert result.dtype == np.uint8

    def test_all_levels(self, sample_image):
        for level in [1, 2, 3, 4]:
            blur = GaussianBlur(level=level)
            result = blur.apply(sample_image)
            assert result.shape == sample_image.shape

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            GaussianBlur(level=5)


class TestMotionBlur:
    def test_output_shape(self, sample_image):
        blur = MotionBlur(level=1)
        result = blur.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_all_levels(self, sample_image):
        for level in [1, 2, 3, 4]:
            blur = MotionBlur(level=level)
            result = blur.apply(sample_image)
            assert result.shape == sample_image.shape


class TestGaussianNoise:
    def test_output_shape(self, sample_image):
        noise = GaussianNoise(level=1, seed=42)
        result = noise.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        noise = GaussianNoise(level=2, seed=42)
        result = noise.apply(sample_image)
        assert result.dtype == np.uint8

    def test_reproducibility(self, sample_image):
        """Same seed should produce same noise."""
        noise1 = GaussianNoise(level=2, seed=42)
        noise2 = GaussianNoise(level=2, seed=42)
        result1 = noise1.apply(sample_image)
        result2 = noise2.apply(sample_image)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds(self, sample_image):
        """Different seeds should produce different noise."""
        noise1 = GaussianNoise(level=2, seed=42)
        noise2 = GaussianNoise(level=2, seed=123)
        result1 = noise1.apply(sample_image)
        result2 = noise2.apply(sample_image)
        assert not np.array_equal(result1, result2)

    def test_severity_increases(self, sample_image):
        """Higher levels should produce more noise."""
        noise1 = GaussianNoise(level=1, seed=42)
        noise4 = GaussianNoise(level=4, seed=42)
        result1 = noise1.apply(sample_image.astype(np.float64))
        result4 = noise4.apply(sample_image.astype(np.float64))
        diff1 = np.mean(np.abs(result1.astype(float) - sample_image.astype(float)))
        diff4 = np.mean(np.abs(result4.astype(float) - sample_image.astype(float)))
        assert diff4 > diff1


class TestBrightnessShift:
    def test_output_shape(self, sample_image):
        shift = BrightnessShift(level=1)
        result = shift.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_darkening(self, sample_image):
        """Levels 1-2 should darken the image."""
        shift = BrightnessShift(level=1)
        result = shift.apply(sample_image)
        assert np.mean(result) <= np.mean(sample_image) + 1  # Allow small rounding

    def test_brightening(self, sample_image):
        """Levels 3-4 should brighten the image."""
        shift = BrightnessShift(level=3)
        result = shift.apply(sample_image)
        assert np.mean(result) >= np.mean(sample_image) - 1


class TestContrastReduction:
    def test_output_shape(self, sample_image):
        reduction = ContrastReduction(level=1)
        result = reduction.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_contrast_reduces(self, sample_image):
        """Higher levels should reduce contrast (lower std)."""
        reduction = ContrastReduction(level=3)
        result = reduction.apply(sample_image)
        assert np.std(result) <= np.std(sample_image) + 1


class TestJPEGCompression:
    def test_output_shape(self, sample_image):
        jpeg = JPEGCompression(level=1)
        result = jpeg.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_all_levels(self, sample_image):
        for level in [1, 2, 3, 4]:
            jpeg = JPEGCompression(level=level)
            result = jpeg.apply(sample_image)
            assert result.shape == sample_image.shape


class TestPerturbationSuite:
    def test_all_configs_count(self):
        suite = PerturbationSuite()
        configs = suite.all_configs()
        assert len(configs) == 24  # 6 types x 4 levels

    def test_custom_levels(self):
        suite = PerturbationSuite(levels=[1, 3])
        configs = suite.all_configs()
        assert len(configs) == 12  # 6 types x 2 levels

    def test_perturbation_names(self):
        suite = PerturbationSuite()
        names = suite.perturbation_names
        assert len(names) == 6
        assert "gaussian_blur" in names
        assert "gaussian_noise" in names

    def test_apply(self, sample_image):
        suite = PerturbationSuite()
        result = suite.apply(sample_image, "gaussian_blur", level=1)
        assert result.shape == sample_image.shape

    def test_apply_invalid_perturbation(self, sample_image):
        suite = PerturbationSuite()
        with pytest.raises(ValueError):
            suite.apply(sample_image, "invalid_perturbation", level=1)

    def test_apply_all(self, sample_image):
        suite = PerturbationSuite(levels=[1, 2])
        results = suite.apply_all(sample_image)
        assert len(results) == 12  # 6 types x 2 levels

    def test_summary(self):
        suite = PerturbationSuite(levels=[1, 2])
        summary = suite.summary()
        assert "Perturbation Suite Summary" in summary
        assert "Total evaluation points: 12" in summary
