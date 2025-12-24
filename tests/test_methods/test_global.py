"""
Unit tests for global thresholding methods.
"""

import pytest
import numpy as np
import cv2

from binarization.methods.global_methods import (
	ManualThreshold,
	OtsuThreshold,
	TriangleThreshold,
	EntropyThreshold,
	MinimumErrorThreshold
)


# Test fixtures
@pytest.fixture
def bimodal_image():
	"""Create a synthetic bimodal image (clear foreground/background)."""
	img = np.zeros((200, 200), dtype=np.uint8)
	# Background: intensity around 50
	img[:] = np.random.normal(50, 10, (200, 200)).astype(np.uint8)
	# Foreground: intensity around 200
	img[50:150, 50:150] = np.random.normal(200, 10, (100, 100)).astype(np.uint8)
	return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def unimodal_image():
	"""Create a synthetic unimodal image (bright background, dark text)."""
	img = np.ones((200, 200), dtype=np.uint8) * 220
	# Add some dark text-like regions
	img[50:60, 50:150] = 30
	img[80:90, 50:150] = 30
	img[110:120, 50:150] = 30
	return img


@pytest.fixture
def gradient_image():
	"""Create an image with gradient illumination."""
	img = np.zeros((200, 200), dtype=np.uint8)
	for i in range(200):
		img[:, i] = int(i * 255 / 200)
	return img


@pytest.fixture
def flat_image():
	"""Create a uniform flat image (edge case)."""
	return np.ones((100, 100), dtype=np.uint8) * 128


class TestManualThreshold:
	"""Test suite for ManualThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = ManualThreshold()
		assert method.name == "manual"
		assert method.description is not None
	
	def test_default_params(self):
		"""Test default parameters."""
		method = ManualThreshold()
		params = method.get_default_params()
		assert 'threshold' in params
		assert params['threshold'] == 127
	
	def test_binarize_basic(self, bimodal_image):
		"""Test basic binarization."""
		method = ManualThreshold()
		result = method.binarize(bimodal_image, threshold=125)
		
		assert result.binary_image.shape == bimodal_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert result.threshold == 125
		assert result.method == "manual"
	
	def test_binarize_normalized(self, bimodal_image):
		"""Test binarization with normalized threshold."""
		method = ManualThreshold()
		result = method.binarize(bimodal_image, threshold=0.5, normalized=True)
		
		assert result.binary_image.dtype == np.uint8
		# Normalized 0.5 should be threshold 127 or 128
		assert 127 <= result.threshold <= 128
	
	def test_threshold_bounds(self, bimodal_image):
		"""Test threshold at boundaries."""
		method = ManualThreshold()
		
		# Threshold at 0 - everything should be white
		result = method.binarize(bimodal_image, threshold=0)
		assert np.all(result.binary_image == 255)
		
		# Threshold at 255 - everything should be black
		result = method.binarize(bimodal_image, threshold=255)
		assert np.all(result.binary_image == 0)
	
	def test_invalid_normalized_threshold(self, bimodal_image):
		"""Test error on invalid normalized threshold."""
		method = ManualThreshold()
		
		with pytest.raises(ValueError, match="Normalized threshold"):
			method.binarize(bimodal_image, threshold=1.5, normalized=True)


class TestOtsuThreshold:
	"""Test suite for OtsuThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = OtsuThreshold()
		assert method.name == "otsu"
	
	def test_binarize_bimodal(self, bimodal_image):
		"""Test on bimodal image - should find good separation."""
		method = OtsuThreshold()
		result = method.binarize(bimodal_image)
		
		assert result.binary_image.shape == bimodal_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		
		# Threshold should be roughly between the two peaks (50 and 200)
		assert 80 < result.threshold < 160
	
	def test_opencv_vs_custom(self, bimodal_image):
		"""Test that OpenCV and custom implementations give similar results."""
		method = OtsuThreshold()
		
		result_opencv = method.binarize(bimodal_image, use_opencv=True)
		result_custom = method.binarize(bimodal_image, use_opencv=False)
		
		# Thresholds should be very close (within a few intensity levels)
		assert abs(result_opencv.threshold - result_custom.threshold) < 5
	
	def test_bimodal_detection(self, bimodal_image, flat_image):
		"""Test bimodal histogram detection."""
		method = OtsuThreshold()
		
		result_bimodal = method.binarize(bimodal_image)
		assert result_bimodal.metadata['histogram_bimodal'] is True
		
		result_flat = method.binarize(flat_image)
		assert result_flat.metadata['histogram_bimodal'] is False
	
	def test_reproducibility(self, bimodal_image):
		"""Test that results are reproducible."""
		method = OtsuThreshold()
		
		result1 = method.binarize(bimodal_image)
		result2 = method.binarize(bimodal_image)
		
		assert result1.threshold == result2.threshold
		assert np.array_equal(result1.binary_image, result2.binary_image)
	
	def test_processing_time(self, bimodal_image):
		"""Test that processing time is recorded."""
		method = OtsuThreshold()
		result = method.binarize(bimodal_image)
		
		assert result.processing_time > 0
		assert result.processing_time < 1.0  # Should be fast


class TestTriangleThreshold:
	"""Test suite for TriangleThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = TriangleThreshold()
		assert method.name == "triangle"
	
	def test_binarize_unimodal(self, unimodal_image):
		"""Test on unimodal image."""
		method = TriangleThreshold()
		result = method.binarize(unimodal_image)
		
		assert result.binary_image.shape == unimodal_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		
		# Should find threshold that separates dark text from bright background
		# Most pixels are bright (220), dark text is 30
		# Threshold should be somewhere in between
		assert 50 < result.threshold < 200
	
	def test_on_bimodal(self, bimodal_image):
		"""Test that triangle works on bimodal too."""
		method = TriangleThreshold()
		result = method.binarize(bimodal_image)
		
		# Should still produce valid binary image
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_flat_image(self, flat_image):
		"""Test behavior on flat image (edge case)."""
		method = TriangleThreshold()
		result = method.binarize(flat_image)
		
		# Should not crash and produce valid output
		assert result.binary_image.dtype == np.uint8
		assert result.binary_image.shape == flat_image.shape


class TestEntropyThreshold:
	"""Test suite for EntropyThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = EntropyThreshold()
		assert method.name == "entropy"
	
	def test_binarize_basic(self, bimodal_image):
		"""Test basic binarization."""
		method = EntropyThreshold()
		result = method.binarize(bimodal_image)
		
		assert result.binary_image.shape == bimodal_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert 0 <= result.threshold <= 255
	
	def test_on_complex_histogram(self, bimodal_image):
		"""Test on image with complex intensity distribution."""
		# Create more complex multi-modal image
		img = np.zeros((200, 200), dtype=np.uint8)
		img[:100, :100] = 50  # Dark region
		img[:100, 100:] = 120  # Medium region
		img[100:, :100] = 180  # Bright region
		img[100:, 100:] = 250  # Very bright region
		
		method = EntropyThreshold()
		result = method.binarize(img)
		
		# Should still produce valid binary output
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_reproducibility(self, bimodal_image):
		"""Test reproducibility."""
		method = EntropyThreshold()
		
		result1 = method.binarize(bimodal_image)
		result2 = method.binarize(bimodal_image)
		
		assert result1.threshold == result2.threshold


class TestMinimumErrorThreshold:
	"""Test suite for MinimumErrorThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = MinimumErrorThreshold()
		assert method.name == "minimum_error"
	
	def test_binarize_basic(self, bimodal_image):
		"""Test basic binarization."""
		method = MinimumErrorThreshold()
		result = method.binarize(bimodal_image)
		
		assert result.binary_image.shape == bimodal_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert 0 <= result.threshold <= 255
	
	def test_gaussian_assumption(self, bimodal_image):
		"""Test on image that fits Gaussian mixture model."""
		# Create image with clear Gaussian distributions
		img = np.zeros((200, 200), dtype=np.uint8)
		img[:] = np.random.normal(60, 15, (200, 200))
		img[50:150, 50:150] = np.random.normal(180, 15, (100, 100))
		img = np.clip(img, 0, 255).astype(np.uint8)
		
		method = MinimumErrorThreshold()
		result = method.binarize(img)
		
		# Should find threshold between the two Gaussian peaks
		assert 90 < result.threshold < 150
	
	def test_comparison_with_otsu(self, bimodal_image):
		"""Compare with Otsu on bimodal image."""
		otsu = OtsuThreshold()
		min_error = MinimumErrorThreshold()
		
		result_otsu = otsu.binarize(bimodal_image)
		result_min_error = min_error.binarize(bimodal_image)
		
		# Results should be similar for bimodal image
		# (both assume separable classes)
		assert abs(result_otsu.threshold - result_min_error.threshold) < 30


# Integration tests
class TestGlobalMethodsIntegration:
	"""Integration tests comparing all global methods."""
	
	def test_all_methods_on_same_image(self, bimodal_image):
		"""Test all methods on the same image."""
		methods = [
			ManualThreshold(),
			OtsuThreshold(),
			TriangleThreshold(),
			EntropyThreshold(),
			MinimumErrorThreshold()
		]
		
		results = []
		for method in methods:
			if isinstance(method, ManualThreshold):
				result = method.binarize(bimodal_image, threshold=127)
			else:
				result = method.binarize(bimodal_image)
			results.append(result)
		
		# All should produce valid binary images
		for result in results:
			assert result.binary_image.dtype == np.uint8
			assert result.binary_image.shape == bimodal_image.shape
			assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_threshold_ordering(self, bimodal_image):
		"""Test that automatic methods give reasonable thresholds."""
		otsu = OtsuThreshold()
		triangle = TriangleThreshold()
		entropy = EntropyThreshold()
		min_error = MinimumErrorThreshold()
		
		results = {
			'otsu': otsu.binarize(bimodal_image),
			'triangle': triangle.binarize(bimodal_image),
			'entropy': entropy.binarize(bimodal_image),
			'min_error': min_error.binarize(bimodal_image)
		}
		
		# All thresholds should be in valid range
		for name, result in results.items():
			assert 0 <= result.threshold <= 255, f"{name} threshold out of range"
	
	def test_performance_comparison(self, bimodal_image):
		"""Compare processing times."""
		methods = [
			OtsuThreshold(),
			TriangleThreshold(),
			EntropyThreshold(),
			MinimumErrorThreshold()
		]
		
		times = {}
		for method in methods:
			result = method.binarize(bimodal_image)
			times[method.name] = result.processing_time
		
		# All should complete in reasonable time
		for name, time_taken in times.items():
			assert time_taken < 1.0, f"{name} took too long: {time_taken}s"


# Parametrized tests
@pytest.mark.parametrize("method_class", [
	ManualThreshold,
	OtsuThreshold,
	TriangleThreshold,
	EntropyThreshold,
	MinimumErrorThreshold
])
def test_rgb_to_grayscale_conversion(method_class):
	"""Test that all methods handle RGB input correctly."""
	# Create RGB image
	rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
	
	method = method_class()
	
	if isinstance(method, ManualThreshold):
		result = method.binarize(rgb_image, threshold=127)
	else:
		result = method.binarize(rgb_image)
	
	# Should convert to grayscale and produce binary
	assert result.binary_image.ndim == 2
	assert result.binary_image.dtype == np.uint8


@pytest.mark.parametrize("method_class", [
	OtsuThreshold,
	TriangleThreshold,
	EntropyThreshold,
	MinimumErrorThreshold
])
def test_empty_image_handling(method_class):
	"""Test handling of edge cases."""
	method = method_class()
	
	# Very small image
	small_image = np.array([[100, 200], [50, 150]], dtype=np.uint8)
	result = method.binarize(small_image)
	assert result.binary_image.shape == small_image.shape