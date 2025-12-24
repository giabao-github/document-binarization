"""
Unit tests for adaptive thresholding methods.
"""

import pytest
import numpy as np
import cv2

from binarization.methods.adaptive_methods import (
	MeanAdaptiveThreshold,
	GaussianAdaptiveThreshold,
	NiblackThreshold,
	SauvolaThreshold,
	WolfThreshold,
	BradleyThreshold
)


# Test fixtures
@pytest.fixture
def gradient_image():
	"""Create image with gradient illumination (perfect for adaptive methods)."""
	img = np.zeros((200, 200), dtype=np.uint8)
	
	# Create gradient background
	for i in range(200):
		for j in range(200):
			img[i, j] = int(50 + (i + j) * 100 / 400)
	
	# Add dark text regions that should be detected despite gradient
	img[50:60, 50:150] = np.maximum(img[50:60, 50:150] - 80, 0)
	img[80:90, 50:150] = np.maximum(img[80:90, 50:150] - 80, 0)
	img[110:120, 50:150] = np.maximum(img[110:120, 50:150] - 80, 0)
	
	return img


@pytest.fixture
def uneven_illumination():
	"""Create image with vignetting/uneven illumination."""
	img = np.ones((200, 200), dtype=np.uint8) * 200
	
	# Create vignetting effect
	center_y, center_x = 100, 100
	for i in range(200):
		for j in range(200):
			dist = np.sqrt((i - center_y)**2 + (j - center_x)**2) / 140.0
			img[i, j] = int(200 * (1.0 - 0.5 * dist))
	
	# Add text
	img[50:60, 50:150] = 30
	img[80:90, 50:150] = 30
	img[110:120, 50:150] = 30
	
	return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def low_contrast_image():
	"""Create image with very low contrast."""
	img = np.ones((200, 200), dtype=np.uint8) * 150
	
	# Very subtle text (low contrast)
	img[50:60, 50:150] = 130
	img[80:90, 50:150] = 130
	img[110:120, 50:150] = 130
	
	return img


class TestMeanAdaptiveThreshold:
	"""Test suite for MeanAdaptiveThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = MeanAdaptiveThreshold()
		assert method.name == "mean_adaptive"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = MeanAdaptiveThreshold()
		params = method.get_default_params()
		assert 'window_size' in params
		assert 'C' in params
		assert params['window_size'] == 15
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization on gradient image."""
		method = MeanAdaptiveThreshold()
		result = method.binarize(gradient_image)
		
		assert result.binary_image.shape == gradient_image.shape
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_window_size_effect(self, gradient_image):
		"""Test effect of different window sizes."""
		method = MeanAdaptiveThreshold()
		
		result_small = method.binarize(gradient_image, window_size=5)
		result_large = method.binarize(gradient_image, window_size=25)
		
		# Both should produce valid binary images
		assert result_small.binary_image.dtype == np.uint8
		assert result_large.binary_image.dtype == np.uint8
		
		# Results should differ
		assert not np.array_equal(result_small.binary_image, result_large.binary_image)
	
	def test_opencv_vs_custom(self, gradient_image):
		"""Test OpenCV vs custom implementation."""
		method = MeanAdaptiveThreshold()
		
		result_opencv = method.binarize(gradient_image, use_opencv=True)
		result_custom = method.binarize(gradient_image, use_opencv=False)
		
		# Results should be very similar
		diff = np.sum(result_opencv.binary_image != result_custom.binary_image)
		total = gradient_image.size
		assert diff / total < 0.05  # Less than 5% difference


class TestGaussianAdaptiveThreshold:
	"""Test suite for GaussianAdaptiveThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = GaussianAdaptiveThreshold()
		assert method.name == "gaussian_adaptive"
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization."""
		method = GaussianAdaptiveThreshold()
		result = method.binarize(gradient_image)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_comparison_with_mean(self, gradient_image):
		"""Compare with mean adaptive."""
		mean_method = MeanAdaptiveThreshold()
		gaussian_method = GaussianAdaptiveThreshold()
		
		result_mean = mean_method.binarize(gradient_image, window_size=15, C=5)
		result_gaussian = gaussian_method.binarize(gradient_image, window_size=15, C=5)
		
		# Results should be similar but not identical
		similarity = np.sum(result_mean.binary_image == result_gaussian.binary_image) / gradient_image.size
		assert 0.8 < similarity < 1.0  # 80-100% similar


class TestNiblackThreshold:
	"""Test suite for NiblackThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = NiblackThreshold()
		assert method.name == "niblack"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = NiblackThreshold()
		params = method.get_default_params()
		assert 'k' in params
		assert params['k'] == -0.2
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization."""
		method = NiblackThreshold()
		result = method.binarize(gradient_image)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert result.metadata['k'] == -0.2
	
	def test_k_parameter_effect(self, gradient_image):
		"""Test effect of k parameter."""
		method = NiblackThreshold()
		
		result_low_k = method.binarize(gradient_image, k=-0.5)
		result_high_k = method.binarize(gradient_image, k=-0.1)
		
		# Different k values should give different results
		assert not np.array_equal(result_low_k.binary_image, result_high_k.binary_image)
		
		# Lower k (more negative) should generally produce darker result
		foreground_low = np.sum(result_low_k.binary_image == 255)
		foreground_high = np.sum(result_high_k.binary_image == 255)
		assert foreground_low < foreground_high
	
	def test_metadata(self, gradient_image):
		"""Test that metadata is properly recorded."""
		method = NiblackThreshold()
		result = method.binarize(gradient_image, window_size=15, k=-0.2)
		
		assert 'mean_threshold' in result.metadata
		assert 'std_threshold' in result.metadata
		assert result.metadata['window_size'] == 15


class TestSauvolaThreshold:
	"""Test suite for SauvolaThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = SauvolaThreshold()
		assert method.name == "sauvola"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = SauvolaThreshold()
		params = method.get_default_params()
		assert 'k' in params
		assert 'R' in params
		assert params['k'] == 0.2
		assert params['R'] == 128.0
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization."""
		method = SauvolaThreshold()
		result = method.binarize(gradient_image)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_on_low_contrast(self, low_contrast_image):
		"""Test on low contrast image."""
		method = SauvolaThreshold()
		result = method.binarize(low_contrast_image, window_size=25, k=0.5)
		
		# Should detect some foreground even in low contrast
		foreground_pixels = np.sum(result.binary_image == 0)  # Dark text
		assert foreground_pixels > 100  # Should detect some text
	
	def test_R_parameter_effect(self, gradient_image):
		"""Test effect of R parameter."""
		method = SauvolaThreshold()
		
		result_low_R = method.binarize(gradient_image, R=64.0)
		result_high_R = method.binarize(gradient_image, R=200.0)
		
		# Different R values should affect sensitivity
		assert not np.array_equal(result_low_R.binary_image, result_high_R.binary_image)
	
	def test_comparison_with_niblack(self, gradient_image):
		"""Compare Sauvola with Niblack."""
		niblack = NiblackThreshold()
		sauvola = SauvolaThreshold()
		
		result_niblack = niblack.binarize(gradient_image, window_size=25, k=-0.2)
		result_sauvola = sauvola.binarize(gradient_image, window_size=25, k=0.2)
		
		# Results should differ (Sauvola improves on Niblack)
		similarity = np.sum(result_niblack.binary_image == result_sauvola.binary_image) / gradient_image.size
		assert similarity < 0.95  # Should have noticeable differences


class TestWolfThreshold:
	"""Test suite for WolfThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = WolfThreshold()
		assert method.name == "wolf"
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization."""
		method = WolfThreshold()
		result = method.binarize(gradient_image)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_on_degraded_image(self, low_contrast_image):
		"""Test on degraded/low-contrast image."""
		method = WolfThreshold()
		result = method.binarize(low_contrast_image, window_size=25, k=0.5)
		
		# Wolf should handle low contrast better
		assert result.binary_image.dtype == np.uint8
		assert 'min_intensity' in result.metadata
		assert 'max_std' in result.metadata
	
	def test_metadata(self, gradient_image):
		"""Test metadata recording."""
		method = WolfThreshold()
		result = method.binarize(gradient_image)
		
		assert 'min_intensity' in result.metadata
		assert 'max_std' in result.metadata
		assert 'mean_threshold' in result.metadata


class TestBradleyThreshold:
	"""Test suite for BradleyThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = BradleyThreshold()
		assert method.name == "bradley"
	
	def test_binarize_basic(self, gradient_image):
		"""Test basic binarization."""
		method = BradleyThreshold()
		result = method.binarize(gradient_image, window_size=25, t=15)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_integral_image_usage(self, gradient_image):
		"""Test that integral image optimization is used."""
		method = BradleyThreshold()
		result = method.binarize(gradient_image)
		
		assert result.metadata['uses_integral_image'] is True
	
	def test_t_parameter_effect(self, gradient_image):
		"""Test effect of t parameter."""
		method = BradleyThreshold()
		
		result_low_t = method.binarize(gradient_image, t=5)
		result_high_t = method.binarize(gradient_image, t=25)
		
		# Different t values should give different results
		assert not np.array_equal(result_low_t.binary_image, result_high_t.binary_image)


# Integration tests
class TestAdaptiveMethodsIntegration:
	"""Integration tests comparing all adaptive methods."""
	
	def test_all_methods_on_gradient(self, gradient_image):
		"""Test all methods on gradient illumination."""
		methods = [
			MeanAdaptiveThreshold(),
			GaussianAdaptiveThreshold(),
			NiblackThreshold(),
			SauvolaThreshold(),
			WolfThreshold(),
			BradleyThreshold()
		]
		
		results = []
		for method in methods:
			result = method.binarize(gradient_image, window_size=25)
			results.append(result)
		
		# All should produce valid binary images
		for result in results:
			assert result.binary_image.dtype == np.uint8
			assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_uneven_illumination_robustness(self, uneven_illumination):
		"""Test robustness to uneven illumination."""
		# Adaptive methods should handle this better than global
		sauvola = SauvolaThreshold()
		result = sauvola.binarize(uneven_illumination, window_size=25)
		
		# Should detect text despite vignetting
		assert result.binary_image.dtype == np.uint8
		
		# Check that text regions are detected
		text_region = result.binary_image[50:60, 50:150]
		# Most of text region should be foreground (0 for dark text)
		foreground_ratio = np.sum(text_region == 0) / text_region.size
		assert foreground_ratio > 0.5
	
	def test_performance_comparison(self, gradient_image):
		"""Compare processing times of different methods."""
		methods = [
			('mean', MeanAdaptiveThreshold()),
			('gaussian', GaussianAdaptiveThreshold()),
			('niblack', NiblackThreshold()),
			('sauvola', SauvolaThreshold()),
			('wolf', WolfThreshold()),
			('bradley', BradleyThreshold())
		]
		
		times = {}
		for name, method in methods:
			result = method.binarize(gradient_image, window_size=25)
			times[name] = result.processing_time
		
		# All should complete in reasonable time
		for name, time_taken in times.items():
			assert time_taken < 2.0, f"{name} took too long: {time_taken}s"
		
		print("\nProcessing times (200x200 image):")
		for name, time_taken in sorted(times.items(), key=lambda x: x[1]):
			print(f"  {name}: {time_taken:.4f}s")


# Parametrized tests
@pytest.mark.parametrize("method_class", [
	MeanAdaptiveThreshold,
	GaussianAdaptiveThreshold,
	NiblackThreshold,
	SauvolaThreshold,
	WolfThreshold,
	BradleyThreshold
])
def test_all_methods_handle_small_image(method_class):
	"""Test that all methods handle small images."""
	method = method_class()
	small_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
	
	result = method.binarize(small_image, window_size=7)
	assert result.binary_image.shape == small_image.shape


@pytest.mark.parametrize("method_class", [
	MeanAdaptiveThreshold,
	GaussianAdaptiveThreshold,
	NiblackThreshold,
	SauvolaThreshold,
	WolfThreshold,
	BradleyThreshold
])
def test_all_methods_rgb_conversion(method_class):
	"""Test RGB to grayscale conversion."""
	method = method_class()
	rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
	
	result = method.binarize(rgb_image, window_size=15)
	assert result.binary_image.ndim == 2


# Edge case tests
class TestEdgeCases:
	"""Test edge cases for adaptive methods."""
	
	def test_uniform_image(self):
		"""Test on completely uniform image."""
		uniform = np.ones((100, 100), dtype=np.uint8) * 128
		
		method = SauvolaThreshold()
		result = method.binarize(uniform, window_size=15)
		
		# Should not crash
		assert result.binary_image.dtype == np.uint8
	
	def test_very_small_window(self, gradient_image):
		"""Test with very small window size."""
		method = SauvolaThreshold()
		result = method.binarize(gradient_image, window_size=3)
		
		assert result.binary_image.dtype == np.uint8
	
	def test_large_window(self, gradient_image):
		"""Test with large window size."""
		method = SauvolaThreshold()
		result = method.binarize(gradient_image, window_size=51)
		
		assert result.binary_image.dtype == np.uint8