"""
Unit tests for advanced binarization methods.
"""

import pytest
import numpy as np
import cv2

from binarization.methods.advanced_methods import (
	CLAHEThreshold,
	MultiScaleThreshold,
	GradientFusionThreshold,
	HybridThreshold
)


# Test fixtures
@pytest.fixture
def low_contrast_image():
	"""Create very low contrast image."""
	img = np.ones((200, 200), dtype=np.uint8) * 140
	# Very subtle text
	img[50:60, 50:150] = 120
	img[80:90, 50:150] = 120
	img[110:120, 50:150] = 120
	return img


@pytest.fixture
def mixed_text_sizes():
	"""Create image with different text sizes."""
	img = np.ones((200, 200), dtype=np.uint8) * 220
	# Small text
	img[30:35, 30:80] = 30
	# Medium text
	img[60:70, 30:100] = 30
	# Large text
	img[100:120, 30:150] = 30
	return img


@pytest.fixture
def thin_strokes():
	"""Create image with thin strokes."""
	img = np.ones((200, 200), dtype=np.uint8) * 200
	# Very thin lines (1-2 pixels)
	img[50, 50:150] = 30
	img[80:82, 50:150] = 30
	img[110, 50:150] = 30
	return img


@pytest.fixture
def uneven_illumination():
	"""Create image with gradient lighting."""
	img = np.zeros((200, 200), dtype=np.uint8)
	for i in range(200):
		for j in range(200):
			img[i, j] = int(50 + (i + j) * 100 / 400)
	# Add text
	img[50:60, 50:150] = np.maximum(img[50:60, 50:150] - 80, 0)
	img[80:90, 50:150] = np.maximum(img[80:90, 50:150] - 80, 0)
	return img


class TestCLAHEThreshold:
	"""Test suite for CLAHEThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = CLAHEThreshold()
		assert method.name == "clahe_threshold"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = CLAHEThreshold()
		params = method.get_default_params()
		assert 'clip_limit' in params
		assert 'tile_size' in params
		assert params['clip_limit'] == 2.0
	
	def test_binarize_low_contrast(self, low_contrast_image):
		"""Test on low contrast image where CLAHE should help."""
		method = CLAHEThreshold()
		result = method.binarize(low_contrast_image)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		
		# Check that contrast was enhanced
		assert 'enhanced_contrast' in result.metadata
		assert result.metadata['enhanced_contrast'] > 1.0  # Contrast increased
	
	def test_different_threshold_methods(self, low_contrast_image):
		"""Test with different threshold methods."""
		method = CLAHEThreshold()
		
		# Test with Otsu
		result_otsu = method.binarize(
			low_contrast_image,
			threshold_method='otsu'
		)
		assert result_otsu.binary_image.dtype == np.uint8
		
		# Test with Sauvola
		result_sauvola = method.binarize(
			low_contrast_image,
			threshold_method='sauvola',
			threshold_params={'window_size': 25, 'k': 0.2}
		)
		assert result_sauvola.binary_image.dtype == np.uint8
		
		# Test with Mean Adaptive
		result_mean = method.binarize(
			low_contrast_image,
			threshold_method='mean_adaptive',
			threshold_params={'window_size': 15, 'C': 5}
		)
		assert result_mean.binary_image.dtype == np.uint8
	
	def test_clip_limit_effect(self, low_contrast_image):
		"""Test effect of clip limit parameter."""
		method = CLAHEThreshold()
		
		result_low = method.binarize(low_contrast_image, clip_limit=1.0)
		result_high = method.binarize(low_contrast_image, clip_limit=4.0)
		
		# Different clip limits should give different results
		assert not np.array_equal(result_low.binary_image, result_high.binary_image)


class TestMultiScaleThreshold:
	"""Test suite for MultiScaleThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = MultiScaleThreshold()
		assert method.name == "multiscale"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = MultiScaleThreshold()
		params = method.get_default_params()
		assert 'scales' in params
		assert len(params['scales']) > 1
	
	def test_binarize_mixed_sizes(self, mixed_text_sizes):
		"""Test on image with mixed text sizes."""
		method = MultiScaleThreshold()
		result = method.binarize(mixed_text_sizes)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert 'num_scales' in result.metadata
	
	def test_different_scales(self, mixed_text_sizes):
		"""Test with different scale configurations."""
		method = MultiScaleThreshold()
		
		# Test with 2 scales
		result_2 = method.binarize(mixed_text_sizes, scales=[0.5, 1.0])
		assert result_2.metadata['num_scales'] == 2
		
		# Test with 4 scales
		result_4 = method.binarize(
			mixed_text_sizes,
			scales=[0.5, 0.75, 1.0, 1.5]
		)
		assert result_4.metadata['num_scales'] == 4
	
	def test_fusion_methods(self, mixed_text_sizes):
		"""Test different fusion methods."""
		method = MultiScaleThreshold()
		
		result_voting = method.binarize(mixed_text_sizes, fusion='voting')
		result_weighted = method.binarize(mixed_text_sizes, fusion='weighted')
		result_max = method.binarize(mixed_text_sizes, fusion='max')
		
		# All should produce valid binary images
		for result in [result_voting, result_weighted, result_max]:
			assert result.binary_image.dtype == np.uint8
			assert set(np.unique(result.binary_image)).issubset({0, 255})
		
		# Results should differ
		assert not np.array_equal(result_voting.binary_image, result_max.binary_image)


class TestGradientFusionThreshold:
	"""Test suite for GradientFusionThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = GradientFusionThreshold()
		assert method.name == "gradient_fusion"
	
	def test_binarize_thin_strokes(self, thin_strokes):
		"""Test on thin strokes where gradient should help."""
		method = GradientFusionThreshold()
		result = method.binarize(thin_strokes)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert 'mean_gradient' in result.metadata
	
	def test_gradient_methods(self, thin_strokes):
		"""Test different gradient computation methods."""
		method = GradientFusionThreshold()
		
		result_sobel = method.binarize(thin_strokes, gradient_method='sobel')
		result_scharr = method.binarize(thin_strokes, gradient_method='scharr')
		result_canny = method.binarize(thin_strokes, gradient_method='canny')
		
		# All should produce valid results
		for result in [result_sobel, result_scharr, result_canny]:
			assert result.binary_image.dtype == np.uint8
	
	def test_gradient_weight_effect(self, thin_strokes):
		"""Test effect of gradient weight."""
		method = GradientFusionThreshold()
		
		result_low = method.binarize(thin_strokes, gradient_weight=0.1)
		result_high = method.binarize(thin_strokes, gradient_weight=0.4)
		
		# Different weights should give different results
		assert not np.array_equal(result_low.binary_image, result_high.binary_image)
	
	def test_thin_stroke_preservation(self, thin_strokes):
		"""Test that thin strokes are better preserved."""
		method = GradientFusionThreshold()
		
		# Gradient fusion should detect thin lines
		result = method.binarize(thin_strokes, gradient_weight=0.3)
		
		# Check that thin line is detected
		# Line at row 50
		detected_pixels = np.sum(result.binary_image[50, 50:150] == 0)
		assert detected_pixels > 50  # Most of the line should be detected


class TestHybridThreshold:
	"""Test suite for HybridThreshold."""
	
	def test_initialization(self):
		"""Test algorithm initialization."""
		method = HybridThreshold()
		assert method.name == "hybrid"
	
	def test_default_params(self):
		"""Test default parameters."""
		method = HybridThreshold()
		params = method.get_default_params()
		assert 'global_method' in params
		assert 'adaptive_method' in params
		assert 'combination' in params
	
	def test_binarize_uneven_illumination(self, uneven_illumination):
		"""Test on uneven illumination."""
		method = HybridThreshold()
		result = method.binarize(uneven_illumination)
		
		assert result.binary_image.dtype == np.uint8
		assert set(np.unique(result.binary_image)).issubset({0, 255})
		assert 'global_threshold' in result.metadata
	
	def test_method_combinations(self, uneven_illumination):
		"""Test different method combinations."""
		method = HybridThreshold()
		
		# Otsu + Sauvola
		result1 = method.binarize(
			uneven_illumination,
			global_method='otsu',
			adaptive_method='sauvola'
		)
		
		# Triangle + Mean Adaptive
		result2 = method.binarize(
			uneven_illumination,
			global_method='triangle',
			adaptive_method='mean_adaptive'
		)
		
		# Both should produce valid results
		assert result1.binary_image.dtype == np.uint8
		assert result2.binary_image.dtype == np.uint8
	
	def test_combination_strategies(self, uneven_illumination):
		"""Test different combination strategies."""
		method = HybridThreshold()
		
		result_weighted = method.binarize(
			uneven_illumination,
			combination='weighted',
			adaptive_weight=0.7
		)
		
		result_and = method.binarize(
			uneven_illumination,
			combination='and'
		)
		
		result_or = method.binarize(
			uneven_illumination,
			combination='or'
		)
		
		result_variance = method.binarize(
			uneven_illumination,
			combination='variance'
		)
		
		# All should produce valid results
		for result in [result_weighted, result_and, result_or, result_variance]:
			assert result.binary_image.dtype == np.uint8
		
		# AND should be more conservative than OR
		foreground_and = np.sum(result_and.binary_image == 255)
		foreground_or = np.sum(result_or.binary_image == 255)
		assert foreground_or >= foreground_and
	
	def test_adaptive_weight_effect(self, uneven_illumination):
		"""Test effect of adaptive weight."""
		method = HybridThreshold()
		
		# More global
		result_global = method.binarize(
			uneven_illumination,
			combination='weighted',
			adaptive_weight=0.2
		)
		
		# More adaptive
		result_adaptive = method.binarize(
			uneven_illumination,
			combination='weighted',
			adaptive_weight=0.8
		)
		
		# Results should differ
		assert not np.array_equal(
			result_global.binary_image,
			result_adaptive.binary_image
		)


# Integration tests
class TestAdvancedMethodsIntegration:
	"""Integration tests for advanced methods."""
	
	def test_all_methods_on_challenging_image(self, low_contrast_image):
		"""Test all advanced methods on challenging image."""
		methods = [
			CLAHEThreshold(),
			MultiScaleThreshold(),
			GradientFusionThreshold(),
			HybridThreshold()
		]
		
		results = []
		for method in methods:
			result = method.binarize(low_contrast_image)
			results.append(result)
		
		# All should produce valid binary images
		for result in results:
			assert result.binary_image.dtype == np.uint8
			assert set(np.unique(result.binary_image)).issubset({0, 255})
	
	def test_performance_comparison(self, uneven_illumination):
		"""Compare processing times."""
		methods = {
			'clahe': CLAHEThreshold(),
			'multiscale': MultiScaleThreshold(),
			'gradient': GradientFusionThreshold(),
			'hybrid': HybridThreshold()
		}
		
		times = {}
		for name, method in methods.items():
			result = method.binarize(uneven_illumination)
			times[name] = result.processing_time
		
		# All should complete in reasonable time
		for name, time_taken in times.items():
			assert time_taken < 5.0, f"{name} took too long: {time_taken}s"
		
		print("\nProcessing times (200x200 image):")
		for name, time_taken in sorted(times.items(), key=lambda x: x[1]):
			print(f"  {name}: {time_taken:.4f}s")
	
	def test_quality_on_mixed_challenges(self, uneven_illumination):
		"""Test that advanced methods handle multiple challenges."""
		# Image with both low contrast and uneven illumination
		challenging = uneven_illumination.copy()
		# Reduce contrast further
		challenging = (challenging * 0.6 + 70).astype(np.uint8)
		
		clahe = CLAHEThreshold()
		hybrid = HybridThreshold()
		
		result_clahe = clahe.binarize(challenging)
		result_hybrid = hybrid.binarize(challenging)
		
		# Should still produce valid results
		assert result_clahe.binary_image.dtype == np.uint8
		assert result_hybrid.binary_image.dtype == np.uint8


# Parametrized tests
@pytest.mark.parametrize("method_class", [
	CLAHEThreshold,
	MultiScaleThreshold,
	GradientFusionThreshold,
	HybridThreshold
])
def test_all_methods_handle_small_image(method_class):
	"""Test that all methods handle small images."""
	method = method_class()
	small_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
	
	result = method.binarize(small_image)
	assert result.binary_image.shape == small_image.shape


@pytest.mark.parametrize("method_class", [
	CLAHEThreshold,
	MultiScaleThreshold,
	GradientFusionThreshold,
	HybridThreshold
])
def test_all_methods_rgb_conversion(method_class):
	"""Test RGB to grayscale conversion."""
	method = method_class()
	rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
	
	result = method.binarize(rgb_image)
	assert result.binary_image.ndim == 2


# Edge case tests
class TestEdgeCases:
	"""Test edge cases for advanced methods."""
	
	def test_clahe_on_uniform_image(self):
		"""Test CLAHE on uniform image."""
		uniform = np.ones((100, 100), dtype=np.uint8) * 128
		method = CLAHEThreshold()
		
		result = method.binarize(uniform)
		assert result.binary_image.dtype == np.uint8
	
	def test_multiscale_with_single_scale(self):
		"""Test multiscale with just one scale."""
		image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
		method = MultiScaleThreshold()
		
		result = method.binarize(image, scales=[1.0])
		assert result.binary_image.dtype == np.uint8
	
	def test_gradient_fusion_zero_weight(self):
		"""Test gradient fusion with zero gradient weight."""
		image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
		method = GradientFusionThreshold()
		
		result = method.binarize(image, gradient_weight=0.0)
		assert result.binary_image.dtype == np.uint8
	
	def test_hybrid_extreme_weights(self):
		"""Test hybrid with extreme weights."""
		image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
		method = HybridThreshold()
		
		# All global
		result_global = method.binarize(image, adaptive_weight=0.0)
		assert result_global.binary_image.dtype == np.uint8
		
		# All adaptive
		result_adaptive = method.binarize(image, adaptive_weight=1.0)
		assert result_adaptive.binary_image.dtype == np.uint8