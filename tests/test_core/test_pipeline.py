"""
Unit tests for the core pipeline functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from binarization.core.pipeline import BinarizationPipeline, run_pipeline
from binarization.core.config import PipelineConfig, get_default_config
from binarization.core.base import (
	BinarizationAlgorithm, 
	BinarizationResult,
	ensure_binary
)


# Test fixtures
@pytest.fixture
def simple_image():
	"""Create a simple test image."""
	# Create 100x100 image with text-like pattern
	img = np.ones((100, 100), dtype=np.uint8) * 200
	# Add dark "text" regions
	img[20:30, 20:80] = 50
	img[40:50, 20:80] = 50
	img[60:70, 20:80] = 50
	return img


@pytest.fixture
def rgb_image():
	"""Create an RGB test image."""
	img = np.ones((100, 100, 3), dtype=np.uint8) * 200
	img[20:30, 20:80] = [50, 50, 50]
	img[40:50, 20:80] = [50, 50, 50]
	return img


@pytest.fixture
def mock_algorithm():
	"""Create a mock binarization algorithm."""
	algorithm = Mock(spec=BinarizationAlgorithm)
	algorithm.name = "mock_method"
	
	def mock_binarize(image, **params):
		# Simple thresholding
		binary = np.where(image < 128, 255, 0).astype(np.uint8)
		return BinarizationResult(
			binary_image=binary,
			method="mock_method",
			parameters=params,
			threshold=128.0
		)
	
	algorithm.binarize = mock_binarize
	return algorithm


@pytest.fixture
def pipeline_with_mock(mock_algorithm):
	"""Create pipeline with mock algorithm."""
	config = get_default_config()
	config.method = "mock_method"
	config.post_processing.enabled = False
	
	pipeline = BinarizationPipeline(config)
	pipeline.register_algorithm("mock_method", mock_algorithm)
	return pipeline


class TestBinarizationPipeline:
	"""Test suite for BinarizationPipeline class."""
	
	def test_pipeline_initialization(self):
		"""Test pipeline initialization."""
		config = get_default_config()
		pipeline = BinarizationPipeline(config)
		
		assert pipeline.config is not None
		assert pipeline.config.method == "otsu"
		assert isinstance(pipeline.algorithm_registry, dict)
	
	def test_register_algorithm(self, mock_algorithm):
		"""Test algorithm registration."""
		pipeline = BinarizationPipeline()
		pipeline.register_algorithm("test_method", mock_algorithm)
		
		assert "test_method" in pipeline.algorithm_registry
		assert pipeline.algorithm_registry["test_method"] == mock_algorithm

	def test_get_algorithm(self, mock_algorithm):
		"""Test getting registered algorithm."""
		pipeline = BinarizationPipeline()
		pipeline.register_algorithm("test_method", mock_algorithm)
		
		retrieved = pipeline.get_algorithm("test_method")
		assert retrieved == mock_algorithm
	
	def test_get_nonexistent_algorithm(self):
		"""Test getting non-existent algorithm raises error."""
		pipeline = BinarizationPipeline()
		
		with pytest.raises(ValueError, match="Algorithm .* not found"):
			pipeline.get_algorithm("nonexistent")
	
	def test_run_basic(self, simple_image, pipeline_with_mock):
		"""Test basic pipeline run."""
		result = pipeline_with_mock.run(simple_image)
		
		assert 'binary_image' in result
		assert 'method' in result
		assert 'processing_time' in result
		assert result['method'] == 'mock_method'
		assert result['binary_image'].dtype == np.uint8
		assert set(np.unique(result['binary_image'])).issubset({0, 255})
	
	def test_run_with_intermediates(self, simple_image, pipeline_with_mock):
		"""Test pipeline run with intermediate results."""
		result = pipeline_with_mock.run(simple_image, return_intermediates=True)
		
		assert 'intermediates' in result
		assert 'preprocessed' in result['intermediates']
		assert 'binarized' in result['intermediates']
	
	def test_run_with_rgb_image(self, rgb_image, pipeline_with_mock):
		"""Test pipeline with RGB image input."""
		result = pipeline_with_mock.run(rgb_image)
		
		assert result['binary_image'].ndim == 2  # Should be grayscale
		assert result['binary_image'].dtype == np.uint8
	
	def test_run_batch(self, simple_image, pipeline_with_mock):
		"""Test batch processing."""
		images = [simple_image, simple_image.copy(), simple_image.copy()]
		results = pipeline_with_mock.run_batch(images)
		
		assert len(results) == 3
		for result in results:
			assert 'binary_image' in result
			assert result['binary_image'].shape == simple_image.shape

	def test_preprocessing_grayscale(self, simple_image, pipeline_with_mock):
		"""Test preprocessing of grayscale image."""
		processed = pipeline_with_mock._preprocess_image(simple_image)
		
		assert processed.dtype == np.uint8
		assert processed.ndim == 2
		assert processed.shape == simple_image.shape
	
	def test_preprocessing_rgb(self, rgb_image, pipeline_with_mock):
		"""Test preprocessing of RGB image."""
		processed = pipeline_with_mock._preprocess_image(rgb_image)
		
		assert processed.dtype == np.uint8
		assert processed.ndim == 2
		assert processed.shape == rgb_image.shape[:2]
	
	def test_preprocessing_float_image(self, pipeline_with_mock):
		"""Test preprocessing of float image."""
		float_img = np.random.rand(100, 100).astype(np.float32)
		processed = pipeline_with_mock._preprocess_image(float_img)
		
		assert processed.dtype == np.uint8
		assert 0 <= processed.min() <= processed.max() <= 255
	
	def test_post_processing_morphology(self, simple_image):
		"""Test post-processing with morphology."""
		config = get_default_config()
		config.post_processing.enabled = True
		config.post_processing.morphology = {
			'opening': {'enabled': True, 'kernel_size': 3},
			'closing': {'enabled': True, 'kernel_size': 3}
		}
		
		pipeline = BinarizationPipeline(config)
		
		# Create a binary image with noise
		binary = np.zeros((100, 100), dtype=np.uint8)
		binary[20:30, 20:80] = 255
		binary[5, 5] = 255  # Small noise
		
		result = pipeline._apply_post_processing(binary)
		
		assert result.dtype == np.uint8
		assert result.shape == binary.shape
		# Small noise should be removed by opening
		assert result[5, 5] == 0
	
	def test_component_filtering(self, pipeline_with_mock):
		"""Test connected component filtering."""
		# Create image with components of different sizes
		binary = np.zeros((100, 100), dtype=np.uint8)
		binary[10:20, 10:20] = 255  # 10x10 = 100 pixels
		binary[50:52, 50:52] = 255  # 2x2 = 4 pixels (should be removed)
		binary[70:90, 70:90] = 255  # 20x20 = 400 pixels
		
		filter_config = {
			'enabled': True,
			'min_area': 50,
			'max_area': 10000,
			'min_aspect_ratio': 0.1,
			'max_aspect_ratio': 10.0
		}
		
		result = pipeline_with_mock._filter_components(binary, filter_config)
		
		# Small component should be removed
		assert result[51, 51] == 0
		# Large components should remain
		assert result[15, 15] == 255
		assert result[80, 80] == 255
	
	def test_remove_borders(self, pipeline_with_mock):
		"""Test border removal."""
		# Create image with border component
		binary = np.zeros((100, 100), dtype=np.uint8)
		binary[0:10, :] = 255  # Top border
		binary[40:60, 40:60] = 255  # Center component
		
		result = pipeline_with_mock._remove_borders(binary)
		
		# Border should be removed
		assert result[5, 50] == 0
		# Center component should remain
		assert result[50, 50] == 255


class TestHelperFunctions:
	"""Test suite for helper functions."""
	
	def test_ensure_binary(self):
		"""Test ensure_binary function."""
		# Test with grayscale values
		img = np.array([[100, 200], [50, 150]], dtype=np.uint8)
		binary = ensure_binary(img)
		
		assert binary.dtype == np.uint8
		assert set(np.unique(binary)).issubset({0, 255})
		assert binary[0, 1] == 255  # 200 > 127
		assert binary[1, 0] == 0    # 50 < 127
	
	def test_ensure_binary_already_binary(self):
		"""Test ensure_binary with already binary image."""
		binary = np.array([[0, 255], [255, 0]], dtype=np.uint8)
		result = ensure_binary(binary)
		
		assert np.array_equal(result, binary)


class TestRunPipeline:
	"""Test suite for run_pipeline convenience function."""
	
	def test_run_pipeline_with_dict_config(self, simple_image, mock_algorithm):
		"""Test run_pipeline with dictionary config."""
		config_dict = {
			'method': 'mock_method',
			'method_params': {},
			'post_processing': {'enabled': False},
			'enhancement': {'enabled': False},
			'evaluation': {'enabled': False}
		}
		
		algorithm_registry = {'mock_method': mock_algorithm}
		
		result = run_pipeline(simple_image, config_dict, algorithm_registry)
		
		assert 'binary_image' in result
		assert result['method'] == 'mock_method'


# Integration test
class TestIntegration:
	"""Integration tests for the complete pipeline."""
	
	def test_end_to_end_simple(self, simple_image, mock_algorithm):
		"""Test end-to-end pipeline with simple configuration."""
		# Setup
		config = get_default_config()
		config.method = "mock_method"
		config.post_processing.enabled = True
		config.post_processing.morphology = {
			'opening': {'enabled': True, 'kernel_size': 3},
			'closing': {'enabled': True, 'kernel_size': 3}
		}
		
		pipeline = BinarizationPipeline(config)
		pipeline.register_algorithm("mock_method", mock_algorithm)
		
		# Run
		result = pipeline.run(simple_image, return_intermediates=True)
		
		# Verify
		assert result['binary_image'].shape == simple_image.shape
		assert 'intermediates' in result
		assert 'preprocessed' in result['intermediates']
		assert 'binarized' in result['intermediates']
		assert 'post_processed' in result['intermediates']
		assert result['processing_time'] > 0