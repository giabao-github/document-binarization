"""
Unified pipeline for document binarization and enhancement.
This module provides the main pipeline that coordinates binarization, post-processing, and text enhancement operations.
"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import cv2
import logging

from .base import BinarizationAlgorithm, BinarizationResult
from .config import PipelineConfig


logger = logging.getLogger(__name__)


class BinarizationPipeline:
	"""
	Main pipeline for document binarization.
	This class orchestrates the complete binarization workflow:
	1. Input validation and preprocessing
	2. Binarization algorithm application
	3. Post-processing operations
	4. Text enhancement (optional)
	5. Quality evaluation (optional)
	Example:
		>>> from binarization.core.pipeline import BinarizationPipeline
		>>> from binarization.core.config import get_default_config
		>>> 
		>>> config = get_default_config()
		>>> config.method = "otsu"
		>>> pipeline = BinarizationPipeline(config)
		>>> 
		>>> result = pipeline.run(image)
		>>> binary = result['binary_image']
	"""
	
	def __init__(
		self,
		config: Optional[PipelineConfig] = None,
		algorithm_registry: Optional[Dict[str, BinarizationAlgorithm]] = None
	):
		"""
		Initialize the pipeline.
		Args:
			config: Pipeline configuration
			algorithm_registry: Dictionary mapping method names to algorithm instances
		"""
		from .config import get_default_config
		
		self.config = config or get_default_config()
		self.algorithm_registry = algorithm_registry or {}
		self.post_processors = []
		self.enhancers = []
		
		logger.info(f"Initialized pipeline with method: {self.config.method}")
	
	def register_algorithm(self, name: str, algorithm: BinarizationAlgorithm) -> None:
		"""
		Register a binarization algorithm.
		Args:
			name: Algorithm identifier
			algorithm: Algorithm instance
		"""
		self.algorithm_registry[name] = algorithm
		logger.debug(f"Registered algorithm: {name}")
	
	def get_algorithm(self, name: str) -> BinarizationAlgorithm:
		"""
		Get registered algorithm by name.
		Args:
			name: Algorithm identifier
		Returns:
			Algorithm instance
		Raises:
			ValueError: If algorithm not found
		"""
		if name not in self.algorithm_registry:
			available = list(self.algorithm_registry.keys())
			raise ValueError(
				f"Algorithm '{name}' not found. Available: {available}"
			)
		return self.algorithm_registry[name]
	
	def run(
		self,
		image: np.ndarray,
		method: Optional[str] = None,
		method_params: Optional[Dict[str, Any]] = None,
		return_intermediates: bool = False
	) -> Dict[str, Any]:
		"""
		Run the complete binarization pipeline.
		Args:
			image: Input grayscale or RGB image
			method: Binarization method to use (overrides config)
			method_params: Method parameters (overrides config)
			return_intermediates: Whether to return intermediate results
		Returns:
			Dictionary containing:
				- binary_image: Final binary output
				- method: Method used
				- parameters: Parameters used
				- processing_time: Total time in seconds
				- intermediates: Dict of intermediate results (if requested)
				- metadata: Additional information
		"""
		start_time = time.time()
		intermediates = {} if return_intermediates else None
		
		# Use provided method or fall back to config
		method = method or self.config.method
		method_params = method_params or self.config.method_params
		
		try:
			# Step 1: Validate and preprocess input
			logger.debug("Step 1: Input validation and preprocessing")
			processed_image = self._preprocess_image(image)
			if return_intermediates:
				intermediates['preprocessed'] = processed_image.copy()
			
			# Step 2: Apply binarization algorithm
			logger.debug(f"Step 2: Applying {method} binarization")
			algorithm = self.get_algorithm(method)
			binarization_result = algorithm.binarize(processed_image, **method_params)
			binary_image = binarization_result.binary_image
			if return_intermediates:
				intermediates['binarized'] = binary_image.copy()
			
			# Step 3: Post-processing
			if self.config.post_processing.enabled:
				logger.debug("Step 3: Applying post-processing")
				binary_image = self._apply_post_processing(binary_image)
				if return_intermediates:
					intermediates['post_processed'] = binary_image.copy()
			
			# Step 4: Text enhancement
			if self.config.enhancement.enabled:
				logger.debug("Step 4: Applying text enhancement")
				binary_image = self._apply_enhancement(binary_image)
				if return_intermediates:
					intermediates['enhanced'] = binary_image.copy()
			
			# Prepare result
			processing_time = time.time() - start_time
			
			result = {
				'binary_image': binary_image,
				'method': method,
				'parameters': method_params,
				'threshold': binarization_result.threshold,
				'processing_time': processing_time,
				'metadata': {
					'input_shape': image.shape,
					'output_shape': binary_image.shape,
					'algorithm_metadata': binarization_result.metadata
				}
			}
			
			if return_intermediates:
				result['intermediates'] = intermediates
			
			logger.info(f"Pipeline completed in {processing_time:.3f}s")
			return result
			
		except Exception as e:
			logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
			raise
	
	def run_batch(
		self,
		images: List[np.ndarray],
		method: Optional[str] = None,
		method_params: Optional[Dict[str, Any]] = None
	) -> List[Dict[str, Any]]:
		"""
		Run pipeline on a batch of images.
		Args:
			images: List of input images
			method: Binarization method to use
			method_params: Method parameters
		Returns:
			List of result dictionaries
		"""
		results = []
		for i, image in enumerate(images):
			logger.info(f"Processing image {i+1}/{len(images)}")
			result = self.run(image, method, method_params)
			results.append(result)
		return results
	
	def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
		"""
		Preprocess input image.
		Args:
			image: Input image
		Returns:
			Preprocessed grayscale uint8 image
		"""
		# Convert to grayscale if needed
		if len(image.shape) == 3:
			if image.shape[2] == 3:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			elif image.shape[2] == 4:
				image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
		
		# Ensure uint8
		if image.dtype != np.uint8:
			if image.dtype in [np.float32, np.float64]:
				if image.max() <= 1.0:
					image = (image * 255).astype(np.uint8)
				else:
					image = image.astype(np.uint8)
			elif image.dtype == np.uint16:
				image = (image // 256).astype(np.uint8)
			else:
				image = image.astype(np.uint8)
		
		return image
	
	def _apply_post_processing(self, binary_image: np.ndarray) -> np.ndarray:
		"""
		Apply post-processing operations.
		Args:
			binary_image: Binary image from binarization
		Returns:
			Post-processed binary image
		"""
		result = binary_image.copy()
		config = self.config.post_processing
		
		# Morphological operations
		if config.morphology:
			morph_config = config.morphology
			
			# Opening (remove small noise)
			if morph_config.get('opening', {}).get('enabled', False):
				kernel_size = morph_config['opening'].get('kernel_size', 3)
				kernel_shape = morph_config['opening'].get('kernel_shape', 'rect')
				kernel = self._get_morphology_kernel(kernel_size, kernel_shape)
				result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
			
			# Closing (fill small gaps)
			if morph_config.get('closing', {}).get('enabled', False):
				kernel_size = morph_config['closing'].get('kernel_size', 3)
				kernel_shape = morph_config['closing'].get('kernel_shape', 'rect')
				kernel = self._get_morphology_kernel(kernel_size, kernel_shape)
				result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
		
		# Connected component filtering
		if config.component_filtering and config.component_filtering.get('enabled', False):
			result = self._filter_components(result, config.component_filtering)
		
		# Border removal
		if config.border_removal:
			result = self._remove_borders(result)
		
		return result
	
	def _apply_enhancement(self, binary_image: np.ndarray) -> np.ndarray:
		"""
		Apply text enhancement operations.
		Args:
			binary_image: Binary image from post-processing
		Returns:
			Enhanced binary image
		"""
		# Placeholder for enhancement operations
		# Will be implemented in enhancement module
		return binary_image
	
	def _get_morphology_kernel(self, size: int, shape: str = 'rect') -> np.ndarray:
		"""
		Get morphology kernel.
		Args:
			size: Kernel size
			shape: Kernel shape ('rect', 'ellipse', 'cross')
		Returns:
			Morphology kernel
		"""
		if shape == 'ellipse':
			return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
		elif shape == 'cross':
			return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
		else:  # rect
			return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
	
	def _filter_components(
		self,
		binary_image: np.ndarray,
		filter_config: Dict[str, Any]
	) -> np.ndarray:
		"""
		Filter connected components by size and aspect ratio.
		Args:
			binary_image: Binary image
			filter_config: Filtering configuration
		Returns:
			Filtered binary image
		"""
		# Find connected components
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		# Create output mask
		output = np.zeros_like(binary_image)
		
		# Filter parameters
		min_area = filter_config.get('min_area', 10)
		max_area = filter_config.get('max_area', 100000)
		min_aspect = filter_config.get('min_aspect_ratio', 0.1)
		max_aspect = filter_config.get('max_aspect_ratio', 10.0)
		
		# Filter each component (skip background label 0)
		for label in range(1, num_labels):
			area = stats[label, cv2.CC_STAT_AREA]
			width = stats[label, cv2.CC_STAT_WIDTH]
			height = stats[label, cv2.CC_STAT_HEIGHT]
			
			# Calculate aspect ratio
			aspect_ratio = width / height if height > 0 else 0
			
			# Check filters
			if min_area <= area <= max_area and min_aspect <= aspect_ratio <= max_aspect:
				output[labels == label] = 255
		
		return output
	
	def _remove_borders(self, binary_image: np.ndarray) -> np.ndarray:
		"""
		Remove border/frame components.
		Args:
			binary_image: Binary image
		Returns:
			Image with borders removed
		"""
		# Find connected components
		num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		output = binary_image.copy()
		h, w = binary_image.shape
		
		# Check each component if it touches borders
		for label in range(1, num_labels):
			x = stats[label, cv2.CC_STAT_LEFT]
			y = stats[label, cv2.CC_STAT_TOP]
			width = stats[label, cv2.CC_STAT_WIDTH]
			height = stats[label, cv2.CC_STAT_HEIGHT]
			
			# Check if component touches image borders
			touches_border = (x == 0 or y == 0 or 
							x + width >= w or y + height >= h)
			
			# Remove large border components
			area = stats[label, cv2.CC_STAT_AREA]
			if touches_border and area > 0.1 * h * w:
				output[labels == label] = 0
		
		return output


def run_pipeline(
	image: np.ndarray,
	config: Dict[str, Any],
	algorithm_registry: Optional[Dict[str, BinarizationAlgorithm]] = None
) -> Dict[str, Any]:
	"""
	Convenience function to run pipeline with dictionary config.
	Args:
		image: Input image
		config: Configuration dictionary
		algorithm_registry: Optional algorithm registry
	Returns:
		Pipeline result dictionary
	"""
	from .config import PipelineConfig
	
	# Convert dict to PipelineConfig
	if isinstance(config, dict):
		config = PipelineConfig.from_dict(config)
	
	pipeline = BinarizationPipeline(config, algorithm_registry)
	return pipeline.run(image)