"""
Morphological operations for binary image post-processing.
This module provides various morphological operations to clean up and enhance binary images after thresholding.
"""

import time
from typing import Dict, Any, Tuple, Optional, Literal
import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology

from ..core.base import PostProcessor


class MorphologicalOperations(PostProcessor):
	"""
	Morphological operations for binary image processing.
	Provides opening, closing, dilation, erosion, thinning, and skeletonization operations with configurable kernels.
	Example:
		>>> processor = MorphologicalOperations()
		>>> cleaned = processor.process(binary_image, operation='opening', kernel_size=3, kernel_shape='ellipse')
	"""
	
	def __init__(self):
		super().__init__(name="morphological_operations")
		self.kernel_shapes = {
			'rect': cv2.MORPH_RECT,
			'ellipse': cv2.MORPH_ELLIPSE,
			'cross': cv2.MORPH_CROSS
		}
	
	def process(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Apply morphological operation to binary image.
		Args:
			binary_image: Input binary image (0/255)
			**params: Operation parameters
				- operation: 'opening', 'closing', 'dilation', 'erosion', 'thinning', 'skeletonize'
				- kernel_size: Kernel size (default: 3)
				- kernel_shape: 'rect', 'ellipse', 'cross' (default: 'rect')
				- iterations: Number of iterations (default: 1)		
		Returns:
			Processed binary image
		"""
		operation = params.get('operation', 'opening')
		kernel_size = params.get('kernel_size', 3)
		kernel_shape = params.get('kernel_shape', 'rect')
		iterations = params.get('iterations', 1)
		
		# Create kernel
		kernel = self._get_kernel(kernel_size, kernel_shape)
		
		# Apply operation
		if operation == 'opening':
			result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
		elif operation == 'closing':
			result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
		elif operation == 'dilation':
			result = cv2.dilate(binary_image, kernel, iterations=iterations)
		elif operation == 'erosion':
			result = cv2.erode(binary_image, kernel, iterations=iterations)
		elif operation == 'thinning':
			# Convert to binary (foreground=True) for skimage using project convention: foreground pixels == 0
			binary_01 = (binary_image < 128).astype(np.uint8)
			thinned = morphology.thin(binary_01)
			# morphology.thin returns True for foreground; convert back to project convention (0 for foreground, 255 for background)
			result = (thinned == 0).astype(np.uint8) * 255
		elif operation == 'skeletonize':
			# Morphological skeleton using project convention (foreground==0)
			binary_01 = (binary_image < 128).astype(np.uint8)
			skel = morphology.skeletonize(binary_01)
			result = (skel == 0).astype(np.uint8) * 255
		elif operation == 'gradient':
			# Morphological gradient (edge detection)
			result = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
		elif operation == 'tophat':
			# Top hat (original - opening)
			result = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)
		elif operation == 'blackhat':
			# Black hat (closing - original)
			result = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
		else:
			raise ValueError(f"Unknown operation: {operation}")
		
		return result
	
	def _get_kernel(self, size: int, shape: str) -> np.ndarray:
		"""
		Create morphological kernel.
		Args:
			size: Kernel size
			shape: Kernel shape ('rect', 'ellipse', 'cross')	
		Returns:
			Morphological kernel
		"""
		if shape not in self.kernel_shapes:
			raise ValueError(f"Unknown kernel shape: {shape}")
		
		return cv2.getStructuringElement(self.kernel_shapes[shape], (size, size))
	
	def opening(self, binary_image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect', iterations: int = 1) -> np.ndarray:
		"""
		Apply morphological opening (erosion followed by dilation).
		Removes small objects and noise while preserving larger structures.
		Args:
			binary_image: Input binary image
			kernel_size: Kernel size
			kernel_shape: Kernel shape
			iterations: Number of iterations
		Returns:
			Opened binary image
		"""
		return self.process(
			binary_image, 
			operation='opening',
			kernel_size=kernel_size, 
			kernel_shape=kernel_shape,
			iterations=iterations
		)
	
	def closing(self, binary_image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect', iterations: int = 1) -> np.ndarray:
		"""
		Apply morphological closing (dilation followed by erosion).
		Fills small holes and gaps while preserving shape.
		Args:
			binary_image: Input binary image
			kernel_size: Kernel size
			kernel_shape: Kernel shape
			iterations: Number of iterations	
		Returns:
			Closed binary image
		"""
		return self.process(
			binary_image, 
			operation='closing',
			kernel_size=kernel_size, 
			kernel_shape=kernel_shape,
			iterations=iterations
		)
	
	def remove_noise(self, binary_image: np.ndarray, min_size: int = 10) -> np.ndarray:
		"""
		Remove small noise components.
		Args:
			binary_image: Input binary image
			min_size: Minimum component size to keep (pixels)	
		Returns:
			Denoised binary image
		"""
		# Opening with small kernel
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
		
		# Additional size filtering via connected components
		from .components import ComponentFilter
		filter_proc = ComponentFilter()
		return filter_proc.process(opened, min_area=min_size)
	
	def fill_gaps(self, binary_image: np.ndarray, gap_size: int = 3) -> np.ndarray:
		"""
		Fill small gaps in text strokes.
		Args:
			binary_image: Input binary image
			gap_size: Maximum gap size to fill (pixels)	
		Returns:
			Gap-filled binary image
		"""
		# Closing with kernel proportional to gap size
		kernel_size = gap_size * 2 + 1
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


class AdaptiveMorphology(PostProcessor):
	"""
	Adaptive morphological operations based on image characteristics.
	Automatically determines kernel sizes and operations based on
	text stroke width and image properties.
	Example:
		>>> processor = AdaptiveMorphology()
		>>> cleaned = processor.process(binary_image, mode='denoise')
	"""
	
	def __init__(self):
		super().__init__(name="adaptive_morphology")
	
	def process(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Apply adaptive morphological processing.
		Args:
			binary_image: Input binary image
			**params: Processing parameters
				- mode: 'denoise', 'fill_gaps', 'smooth', 'auto'
				- aggressiveness: 'low', 'medium', 'high'		
		Returns:
			Processed binary image
		"""
		mode = params.get('mode', 'auto')
		aggressiveness = params.get('aggressiveness', 'medium')
		
		# Estimate stroke width
		stroke_width = self._estimate_stroke_width(binary_image)
		
		# Determine kernel size based on stroke width
		if aggressiveness == 'low':
			kernel_size = max(2, stroke_width // 4)
		elif aggressiveness == 'medium':
			kernel_size = max(2, stroke_width // 3)
		else:  # high
			kernel_size = max(3, stroke_width // 2)
		
		# Ensure odd kernel size
		if kernel_size % 2 == 0:
			kernel_size += 1
		
		morph_ops = MorphologicalOperations()
		
		if mode == 'denoise':
			# Opening to remove noise
			result = morph_ops.opening(binary_image, kernel_size=kernel_size)
		elif mode == 'fill_gaps':
			# Closing to fill gaps
			result = morph_ops.closing(binary_image, kernel_size=kernel_size)
		elif mode == 'smooth':
			# Opening then closing
			result = morph_ops.opening(binary_image, kernel_size=kernel_size)
			result = morph_ops.closing(result, kernel_size=kernel_size)
		elif mode == 'auto':
			# Automatic processing based on image analysis
			noise_level = self._estimate_noise(binary_image)
			gap_level = self._estimate_gaps(binary_image)
			
			result = binary_image.copy()
			
			# If noisy, apply opening
			if noise_level > 0.05:  # >5% noise
				result = morph_ops.opening(result, kernel_size=kernel_size)
			
			# If gaps present, apply closing
			if gap_level > 0.03:  # >3% gaps
				result = morph_ops.closing(result, kernel_size=kernel_size)
		else:
			raise ValueError(f"Unknown mode: {mode}")
		
		return result
	
	def _estimate_stroke_width(self, binary_image: np.ndarray) -> int:
		"""
    Estimate average stroke width.
		Args:
			binary_image: Input binary image	
		Returns:
			Estimated stroke width in pixels
		"""
		# Use distance transform on foreground
		foreground = (binary_image == 0).astype(np.uint8)
		
		if foreground.sum() == 0:
			return 3  # Default
		
		dist_transform = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
		
		# Stroke width is approximately 2x the mean distance
		mean_dist = np.mean(dist_transform[dist_transform > 0])
		stroke_width = int(2 * mean_dist)
		
		return max(2, min(stroke_width, 20))  # Clip to reasonable range
	
	def _estimate_noise(self, binary_image: np.ndarray) -> float:
		"""
		Estimate noise level in binary image.
		Args:
			binary_image: Input binary image	
		Returns:
			Estimated noise ratio (0-1)
		"""
		# Small components are likely noise
		num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		if num_labels <= 1:
			return 0.0
		
		# Count small components
		areas = stats[1:, cv2.CC_STAT_AREA]
		small_components = np.sum(areas < 10)
		
		noise_ratio = small_components / max(num_labels - 1, 1)
		return min(noise_ratio, 1.0)
	
	def _estimate_gaps(self, binary_image: np.ndarray) -> float:
		"""
		Estimate gap level in text strokes.
		Args:
			binary_image: Input binary image	
		Returns:
			Estimated gap ratio (0-1)
		"""
		# Apply closing and see how much changes
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
		
		# Count pixels that changed
		changed = np.sum(closed != binary_image)
		total = binary_image.size
		
		gap_ratio = changed / total
		return min(gap_ratio, 1.0)