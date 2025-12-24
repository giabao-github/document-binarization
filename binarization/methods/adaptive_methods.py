"""
Adaptive (local) thresholding methods for document binarization.
These methods compute different threshold values for each pixel based on
local neighborhood statistics, making them robust to non-uniform illumination.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
import warnings

from ..core.base import BinarizationAlgorithm, BinarizationResult, ensure_binary


class MeanAdaptiveThreshold(BinarizationAlgorithm):
	"""
	Mean adaptive thresholding.
	Computes threshold for each pixel as the mean of its local neighborhood
	minus a constant C. Simple but effective for documents with variable lighting.
	Formula: T(x,y) = mean(neighborhood) - C
	Best for: Documents with gradual illumination changes.
	Example:
		>>> method = MeanAdaptiveThreshold()
		>>> result = method.binarize(image, window_size=15, C=5)
	"""
	
	def __init__(self):
		super().__init__(
			name="mean_adaptive",
			description="Mean adaptive thresholding with local mean"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 15,
			'C': 5,  # Constant subtracted from mean
			'use_opencv': True
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 101),  # Must be odd
			'C': (-50, 50)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 15)
		C = params.get('C', 5)
		use_opencv = params.get('use_opencv', True)
		
		# Ensure window_size is odd
		if window_size % 2 == 0:
			window_size += 1
		
		if use_opencv:
			# Use OpenCV's implementation
			binary = cv2.adaptiveThreshold(
				image,
				255,
				cv2.ADAPTIVE_THRESH_MEAN_C,
				cv2.THRESH_BINARY,
				window_size,
				C
			)
		else:
			# Custom implementation
			# Compute local mean using convolution
			kernel = np.ones((window_size, window_size)) / (window_size ** 2)
			local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
			
			# Apply threshold
			binary = np.where(image > local_mean - C, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=None,  # Varies per pixel
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				'C': C,
				'implementation': 'opencv' if use_opencv else 'custom'
			}
		)


class GaussianAdaptiveThreshold(BinarizationAlgorithm):
	"""
	Gaussian adaptive thresholding.
	Similar to mean adaptive but uses a Gaussian-weighted neighborhood,
	giving more weight to pixels closer to the center.
	Formula: T(x,y) = gaussian_weighted_mean(neighborhood) - C
	Best for: Documents with noise where nearby pixels are more relevant.
	Example:
		>>> method = GaussianAdaptiveThreshold()
		>>> result = method.binarize(image, window_size=15, C=5)
	"""
	
	def __init__(self):
		super().__init__(
			name="gaussian_adaptive",
			description="Gaussian adaptive thresholding with weighted mean"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 15,
			'C': 5,
			'use_opencv': True
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 101),
			'C': (-50, 50)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 15)
		C = params.get('C', 5)
		use_opencv = params.get('use_opencv', True)
		
		# Ensure window_size is odd
		if window_size % 2 == 0:
			window_size += 1
		
		if use_opencv:
			# Use OpenCV's implementation
			binary = cv2.adaptiveThreshold(
				image,
				255,
				cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
				cv2.THRESH_BINARY,
				window_size,
				C
			)
		else:
			# Custom implementation with Gaussian blur
			sigma = window_size / 6.0  # Standard choice
			local_mean = cv2.GaussianBlur(image.astype(np.float32), (window_size, window_size), sigma)
			
			# Apply threshold
			binary = np.where(image > local_mean - C, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=None,
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				'C': C,
				'implementation': 'opencv' if use_opencv else 'custom'
			}
		)


class NiblackThreshold(BinarizationAlgorithm):
	"""
	Niblack's local thresholding method.
	Computes threshold based on local mean and standard deviation.
	Works well for text but can produce noisy backgrounds.
	Formula: T(x,y) = m(x,y) + k × σ(x,y)
	where m = local mean, σ = local standard deviation
	Best for: High-quality documents with consistent text.
	Limitation: Can produce noisy backgrounds in low-contrast regions.
	Example:
		>>> method = NiblackThreshold()
		>>> result = method.binarize(image, window_size=15, k=-0.2)
	"""
	
	def __init__(self):
		super().__init__(
			name="niblack",
			description="Niblack's adaptive thresholding with local statistics"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 15,
			'k': -0.2,  # Typical range: -0.2 to -0.5
			'use_opencv': False  # OpenCV doesn't have native Niblack
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 101),
			'k': (-1.0, 0.5)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 15)
		k = params.get('k', -0.2)
		
		# Ensure window_size is odd
		if window_size % 2 == 0:
			window_size += 1
		
		# Compute local mean and standard deviation
		image_float = image.astype(np.float32)
		
		# Use uniform filter for mean
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		
		# Compute local standard deviation
		# σ² = E[X²] - E[X]²
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = local_mean_sq - local_mean ** 2
		local_variance = np.maximum(local_variance, 0)  # Avoid negative due to numerical errors
		local_std = np.sqrt(local_variance)
		
		# Niblack's threshold
		threshold = local_mean + k * local_std
		
		# Apply threshold
		binary = np.where(image_float > threshold, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(np.mean(threshold)),  # Average threshold
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				'k': k,
				'mean_threshold': float(np.mean(threshold)),
				'std_threshold': float(np.std(threshold))
			}
		)


class SauvolaThreshold(BinarizationAlgorithm):
	"""
	Sauvola's adaptive thresholding method.
	Improvement over Niblack that uses dynamic range of standard deviation
	to better handle varying contrast. Most popular adaptive method for documents.
	Formula: T(x,y) = m(x,y) × [1 + k × (σ(x,y)/R - 1)]
	where m = local mean, σ = local std, R = dynamic range (default 128)
	Best for: Most document types, especially degraded documents.
	Example:
		>>> method = SauvolaThreshold()
		>>> result = method.binarize(image, window_size=25, k=0.2)
	"""
	
	def __init__(self):
		super().__init__(
			name="sauvola",
			description="Sauvola's adaptive thresholding with dynamic range"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 25,
			'k': 0.2,  # Typical range: 0.2 to 0.5
			'R': 128.0  # Dynamic range of standard deviation
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 101),
			'k': (0.0, 1.0),
			'R': (1.0, 255.0)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.2)
		R = params.get('R', 128.0)
		
		# Ensure window_size is odd
		if window_size % 2 == 0:
			window_size += 1
		
		# Compute local statistics
		image_float = image.astype(np.float32)
		
		# Local mean
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		
		# Local standard deviation
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = local_mean_sq - local_mean ** 2
		local_variance = np.maximum(local_variance, 0)
		local_std = np.sqrt(local_variance)
		
		# Sauvola's threshold formula
		threshold = local_mean * (1.0 + k * (local_std / R - 1.0))
		
		# Apply threshold
		binary = np.where(image_float > threshold, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(np.mean(threshold)),
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				'k': k,
				'R': R,
				'mean_threshold': float(np.mean(threshold)),
				'std_threshold': float(np.std(threshold))
			}
		)


class WolfThreshold(BinarizationAlgorithm):
	"""
	Wolf's adaptive thresholding method.
	Enhancement of Sauvola's method that adds image statistics to better
	handle low-contrast regions. Particularly effective for degraded documents.
	Formula: T(x,y) = (1-k)×m(x,y) + k×min_I + k×σ(x,y)/max_σ×(m(x,y)-min_I)
	where min_I = minimum image intensity, max_σ = maximum std deviation
	Best for: Degraded documents with very low contrast regions.
	Example:
		>>> method = WolfThreshold()
		>>> result = method.binarize(image, window_size=25, k=0.5)
	"""
	
	def __init__(self):
		super().__init__(
			name="wolf",
			description="Wolf's adaptive thresholding (enhanced Sauvola)"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 25,
			'k': 0.5
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 101),
			'k': (0.0, 1.0)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.5)
		
		# Ensure window_size is odd
		if window_size % 2 == 0:
			window_size += 1
		
		# Compute local statistics
		image_float = image.astype(np.float32)
		
		# Local mean
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		
		# Local standard deviation
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = local_mean_sq - local_mean ** 2
		local_variance = np.maximum(local_variance, 0)
		local_std = np.sqrt(local_variance)
		
		# Global statistics
		min_I = float(np.min(image_float))
		max_std = float(np.max(local_std))
		
		# Avoid division by zero
		if max_std == 0:
			max_std = 1.0
		
		# Wolf's threshold formula
		threshold = ((1.0 - k) * local_mean + 
					k * min_I + 
					k * (local_std / max_std) * (local_mean - min_I))
		
		# Apply threshold
		binary = np.where(image_float > threshold, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(np.mean(threshold)),
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				'k': k,
				'min_intensity': min_I,
				'max_std': max_std,
				'mean_threshold': float(np.mean(threshold)),
				'std_threshold': float(np.std(threshold))
			}
		)


class BradleyThreshold(BinarizationAlgorithm):
	"""
	Bradley's adaptive thresholding using integral images.
	Fast adaptive thresholding using integral images for efficient computation
	of local statistics. Offers significant speedup over standard methods.
	Uses integral images to compute mean in O(1) time per pixel regardless
	of window size, making it ~20x faster than naive sliding window.
	Best for: Large images where speed is critical.
	Example:
		>>> method = BradleyThreshold()
		>>> result = method.binarize(image, window_size=25, t=15)
	"""
	
	def __init__(self):
		super().__init__(
			name="bradley",
			description="Bradley's adaptive thresholding with integral images"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'window_size': 25,
			't': 15,  # Percentage threshold (0-100)
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (3, 201),
			't': (0, 50)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		window_size = params.get('window_size', 25)
		t = params.get('t', 15)
		
		h, w = image.shape
		
		# Compute integral image
		image_float = image.astype(np.float32)
		integral = cv2.integral(image_float)
		
		# Initialize output
		binary = np.zeros_like(image, dtype=np.uint8)
		
		# Half window size
		s2 = window_size // 2
		
		# Compute threshold for each pixel using integral image
		for i in range(h):
			for j in range(w):
				# Define window bounds
				y1 = max(0, i - s2)
				y2 = min(h - 1, i + s2)
				x1 = max(0, j - s2)
				x2 = min(w - 1, j + s2)
				
				# Compute area
				count = (y2 - y1 + 1) * (x2 - x1 + 1)
				
				# Compute sum using integral image
				# Note: integral image is (h+1)×(w+1)
				sum_val = (
					integral[y2+1, x2+1] - 
					integral[y1, x2+1] - 
					integral[y2+1, x1] + 
					integral[y1, x1]
				)
				
				# Compute mean
				mean = sum_val / count if count > 0 else 0
				
				# Apply threshold
				if image[i, j] > mean * (1.0 - t / 100.0):
					binary[i, j] = 255
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=None,
			processing_time=processing_time,
			metadata={
				'window_size': window_size,
				't': t,
				'uses_integral_image': True
			}
		)