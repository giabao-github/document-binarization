"""
Global thresholding methods for document binarization.
This module implements various global thresholding algorithms that compute a single threshold value for the entire image.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
import warnings

from ..core.base import BinarizationAlgorithm, BinarizationResult, ensure_binary


class ManualThreshold(BinarizationAlgorithm):
	"""
	Simple manual thresholding.
	Applies a user-specified threshold value to binarize the image.
	Pixels above threshold become foreground (255), below become background (0).
	Best for: Images with known intensity distribution or for quick testing.
	Example:
		>>> method = ManualThreshold()
		>>> result = method.binarize(image, threshold=128)
	"""
	
	def __init__(self):
		super().__init__(
			name="manual",
			description="Manual threshold with user-specified value"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'threshold': 127,
			'normalized': False  # If True, threshold is 0-1 range
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'threshold': (0, 255),
			'normalized': (False, True)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		threshold = params.get('threshold', 127)
		normalized = params.get('normalized', False)
		
		# Handle normalized threshold
		if normalized:
			if not 0 <= threshold <= 1:
				raise ValueError(f"Normalized threshold must be in [0,1], got {threshold}")
			threshold = int(threshold * 255)
		
		# Apply threshold
		binary = np.where(image > threshold, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(threshold),
			processing_time=processing_time,
			metadata={'normalized': normalized}
		)


class OtsuThreshold(BinarizationAlgorithm):
	"""
	Otsu's automatic thresholding method.
	Automatically determines optimal threshold by minimizing intra-class variance
	(or equivalently, maximizing inter-class variance). Works best for bimodal
	histograms with clear separation between foreground and background.
	Best for: Clean scanned documents with good contrast.
	Example:
		>>> method = OtsuThreshold()
		>>> result = method.binarize(image)
	"""
	
	def __init__(self):
		super().__init__(
			name="otsu",
			description="Otsu's automatic threshold selection"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'use_opencv': True  # Use OpenCV implementation (faster)
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'use_opencv': (False, True)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		use_opencv = params.get('use_opencv', True)
		
		if use_opencv:
			# Use OpenCV's optimized implementation
			threshold_value, binary = cv2.threshold(
				image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
			)
		else:
			# Custom implementation
			threshold_value = self._compute_otsu_threshold(image)
			binary = np.where(image > threshold_value, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(threshold_value),
			processing_time=processing_time,
			metadata={
				'implementation': 'opencv' if use_opencv else 'custom',
				'histogram_bimodal': self._check_bimodal(image)
			}
		)
	
	def _compute_otsu_threshold(self, image: np.ndarray) -> float:
		"""
		Compute Otsu threshold using inter-class variance maximization.
		Args:
			image: Grayscale image
		Returns:
			Optimal threshold value
		"""
		# Compute histogram
		hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
		hist = hist.astype(float)
		
		# Normalize histogram to get probabilities
		hist_norm = hist / hist.sum()
		
		# Cumulative sums
		cumsum = np.cumsum(hist_norm)
		cumsum_mean = np.cumsum(hist_norm * np.arange(256))
		
		# Global mean
		global_mean = cumsum_mean[-1]
		
		# Avoid division by zero
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			
			# Between-class variance
			# σ²_b = w0 * w1 * (μ0 - μ1)²
			between_class_variance = np.zeros(256)
			
			for t in range(256):
				w0 = cumsum[t]  # Weight of background
				w1 = 1.0 - w0   # Weight of foreground
				
				if w0 == 0 or w1 == 0:
					continue
				
				mu0 = cumsum_mean[t] / w0  # Mean of background
				mu1 = (global_mean - cumsum_mean[t]) / w1  # Mean of foreground
				
				between_class_variance[t] = w0 * w1 * (mu0 - mu1) ** 2
		
		# Find threshold with maximum between-class variance
		optimal_threshold = np.argmax(between_class_variance)
		
		return float(optimal_threshold)
	
	def _check_bimodal(self, image: np.ndarray) -> bool:
		"""
		Check if histogram is approximately bimodal.
		Args:
			image: Grayscale image
		Returns:
			True if histogram appears bimodal
		"""
		hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
		
		# Smooth histogram
		hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=2)
		
		# Find peaks (local maxima)
		peaks = []
		for i in range(1, len(hist_smooth) - 1):
			if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
				if hist_smooth[i] > hist_smooth.max() * 0.1:  # Significant peak
					peaks.append(i)
		
		# Bimodal if we have 2 significant peaks
		return len(peaks) >= 2


class TriangleThreshold(BinarizationAlgorithm):
	"""
	Triangle (chord) thresholding method.
	Geometric method that works well for images with uni-modal histograms
	where one peak is much larger than the other. Constructs a line from
	the histogram peak to the farthest end, then finds the point with
	maximum perpendicular distance.
	Best for: Images with bright background and dark foreground (or vice versa).
	Example:
		>>> method = TriangleThreshold()
		>>> result = method.binarize(image)
	"""
	
	def __init__(self):
		super().__init__(
			name="triangle",
			description="Triangle/chord thresholding method"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Compute threshold
		threshold_value = self._compute_triangle_threshold(image)
		
		# Apply threshold
		binary = np.where(image > threshold_value, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(threshold_value),
			processing_time=processing_time,
			metadata={}
		)
	
	def _compute_triangle_threshold(self, image: np.ndarray) -> float:
		"""
		Compute triangle threshold.
		Args:
			image: Grayscale image
		Returns:
			Optimal threshold value
		"""
		# Compute histogram
		hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
		hist = hist.astype(float)
		
		# Find the peak (mode) of the histogram
		peak_idx = np.argmax(hist)
		
		# Determine which side has more weight
		left_weight = np.sum(hist[:peak_idx])
		right_weight = np.sum(hist[peak_idx:])
		
		# Choose the side opposite to the peak
		if left_weight > right_weight:
			# Peak is on right, search on left
			search_range = np.arange(0, peak_idx)
		else:
			# Peak is on left, search on right
			search_range = np.arange(peak_idx, 256)
		
		if len(search_range) == 0:
			# Fallback to peak
			return float(peak_idx)
		
		# Find the farthest non-zero point
		non_zero_indices = np.where(hist[search_range] > 0)[0]
		if len(non_zero_indices) == 0:
			return float(peak_idx)
		
		if left_weight > right_weight:
			far_idx = search_range[non_zero_indices[0]]
		else:
			far_idx = search_range[non_zero_indices[-1]]
		
		# Create line from peak to far point
		x1, y1 = peak_idx, hist[peak_idx]
		x2, y2 = far_idx, hist[far_idx]
		
		# Compute perpendicular distance from each point to this line
		max_distance = 0
		threshold_idx = peak_idx
		
		# Line equation: ax + by + c = 0
		a = y2 - y1
		b = x1 - x2
		c = x2 * y1 - x1 * y2
		
		norm = np.sqrt(a**2 + b**2)
		if norm == 0:
			return float(peak_idx)
		
		for idx in search_range:
			if hist[idx] > 0:
				# Perpendicular distance from point (idx, hist[idx]) to line
				distance = abs(a * idx + b * hist[idx] + c) / norm
				
				if distance > max_distance:
					max_distance = distance
					threshold_idx = idx
		
		return float(threshold_idx)


class EntropyThreshold(BinarizationAlgorithm):
	"""
	Kapur's entropy-based thresholding method.
	Selects threshold that maximizes the sum of entropies of the foreground
	and background distributions. Based on information theory.
	Best for: Images with complex intensity distributions.
	Example:
		>>> method = EntropyThreshold()
		>>> result = method.binarize(image)
	"""
	
	def __init__(self):
		super().__init__(
			name="entropy",
			description="Kapur's entropy-based thresholding"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Compute threshold
		threshold_value = self._compute_entropy_threshold(image)
		
		# Apply threshold
		binary = np.where(image > threshold_value, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(threshold_value),
			processing_time=processing_time,
			metadata={}
		)
	
	def _compute_entropy_threshold(self, image: np.ndarray) -> float:
		"""
		Compute entropy-based threshold using Kapur's method.
		Args:
			image: Grayscale image
		Returns:
			Optimal threshold value
		"""
		# Compute histogram
		hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
		hist = hist.astype(float)
		
		# Normalize to get probabilities
		hist_norm = hist / (hist.sum() + 1e-10)
		
		# Compute cumulative sums
		cumsum = np.cumsum(hist_norm)
		
		max_entropy = -np.inf
		optimal_threshold = 0
		
		for t in range(1, 255):
			# Background probabilities
			pb = cumsum[t]
			if pb == 0 or pb == 1:
				continue
			
			# Foreground probabilities
			pf = 1.0 - pb
			
			# Background entropy
			hist_bg = hist_norm[:t+1]
			hist_bg_normalized = hist_bg / (pb + 1e-10)
			
			# Remove zeros to avoid log(0)
			hist_bg_nonzero = hist_bg_normalized[hist_bg_normalized > 0]
			if len(hist_bg_nonzero) > 0:
				entropy_bg = -np.sum(hist_bg_nonzero * np.log(hist_bg_nonzero + 1e-10))
			else:
				entropy_bg = 0
			
			# Foreground entropy
			hist_fg = hist_norm[t+1:]
			hist_fg_normalized = hist_fg / (pf + 1e-10)
			
			hist_fg_nonzero = hist_fg_normalized[hist_fg_normalized > 0]
			if len(hist_fg_nonzero) > 0:
				entropy_fg = -np.sum(hist_fg_nonzero * np.log(hist_fg_nonzero + 1e-10))
			else:
				entropy_fg = 0
			
			# Total entropy
			total_entropy = entropy_bg + entropy_fg
			
			if total_entropy > max_entropy:
				max_entropy = total_entropy
				optimal_threshold = t
		
		return float(optimal_threshold)


class MinimumErrorThreshold(BinarizationAlgorithm):
	"""
	Kittler-Illingworth minimum error thresholding.
	Assumes the histogram is a mixture of two Gaussian distributions and
	uses maximum likelihood estimation to find the optimal threshold.
	More robust than Otsu for non-uniform illumination.
	Best for: Images with Gaussian-like intensity distributions.
	Example:
		>>> method = MinimumErrorThreshold()
		>>> result = method.binarize(image)
	"""
	
	def __init__(self):
		super().__init__(
			name="minimum_error",
			description="Kittler-Illingworth minimum error thresholding"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Compute threshold
		threshold_value = self._compute_minimum_error_threshold(image)
		
		# Apply threshold
		binary = np.where(image > threshold_value, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(threshold_value),
			processing_time=processing_time,
			metadata={}
		)
	
	def _compute_minimum_error_threshold(self, image: np.ndarray) -> float:
		"""
		Compute minimum error threshold using Kittler-Illingworth method.
		Args:
			image: Grayscale image
		Returns:
			Optimal threshold value
		"""
		# Compute histogram
		hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
		hist = hist.astype(float)
		
		# Normalize
		hist_norm = hist / (hist.sum() + 1e-10)
		
		# Intensity values
		intensities = np.arange(256)
		
		min_criterion = np.inf
		optimal_threshold = 0
		
		for t in range(1, 255):
			# Background class
			hist_bg = hist[:t+1]
			w_bg = np.sum(hist_bg)
			
			if w_bg == 0:
				continue
			
			mean_bg = np.sum(intensities[:t+1] * hist_bg) / w_bg
			var_bg = np.sum(((intensities[:t+1] - mean_bg) ** 2) * hist_bg) / w_bg
			
			# Foreground class
			hist_fg = hist[t+1:]
			w_fg = np.sum(hist_fg)
			
			if w_fg == 0:
				continue
			
			mean_fg = np.sum(intensities[t+1:] * hist_fg) / w_fg
			var_fg = np.sum(((intensities[t+1:] - mean_fg) ** 2) * hist_fg) / w_fg
			
			# Avoid log of zero or negative
			if var_bg <= 0:
				var_bg = 1e-10
			if var_fg <= 0:
				var_fg = 1e-10
			
			# Criterion function (to minimize)
			# J(t) = 1 + 2*[w_bg*log(σ_bg) + w_fg*log(σ_fg)] - 2*[w_bg*log(w_bg) + w_fg*log(w_fg)]
			criterion = (1 + 2 * (w_bg * np.log(np.sqrt(var_bg)) + 
								  w_fg * np.log(np.sqrt(var_fg))) -
						 2 * (w_bg * np.log(w_bg + 1e-10) + 
							  w_fg * np.log(w_fg + 1e-10)))
			
			if criterion < min_criterion:
				min_criterion = criterion
				optimal_threshold = t
		
		return float(optimal_threshold)