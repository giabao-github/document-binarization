"""
Advanced and hybrid binarization methods.
This module implements sophisticated binarization techniques that combine
preprocessing, multiple features, or hybrid approaches for challenging documents.
"""

import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from scipy import ndimage
import warnings

from ..core.base import BinarizationAlgorithm, BinarizationResult, ensure_binary


class CLAHEThreshold(BinarizationAlgorithm):
	"""
	CLAHE preprocessing followed by adaptive thresholding.
	Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to
	enhance local contrast, then applies threshold. Excellent for low-contrast
	documents and images with uneven illumination.
	CLAHE divides image into tiles and applies histogram equalization with
	contrast limiting to prevent noise amplification.
	Best for: Low-contrast documents, uneven illumination, degraded scans.
	Example:
		>>> method = CLAHEThreshold()
		>>> result = method.binarize(image, clip_limit=2.0, tile_size=8, threshold_method='sauvola')
	"""
	
	def __init__(self):
		super().__init__(
			name="clahe_threshold",
			description="CLAHE preprocessing + adaptive threshold"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'clip_limit': 2.0,       # Contrast limiting (1.0-4.0)
			'tile_size': 8,          # Grid size for CLAHE
			'threshold_method': 'sauvola',  # or 'otsu', 'mean_adaptive'
			'threshold_params': {}   # Parameters for threshold method
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'clip_limit': (1.0, 8.0),
			'tile_size': (4, 16)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		clip_limit = params.get('clip_limit', 2.0)
		tile_size = params.get('tile_size', 8)
		threshold_method = params.get('threshold_method', 'sauvola')
		threshold_params = params.get('threshold_params', {})
		
		# Apply CLAHE
		clahe = cv2.createCLAHE(
			clipLimit=clip_limit,
			tileGridSize=(tile_size, tile_size)
		)
		enhanced = clahe.apply(image)
		
		# Apply thresholding
		if threshold_method == 'otsu':
			_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			threshold_value = _
		elif threshold_method == 'sauvola':
			# Use Sauvola on enhanced image
			binary, threshold_value = self._apply_sauvola(enhanced, **threshold_params)
		elif threshold_method == 'mean_adaptive':
			window_size = threshold_params.get('window_size', 15)
			C = threshold_params.get('C', 5)
			binary = cv2.adaptiveThreshold(
				enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
				cv2.THRESH_BINARY, window_size, C
			)
			threshold_value = None
		else:
			raise ValueError(f"Unknown threshold method: {threshold_method}")
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=threshold_value,
			processing_time=processing_time,
			metadata={
				'clahe_clip_limit': clip_limit,
				'clahe_tile_size': tile_size,
				'threshold_method': threshold_method,
				'enhanced_contrast': float(np.std(enhanced) / np.std(image))
			}
		)
	
	def _apply_sauvola(self, image: np.ndarray, **params) -> Tuple[np.ndarray, float]:
		"""Apply Sauvola thresholding."""
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.2)
		R = params.get('R', 128.0)
		
		if window_size % 2 == 0:
			window_size += 1
		
		image_float = image.astype(np.float32)
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = local_mean_sq - local_mean ** 2
		local_variance = np.maximum(local_variance, 0)
		local_std = np.sqrt(local_variance)
		
		threshold = local_mean * (1.0 + k * (local_std / R - 1.0))
		binary = np.where(image_float > threshold, 255, 0).astype(np.uint8)
		
		return binary, float(np.mean(threshold))


class MultiScaleThreshold(BinarizationAlgorithm):
	"""
	Multi-scale binarization combining multiple resolutions.
	Processes image at multiple scales and combines results. Captures both
	fine details (small scale) and global structure (large scale).
	Algorithm:
	1. Generate image pyramid (multiple resolutions)
	2. Apply binarization at each scale
	3. Combine results using weighted fusion or voting
	Best for: Documents with mixed text sizes, complex layouts.
	Example:
		>>> method = MultiScaleThreshold()
		>>> result = method.binarize(image, scales=[0.5, 1.0, 2.0])
	"""
	
	def __init__(self):
		super().__init__(
			name="multiscale",
			description="Multi-scale binarization with pyramid"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'scales': [0.75, 1.0, 1.5],  # Scale factors
			'method': 'sauvola',          # Base method
			'fusion': 'voting',           # 'voting', 'weighted', 'max'
			'window_size': 25,
			'k': 0.2
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'window_size': (11, 51),
			'k': (0.1, 0.5)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		scales = params.get('scales', [0.75, 1.0, 1.5])
		method = params.get('method', 'sauvola')
		fusion = params.get('fusion', 'voting')
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.2)
		
		h, w = image.shape
		results = []
		
		# Process at each scale
		for scale in scales:
			if scale != 1.0:
				# Resize image
				scaled_h, scaled_w = int(h * scale), int(w * scale)
				scaled_image = cv2.resize(image, (scaled_w, scaled_h))
			else:
				scaled_image = image
			
			# Apply binarization
			if method == 'sauvola':
				scaled_binary = self._apply_sauvola_simple(
					scaled_image, window_size, k
				)
			elif method == 'otsu':
				_, scaled_binary = cv2.threshold(
					scaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
				)
			else:
				raise ValueError(f"Unknown method: {method}")
			
			# Resize back to original size
			if scale != 1.0:
				scaled_binary = cv2.resize(scaled_binary, (w, h))
			
			results.append(scaled_binary)
		
		# Fuse results
		if fusion == 'voting':
			# Majority voting
			stacked = np.stack(results, axis=0)
			binary = np.median(stacked, axis=0).astype(np.uint8)
		elif fusion == 'weighted':
			# Weighted average (prefer middle scale)
			weights = np.array([0.25, 0.5, 0.25])[:len(results)]
			weights = weights / weights.sum()
			binary = np.zeros_like(results[0], dtype=np.float32)
			for w, result in zip(weights, results):
				binary += w * result.astype(np.float32)
			binary = ensure_binary(binary.astype(np.uint8))
		elif fusion == 'max':
			# Take maximum (most permissive)
			binary = np.maximum.reduce(results)
		else:
			raise ValueError(f"Unknown fusion method: {fusion}")
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=None,
			processing_time=processing_time,
			metadata={
				'scales': scales,
				'base_method': method,
				'fusion_method': fusion,
				'num_scales': len(scales)
			}
		)
	
	def _apply_sauvola_simple(
		self, image: np.ndarray, window_size: int, k: float
	) -> np.ndarray:
		"""Simple Sauvola implementation."""
		if window_size % 2 == 0:
			window_size += 1
		
		image_float = image.astype(np.float32)
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = np.maximum(local_mean_sq - local_mean ** 2, 0)
		local_std = np.sqrt(local_variance)
		
		threshold = local_mean * (1.0 + k * (local_std / 128.0 - 1.0))
		return np.where(image_float > threshold, 255, 0).astype(np.uint8)


class GradientFusionThreshold(BinarizationAlgorithm):
	"""
	Gradient-based fusion combining intensity and edge information.
	Combines traditional intensity-based thresholding with edge detection
	to better preserve text boundaries and thin strokes.
	Algorithm:
	1. Compute gradient magnitude (edges)
	2. Apply intensity-based threshold
	3. Fuse using weighted combination or logical operations
	Best for: Thin strokes, low-contrast text, preserving text boundaries.
	Example:
		>>> method = GradientFusionThreshold()
		>>> result = method.binarize(image, gradient_weight=0.3)
	"""
	
	def __init__(self):
		super().__init__(
			name="gradient_fusion",
			description="Gradient-based fusion of intensity and edges"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'intensity_method': 'sauvola',  # Base intensity threshold
			'gradient_method': 'sobel',      # 'sobel', 'scharr', 'canny'
			'gradient_weight': 0.3,          # Weight for gradient (0-1)
			'gradient_threshold': 30,        # Threshold for gradient magnitude
			'window_size': 25,
			'k': 0.2
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'gradient_weight': (0.0, 0.5),
			'gradient_threshold': (10, 100),
			'window_size': (11, 51),
			'k': (0.1, 0.5)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		intensity_method = params.get('intensity_method', 'sauvola')
		gradient_method = params.get('gradient_method', 'sobel')
		gradient_weight = params.get('gradient_weight', 0.3)
		gradient_threshold = params.get('gradient_threshold', 30)
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.2)
		
		# Step 1: Intensity-based thresholding
		if intensity_method == 'sauvola':
			intensity_binary = self._apply_sauvola_simple(image, window_size, k)
		elif intensity_method == 'otsu':
			_, intensity_binary = cv2.threshold(
				image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
			)
		else:
			raise ValueError(f"Unknown intensity method: {intensity_method}")
		
		# Step 2: Gradient-based edge detection
		if gradient_method == 'sobel':
			grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
			grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
			gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
		elif gradient_method == 'scharr':
			grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
			grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
			gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
		elif gradient_method == 'canny':
			edges = cv2.Canny(image, 50, 150)
			gradient_mag = edges.astype(np.float64)
		else:
			raise ValueError(f"Unknown gradient method: {gradient_method}")
		
		# Normalize gradient
		gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
		
		# Threshold gradient
		gradient_binary = np.where(gradient_mag > gradient_threshold, 255, 0).astype(np.uint8)
		
		# Step 3: Fuse intensity and gradient
		# Weighted combination
		intensity_weight = 1.0 - gradient_weight
		fused = (intensity_weight * intensity_binary.astype(np.float32) + 
				gradient_weight * gradient_binary.astype(np.float32))
		
		binary = np.where(fused > 127, 255, 0).astype(np.uint8)
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=None,
			processing_time=processing_time,
			metadata={
				'intensity_method': intensity_method,
				'gradient_method': gradient_method,
				'gradient_weight': gradient_weight,
				'mean_gradient': float(np.mean(gradient_mag))
			}
		)
	
	def _apply_sauvola_simple(
		self, image: np.ndarray, window_size: int, k: float
	) -> np.ndarray:
		"""Simple Sauvola implementation."""
		if window_size % 2 == 0:
			window_size += 1
		
		image_float = image.astype(np.float32)
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = np.maximum(local_mean_sq - local_mean ** 2, 0)
		local_std = np.sqrt(local_variance)
		
		threshold = local_mean * (1.0 + k * (local_std / 128.0 - 1.0))
		return np.where(image_float > threshold, 255, 0).astype(np.uint8)


class HybridThreshold(BinarizationAlgorithm):
	"""
	Hybrid global-adaptive thresholding.
	Combines global and adaptive approaches to get benefits of both:
	- Global method for overall structure
	- Adaptive method for local details
	Algorithm:
	1. Apply global threshold (e.g., Otsu)
	2. Apply adaptive threshold (e.g., Sauvola)
	3. Combine using confidence-based weighting or logical operations
	Best for: Documents with both uniform and varying regions.
	Example:
		>>> method = HybridThreshold()
		>>> result = method.binarize(image, global_method='otsu', adaptive_method='sauvola')
	"""
	
	def __init__(self):
		super().__init__(
			name="hybrid",
			description="Hybrid global-adaptive thresholding"
		)
	
	def get_default_params(self) -> Dict[str, Any]:
		return {
			'global_method': 'otsu',      # Global threshold method
			'adaptive_method': 'sauvola',  # Adaptive threshold method
			'combination': 'weighted',     # 'weighted', 'and', 'or', 'variance'
			'adaptive_weight': 0.7,        # Weight for adaptive (0-1)
			'window_size': 25,
			'k': 0.2
		}
	
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		return {
			'adaptive_weight': (0.0, 1.0),
			'window_size': (11, 51),
			'k': (0.1, 0.5)
		}
	
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		start_time = time.time()
		
		# Validate and preprocess
		image = self.validate_image(image)
		
		# Get parameters
		global_method = params.get('global_method', 'otsu')
		adaptive_method = params.get('adaptive_method', 'sauvola')
		combination = params.get('combination', 'weighted')
		adaptive_weight = params.get('adaptive_weight', 0.7)
		window_size = params.get('window_size', 25)
		k = params.get('k', 0.2)
		
		# Apply global threshold
		if global_method == 'otsu':
			global_threshold, global_binary = cv2.threshold(
				image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
			)
		elif global_method == 'triangle':
			global_threshold = self._triangle_threshold(image)
			global_binary = np.where(image > global_threshold, 255, 0).astype(np.uint8)
		else:
			raise ValueError(f"Unknown global method: {global_method}")
		
		# Apply adaptive threshold
		if adaptive_method == 'sauvola':
			adaptive_binary = self._apply_sauvola_simple(image, window_size, k)
		elif adaptive_method == 'mean_adaptive':
			adaptive_binary = cv2.adaptiveThreshold(
				image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
				cv2.THRESH_BINARY, window_size if window_size % 2 == 1 else window_size + 1, 5
			)
		else:
			raise ValueError(f"Unknown adaptive method: {adaptive_method}")
		
		# Combine methods
		if combination == 'weighted':
			# Weighted average
			global_weight = 1.0 - adaptive_weight
			combined = (global_weight * global_binary.astype(np.float32) +
					   adaptive_weight * adaptive_binary.astype(np.float32))
			binary = np.where(combined > 127, 255, 0).astype(np.uint8)
		elif combination == 'and':
			# Logical AND (more conservative)
			binary = cv2.bitwise_and(global_binary, adaptive_binary)
		elif combination == 'or':
			# Logical OR (more permissive)
			binary = cv2.bitwise_or(global_binary, adaptive_binary)
		elif combination == 'variance':
			# Use local variance to decide which method to trust
			local_var = ndimage.generic_filter(
				image.astype(np.float32), np.var, size=window_size
			)
			# High variance regions → trust adaptive, low variance → trust global
			var_normalized = local_var / (local_var.max() + 1e-10)
			binary = np.where(
				var_normalized > 0.3,  # Threshold for "high variance"
				adaptive_binary,
				global_binary
			).astype(np.uint8)
		else:
			raise ValueError(f"Unknown combination method: {combination}")
		
		processing_time = time.time() - start_time
		
		return BinarizationResult(
			binary_image=binary,
			method=self.name,
			parameters=params,
			threshold=float(global_threshold) if isinstance(global_threshold, (int, float, np.number)) else None,
			processing_time=processing_time,
			metadata={
				'global_method': global_method,
				'adaptive_method': adaptive_method,
				'combination': combination,
				'global_threshold': float(global_threshold) if isinstance(global_threshold, (int, float, np.number)) else None
			}
		)
	
	def _apply_sauvola_simple(
		self, image: np.ndarray, window_size: int, k: float
	) -> np.ndarray:
		"""Simple Sauvola implementation."""
		if window_size % 2 == 0:
			window_size += 1
		
		image_float = image.astype(np.float32)
		local_mean = ndimage.uniform_filter(image_float, size=window_size)
		local_mean_sq = ndimage.uniform_filter(image_float ** 2, size=window_size)
		local_variance = np.maximum(local_mean_sq - local_mean ** 2, 0)
		local_std = np.sqrt(local_variance)
		
		threshold = local_mean * (1.0 + k * (local_std / 128.0 - 1.0))
		return np.where(image_float > threshold, 255, 0).astype(np.uint8)
	
	def _triangle_threshold(self, image: np.ndarray) -> float:
		"""Simple triangle threshold implementation."""
		hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
		peak_idx = np.argmax(hist)
		
		# Determine search direction
		left_weight = np.sum(hist[:peak_idx])
		right_weight = np.sum(hist[peak_idx:])
		
		if left_weight > right_weight:
			search_range = np.arange(0, peak_idx)
		else:
			search_range = np.arange(peak_idx, 256)
		
		if len(search_range) == 0:
			return float(peak_idx)
		
		# Find farthest non-zero point
		non_zero = np.where(hist[search_range] > 0)[0]
		if len(non_zero) == 0:
			return float(peak_idx)
		
		far_idx = search_range[non_zero[-1] if left_weight <= right_weight else non_zero[0]]
		
		# Find max distance to line
		x1, y1 = peak_idx, hist[peak_idx]
		x2, y2 = far_idx, hist[far_idx]
		
		a, b, c = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
		norm = np.sqrt(a**2 + b**2)
		
		if norm == 0:
			return float(peak_idx)
		
		max_dist = 0
		threshold_idx = peak_idx
		
		for idx in search_range:
			if hist[idx] > 0:
				dist = abs(a * idx + b * hist[idx] + c) / norm
				if dist > max_dist:
					max_dist = dist
					threshold_idx = idx
		
		return float(threshold_idx)