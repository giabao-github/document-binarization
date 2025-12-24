"""
Text enhancement operations for improving OCR readability.
This module provides operations to enhance text quality including stroke normalization, character connection, and separation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology

from ..core.base import TextEnhancer


class StrokeEnhancer(TextEnhancer):
	"""
	Enhance and normalize text strokes.
	Provides operations to normalize stroke width, sharpen characters, and improve stroke consistency.
	Example:
		>>> enhancer = StrokeEnhancer()
		>>> enhanced = enhancer.enhance(binary_image, operation='normalize')
	"""
	
	def __init__(self):
		super().__init__(name="stroke_enhancer")
	
	def enhance(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Enhance text strokes.
		Args:
			binary_image: Input binary image
			**params: Enhancement parameters
				- operation: 'normalize', 'thicken', 'thin', 'sharpen'
				- target_width: Target stroke width (default: auto)
				- strength: Enhancement strength 0-1 (default: 0.5)		
		Returns:
			Enhanced binary image
		"""
		operation = params.get('operation', 'normalize')
		target_width = params.get('target_width', None)
		strength = params.get('strength', 0.5)
		
		if operation == 'normalize':
			return self.normalize_stroke_width(binary_image, target_width)
		elif operation == 'thicken':
			return self.thicken_strokes(binary_image, strength)
		elif operation == 'thin':
			return self.thin_strokes(binary_image, strength)
		elif operation == 'sharpen':
			return self.sharpen_text(binary_image, strength)
		else:
			raise ValueError(f"Unknown operation: {operation}")
	
	def normalize_stroke_width(
		self, 
		binary_image: np.ndarray, 
		target_width: Optional[int] = None
	) -> np.ndarray:
		"""
		Normalize stroke width across the image.
		Args:
			binary_image: Input binary image
			target_width: Target stroke width (None = auto-detect)	
		Returns:
			Normalized binary image
		"""
		# Estimate current stroke width
		current_width = self._estimate_stroke_width(binary_image)
		
		if target_width is None:
			# Use median stroke width as target
			target_width = current_width
		
		if target_width == current_width:
			return binary_image
		
		# Adjust stroke width
		if target_width > current_width:
			# Need to thicken
			iterations = (target_width - current_width) // 2
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			result = cv2.dilate(binary_image, kernel, iterations=max(1, iterations))
		else:
			# Need to thin
			iterations = (current_width - target_width) // 2
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			result = cv2.erode(binary_image, kernel, iterations=max(1, iterations))
		
		return result
	
	def thicken_strokes(self, binary_image: np.ndarray, strength: float = 0.5) -> np.ndarray:
		"""
		Thicken text strokes.
		Args:
			binary_image: Input binary image
			strength: Thickening strength (0-1)	
		Returns:
			Thickened binary image
		"""
		iterations = max(1, int(strength * 5))
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		return cv2.dilate(binary_image, kernel, iterations=iterations)
	
	def thin_strokes(self, binary_image: np.ndarray, strength: float = 0.5) -> np.ndarray:
		"""
		Thin text strokes.
		Args:
			binary_image: Input binary image
			strength: Thinning strength (0-1)	
		Returns:
			Thinned binary image
		"""
		if strength > 0.7:
			# Use morphological thinning
			binary_01 = (binary_image < 128).astype(np.uint8)
			thinned = morphology.thin(binary_01)
			return (thinned == 0).astype(np.uint8) * 255
		else:
			# Use erosion
			iterations = max(1, int(strength * 3))
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			return cv2.erode(binary_image, kernel, iterations=iterations)
	
	def sharpen_text(self, binary_image: np.ndarray, strength: float = 0.5) -> np.ndarray:
		"""
		Sharpen text characters.
		Args:
			binary_image: Input binary image
			strength: Sharpening strength (0-1)
		Returns:
			Sharpened binary image
		"""
		# Apply morphological gradient for edge enhancement
		kernel_size = 3
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		
		# Morphological gradient
		gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
		
		# Blend with original
		alpha = strength
		sharpened = cv2.addWeighted(
			binary_image.astype(np.float32), 
			1.0, 
			gradient.astype(np.float32), 
			-alpha, 
			0
		)
		
		# Threshold back to binary
		return (sharpened > 127).astype(np.uint8) * 255
	
	def _estimate_stroke_width(self, binary_image: np.ndarray) -> int:
		"""
		Estimate average stroke width.
		Args:
			binary_image: Input binary image	
		Returns:
			Estimated stroke width in pixels
		"""
		# Distance transform on foreground
		foreground = (binary_image == 0).astype(np.uint8)
		
		if foreground.sum() == 0:
			return 3
		
		dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
		
		# Stroke width ≈ 2 × median distance
		nonzero_dist = dist[dist > 0]
		if len(nonzero_dist) == 0:
			return 3
		
		median_dist = np.median(nonzero_dist)
		stroke_width = int(2 * median_dist)
		
		return max(2, min(stroke_width, 15))


class CharacterConnector(TextEnhancer):
	"""
	Connect broken characters and fix stroke discontinuities.
	Detects and repairs broken characters by analyzing proximity and alignment of stroke fragments.
	Example:
		>>> connector = CharacterConnector()
		>>> fixed = connector.enhance(binary_image, max_gap=5)
	"""
	
	def __init__(self):
		super().__init__(name="character_connector")
	
	def enhance(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Connect broken characters.
		Args:
			binary_image: Input binary image
			**params: Connection parameters
				- max_gap: Maximum gap to bridge (pixels, default: 5)
				- method: 'morphology' or 'skeleton' (default: 'morphology')		
		Returns:
			Binary image with connected characters
		"""
		max_gap = params.get('max_gap', 5)
		method = params.get('method', 'morphology')
		
		if method == 'morphology':
			return self._connect_morphology(binary_image, max_gap)
		elif method == 'skeleton':
			return self._connect_skeleton(binary_image, max_gap)
		else:
			raise ValueError(f"Unknown method: {method}")
	
	def _connect_morphology(self, binary_image: np.ndarray, max_gap: int) -> np.ndarray:
		"""
		Connect using morphological closing.
		Args:
			binary_image: Input binary image
			max_gap: Maximum gap size
		Returns:
			Connected binary image
		"""
		# Use closing with kernel sized to gap
		kernel_size = max_gap * 2 + 1
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
	
	def _connect_skeleton(self, binary_image: np.ndarray, max_gap: int) -> np.ndarray:
		"""
		Connect using skeleton analysis and endpoint matching.
		Args:
			binary_image: Input binary image
			max_gap: Maximum gap size
		Returns:
			Connected binary image
		"""
		# Skeletonize
		foreground = (binary_image < 128).astype(np.uint8)
		skeleton = morphology.skeletonize(foreground)
		
		# Find endpoints
		endpoints = self._find_endpoints(skeleton)
		
		# Connect nearby endpoints
		result = binary_image.copy()
		for i, (y1, x1) in enumerate(endpoints):
			for y2, x2 in endpoints[i+1:]:
				dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
				if dist <= max_gap:
					# Draw line connecting endpoints
					cv2.line(result, (x1, y1), (x2, y2), 0, 2)
		
		return result
	
	def _find_endpoints(self, skeleton: np.ndarray) -> list:
		"""
		Find skeleton endpoints.
		Args:
			skeleton: Binary skeleton	
		Returns:
			List of (y, x) endpoint coordinates
		"""
		# Endpoint has only 1 neighbor
		kernel = np.array([
			[1, 1, 1],
			[1, 0, 1],
			[1, 1, 1]], dtype=np.uint8)
		
		neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
		endpoints_mask = (skeleton > 0) & (neighbor_count == 1)
		
		endpoints = np.argwhere(endpoints_mask)
		return endpoints.tolist()


class CharacterSeparator(TextEnhancer):
	"""
	Separate touching characters.
	Detects and separates characters that are incorrectly joined, improving OCR accuracy on dense text.
	Example:
		>>> separator = CharacterSeparator()
		>>> separated = separator.enhance(binary_image)
	"""
	
	def __init__(self):
		super().__init__(name="character_separator")
	
	def enhance(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Separate touching characters.
		Args:
			binary_image: Input binary image
			**params: Separation parameters
				- method: 'projection', 'watershed', 'erosion' (default: 'projection')
				- sensitivity: Separation sensitivity 0-1 (default: 0.5)		
		Returns:
			Binary image with separated characters
		"""
		method = params.get('method', 'projection')
		sensitivity = params.get('sensitivity', 0.5)
		
		if method == 'projection':
			return self._separate_projection(binary_image, sensitivity)
		elif method == 'watershed':
			return self._separate_watershed(binary_image)
		elif method == 'erosion':
			return self._separate_erosion(binary_image, sensitivity)
		else:
			raise ValueError(f"Unknown method: {method}")
	
	def _separate_projection(self, binary_image: np.ndarray, sensitivity: float) -> np.ndarray:
		"""
		Separate using vertical projection analysis.
		Args:
			binary_image: Input binary image
			sensitivity: Separation sensitivity	
		Returns:
			Separated binary image
		"""
		# Find components
		inverted = cv2.bitwise_not(binary_image)
		num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
			inverted, 
			connectivity=8
		)
		
		result = binary_image.copy()
		
		# Process each component
		for label in range(1, num_labels):
			x = stats[label, cv2.CC_STAT_LEFT]
			y = stats[label, cv2.CC_STAT_TOP]
			w = stats[label, cv2.CC_STAT_WIDTH]
			h = stats[label, cv2.CC_STAT_HEIGHT]
			
			# Skip small components
			if w < 10 or h < 10:
				continue
			
			# Extract component region
			component = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
			
			# Vertical projection
			projection = np.sum(component == 0, axis=0)
			
			# Find minima (potential separation points)
			threshold = np.mean(projection) * (1 - sensitivity)
			minima = np.where(projection < threshold)[0]
			
			if len(minima) > 0:
				# Find gaps
				gaps = np.diff(minima)
				separation_points = minima[:-1][gaps > 1]
				
				# Draw separation lines
				for sep_x in separation_points:
					result[y:y+h, x+sep_x] = 255
		
		return result
	
	def _separate_watershed(self, binary_image: np.ndarray) -> np.ndarray:
		"""
		Separate using watershed segmentation.
		Args:
			binary_image: Input binary image	
		Returns:
			Separated binary image
		"""
		# Distance transform
		foreground = (binary_image == 0).astype(np.uint8)
		dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
		
		# Find sure foreground (peaks)
		_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
		sure_fg = sure_fg.astype(np.uint8)
		
		# Find unknown region
		kernel = np.ones((3,3), np.uint8)
		sure_bg = cv2.dilate(foreground, kernel, iterations=3)
		unknown = cv2.subtract(sure_bg, sure_fg)
		
		# Marker labelling
		_, markers = cv2.connectedComponents(sure_fg)
		markers = markers + 1
		markers[unknown == 255] = 0
		
		# Watershed
		binary_3ch = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
		markers = cv2.watershed(binary_3ch, markers)
		
		# Create result (markers == -1 are boundaries)
		result = binary_image.copy()
		result[markers == -1] = 255
		
		return result
	
	def _separate_erosion(self, binary_image: np.ndarray, sensitivity: float) -> np.ndarray:
		"""
		Separate using erosion.
		Args:
			binary_image: Input binary image
			sensitivity: Erosion strength	
		Returns:
			Separated binary image
		"""
		iterations = max(1, int(sensitivity * 3))
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		return cv2.erode(binary_image, kernel, iterations=iterations)