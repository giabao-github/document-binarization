"""
Connected component analysis and filtering for binary images.
This module provides tools for analyzing and filtering connected components to remove noise, borders, and non-text elements.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from ..core.base import PostProcessor


class ComponentFilter(PostProcessor):
	"""
	Filter connected components by size, aspect ratio, and position.
	Removes components that don't match typical text characteristics.
	Example:
		>>> filter = ComponentFilter()
		>>> cleaned = filter.process(binary_image, min_area=10, max_area=10000)
	"""
	
	def __init__(self):
		super().__init__(name="component_filter")
	
	def process(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Filter connected components.
		Args:
			binary_image: Input binary image (0/255)
			**params: Filtering parameters
				- min_area: Minimum component area (default: 10)
				- max_area: Maximum component area (default: 100000)
				- min_aspect_ratio: Minimum width/height ratio (default: 0.05)
				- max_aspect_ratio: Maximum width/height ratio (default: 20.0)
				- remove_border: Remove components touching borders (default: False)
				- min_solidity: Minimum solidity (area/convex_hull_area) (default: 0.0)
		Returns:
			Filtered binary image
		"""
		min_area = params.get('min_area', 10)
		max_area = params.get('max_area', 100000)
		min_aspect = params.get('min_aspect_ratio', 0.05)
		max_aspect = params.get('max_aspect_ratio', 20.0)
		remove_border = params.get('remove_border', False)
		min_solidity = params.get('min_solidity', 0.0)
		
		# Find connected components
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		# Create output mask
		output = np.zeros_like(binary_image)
		h, w = binary_image.shape
		
		# Filter each component (skip background label 0)
		for label in range(1, num_labels):
			area = stats[label, cv2.CC_STAT_AREA]
			x = stats[label, cv2.CC_STAT_LEFT]
			y = stats[label, cv2.CC_STAT_TOP]
			width = stats[label, cv2.CC_STAT_WIDTH]
			height = stats[label, cv2.CC_STAT_HEIGHT]
			
			# Check area
			if not (min_area <= area <= max_area):
				continue
			
			# Check aspect ratio
			if height > 0:
				aspect_ratio = width / height
				if not (min_aspect <= aspect_ratio <= max_aspect):
					continue
			
			# Check border touching
			if remove_border:
				touches_border = (x == 0 or y == 0 or 
								x + width >= w or y + height >= h)
				if touches_border:
					continue
			
			# Check solidity if required
			if min_solidity > 0:
				component_mask = (labels == label).astype(np.uint8)
				contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				if contours:
					hull = cv2.convexHull(contours[0])
					hull_area = cv2.contourArea(hull)
					if hull_area > 0:
						solidity = area / hull_area
						if solidity < min_solidity:
							continue
			
			# Keep this component
			output[labels == label] = 255
		
		return output
	
	def remove_small_components(self, binary_image: np.ndarray, min_area: int = 10) -> np.ndarray:
		"""
		Remove small noise components.
		Args:
			binary_image: Input binary image
			min_area: Minimum area to keep	
		Returns:
			Filtered image
		"""
		return self.process(binary_image, min_area=min_area)
	
	def remove_large_components(self, binary_image: np.ndarray, max_area: int = 100000) -> np.ndarray:
		"""
		Remove abnormally large components.
		Args:
			binary_image: Input binary image
			max_area: Maximum area to keep	
		Returns:
			Filtered image
		"""
		return self.process(binary_image, max_area=max_area)
	
	def remove_border_components(self, binary_image: np.ndarray, margin: int = 0) -> np.ndarray:
		"""
		Remove components touching image borders.
		Args:
			binary_image: Input binary image
			margin: Additional margin from border (pixels)	
		Returns:
			Filtered image
		"""
		if margin > 0:
			# Create a mask excluding border region
			mask = np.ones_like(binary_image)
			mask[:margin, :] = 0
			mask[-margin:, :] = 0
			mask[:, :margin] = 0
			mask[:, -margin:] = 0
			binary_image = cv2.bitwise_and(binary_image, mask)
		
		return self.process(binary_image, remove_border=True)


class ComponentAnalyzer:
	"""
	Analyze connected components and extract statistics.
	Provides detailed information about components for analysis and debugging.
	Example:
		>>> analyzer = ComponentAnalyzer()
		>>> stats = analyzer.analyze(binary_image)
		>>> print(f"Found {stats['num_components']} components")
	"""
	
	def __init__(self):
		pass
	
	def analyze(self, binary_image: np.ndarray) -> Dict[str, Any]:
		"""
		Analyze connected components in binary image.
		Args:
			binary_image: Input binary image	
		Returns:
			Dictionary with component statistics
		"""
		# Find connected components
		num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
			binary_image, 
			connectivity=8
		)
		
		if num_labels <= 1:
			return {
				'num_components': 0,
				'areas': [],
				'aspect_ratios': [],
				'centroids': [],
				'bounding_boxes': []
			}
		
		# Extract component properties (skip background)
		areas = stats[1:, cv2.CC_STAT_AREA].tolist()
		widths = stats[1:, cv2.CC_STAT_WIDTH]
		heights = stats[1:, cv2.CC_STAT_HEIGHT]
		
		aspect_ratios = []
		for w, h in zip(widths, heights):
			if h > 0:
				aspect_ratios.append(w / h)
			else:
				aspect_ratios.append(0.0)
		
		bounding_boxes = []
		for label in range(1, num_labels):
			x = stats[label, cv2.CC_STAT_LEFT]
			y = stats[label, cv2.CC_STAT_TOP]
			w = stats[label, cv2.CC_STAT_WIDTH]
			h = stats[label, cv2.CC_STAT_HEIGHT]
			bounding_boxes.append((x, y, w, h))
		
		return {
			'num_components': num_labels - 1,
			'areas': areas,
			'aspect_ratios': aspect_ratios,
			'centroids': centroids[1:].tolist(),
			'bounding_boxes': bounding_boxes,
			'mean_area': float(np.mean(areas)) if areas else 0.0,
			'median_area': float(np.median(areas)) if areas else 0.0,
			'mean_aspect_ratio': float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
		}
	
	def get_component_mask(self, binary_image: np.ndarray, component_id: int) -> np.ndarray:
		"""
		Get mask for a specific component.
		Args:
			binary_image: Input binary image
			component_id: Component ID (1-based)	
		Returns:
			Binary mask of the component
		"""
		_, labels, _, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
		return (labels == component_id).astype(np.uint8) * 255
	
	def visualize_components(self, binary_image: np.ndarray, color: bool = True) -> np.ndarray:
		"""
		Create visualization of connected components.
		Args:
			binary_image: Input binary image
			color: Use random colors for each component
		Returns:
			Visualization image (BGR if color=True, grayscale otherwise)
		"""
		num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		if color:
			# Create random colors for each label
			np.random.seed(42)  # For reproducibility
			colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
			colors[0] = [0, 0, 0]  # Background is black
			
			# Map labels to colors
			colored = colors[labels]
			return colored
		else:
			# Grayscale with different intensities
			normalized = (labels * (255 // max(num_labels, 1))).astype(np.uint8)
			return normalized


class BorderRemover(PostProcessor):
	"""
	Remove borders, frames, and margin decorations from documents.
	Detects and removes large rectangular components at image edges that are likely to be borders or frames rather than text.
	Example:
		>>> remover = BorderRemover()
		>>> clean = remover.process(binary_image, border_threshold=0.7)
	"""
	
	def __init__(self):
		super().__init__(name="border_remover")
	
	def process(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Remove border components.
		Args:
			binary_image: Input binary image
			**params: Removal parameters
				- border_threshold: Fraction of edge that must be touched (default: 0.7)
				- min_thickness: Minimum border thickness to remove (default: 5)
				- max_thickness: Maximum border thickness (default: 50)	
		Returns:
			Image with borders removed
		"""
		border_threshold = params.get('border_threshold', 0.7)
		min_thickness = params.get('min_thickness', 5)
		max_thickness = params.get('max_thickness', 50)
		
		h, w = binary_image.shape
		output = binary_image.copy()
		
		# Find connected components
		num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
			binary_image, connectivity=8
		)
		
		# Check each component
		for label in range(1, num_labels):
			x = stats[label, cv2.CC_STAT_LEFT]
			y = stats[label, cv2.CC_STAT_TOP]
			width = stats[label, cv2.CC_STAT_WIDTH]
			height = stats[label, cv2.CC_STAT_HEIGHT]
			area = stats[label, cv2.CC_STAT_AREA]
			
			# Check if component is likely a border
			is_border = False
			
			# Horizontal borders (top/bottom)
			if height >= min_thickness and height <= max_thickness:
				if (y == 0 or y + height >= h) and width > w * border_threshold:
					is_border = True
			
			# Vertical borders (left/right)
			if width >= min_thickness and width <= max_thickness:
				if (x == 0 or x + width >= w) and height > h * border_threshold:
					is_border = True
			
			# Large edge-touching components
			touches_edge = (x == 0 or y == 0 or x + width >= w or y + height >= h)
			if touches_edge and area > 0.05 * h * w:
				# Very large component touching edge, likely border
				is_border = True
			
			# Remove border component
			if is_border:
				output[labels == label] = 255  # Set to background
		
		return output
	
	def remove_rectangular_borders(self, binary_image: np.ndarray, margin: int = 5) -> np.ndarray:
		"""
		Remove rectangular border frames.
		Args:
			binary_image: Input binary image
			margin: Margin to check for borders (pixels)	
		Returns:
			Image with rectangular borders removed
		"""
		h, w = binary_image.shape
		output = binary_image.copy()
		
		# Check top/bottom margins
		for row in [range(margin), range(h - margin, h)]:
			if np.mean(output[row, :] == 0) > 0.8:  # 80% foreground
				output[row, :] = 255  # Remove
		
		# Check left/right margins
		for col in [range(margin), range(w - margin, w)]:
			if np.mean(output[:, col] == 0) > 0.8:
				output[:, col] = 255  # Remove
		
		return output