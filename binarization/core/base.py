"""
Core base classes and interfaces for binarization algorithms.
This module defines the abstract base classes that all binarization methods must implement, ensuring consistent interfaces across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from enum import Enum


class BinarizationMethod(Enum):
	"""Enumeration of available binarization methods."""
	# Global methods
	MANUAL = "manual"
	OTSU = "otsu"
	TRIANGLE = "triangle"
	ENTROPY = "entropy"
	MINIMUM_ERROR = "minimum_error"
	
	# Adaptive methods
	MEAN_ADAPTIVE = "mean_adaptive"
	GAUSSIAN_ADAPTIVE = "gaussian_adaptive"
	SAUVOLA = "sauvola"
	NIBLACK = "niblack"
	WOLF = "wolf"
	BRADLEY = "bradley"
	
	# Advanced methods
	CLAHE_OTSU = "clahe_otsu"
	SWT_GUIDED = "swt_guided"
	GRADIENT_FUSION = "gradient_fusion"
	HYBRID = "hybrid"


@dataclass
class BinarizationResult:
	"""
	Container for binarization results and metadata.
	Attributes:
		binary_image: Binary output image (0/255)
		method: Name of the method used
		parameters: Parameters used for binarization
		threshold: Computed threshold value(s)
		processing_time: Time taken in seconds
		metadata: Additional algorithm-specific metadata
	"""
	binary_image: np.ndarray
	method: str
	parameters: Dict[str, Any]
	threshold: Optional[float] = None
	processing_time: float = 0.0
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def __post_init__(self):
		"""Validate the binary image."""
		if self.binary_image.dtype != np.uint8:
			raise ValueError(f"Binary image must be uint8, got {self.binary_image.dtype}")
		
		unique_vals = np.unique(self.binary_image)
		if not np.array_equal(unique_vals, [0, 255]) and \
			not np.array_equal(unique_vals, [0]) and \
			not np.array_equal(unique_vals, [255]):
			raise ValueError(f"Binary image must contain only 0 and 255, got {unique_vals}")


class BinarizationAlgorithm(ABC):
	"""
	Abstract base class for all binarization algorithms.
	All binarization methods must inherit from this class and implement
	the binarize method. This ensures a consistent interface across all
	algorithms in the system.
	"""
  
	def __init__(self, name: str, description: str = ""):
		"""Initialize the algorithm.
		
		Args:
			name: Unique name for the algorithm
			description: Human-readable description
		"""
		self.name = name
		self.description = description
		self._default_params = {}
	
	@abstractmethod
	def binarize(self, image: np.ndarray, **params) -> BinarizationResult:
		"""
		Apply binarization to the input image.
		Args:
			image: Input grayscale image (uint8 or float)
			**params: Algorithm-specific parameters
		Returns:
			BinarizationResult containing binary image and metadata
		Raises:
			ValueError: If image format is invalid
		"""
		pass
	
	@abstractmethod
	def get_default_params(self) -> Dict[str, Any]:
		"""
		Return default parameters for this algorithm.
		Returns:
			Dictionary of parameter names and default values
		"""
		pass
	
	@abstractmethod
	def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
		"""
		Return parameter ranges for optimization.
		Returns:
			Dictionary mapping parameter names to (min, max) tuples
		"""
		pass
	
	def validate_image(self, image: np.ndarray) -> np.ndarray:
		"""
		Validate and preprocess input image.
		Args:
			image: Input image
		Returns:
			Validated grayscale image as uint8
		Raises:
			ValueError: If image is invalid
		"""
		if image is None or image.size == 0:
			raise ValueError("Input image is empty or None")
		
		# Handle different input types
		if len(image.shape) == 3:
			# Convert RGB/BGR to grayscale
			if image.shape[2] == 3:
				# Use luminosity method: 0.299*R + 0.587*G + 0.114*B
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			elif image.shape[2] == 4:
				# RGBA - use only RGB channels
				image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
		
		# Convert to uint8 if needed
		if image.dtype == np.float32 or image.dtype == np.float64:
			if image.max() <= 1.0:
				image = (image * 255).astype(np.uint8)
			else:
				image = image.astype(np.uint8)
		elif image.dtype == np.uint16:
			# Scale 16-bit to 8-bit
			image = (image // 256).astype(np.uint8)
		elif image.dtype != np.uint8:
			image = image.astype(np.uint8)
		
		return image
	
	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(name='{self.name}')"


class PostProcessor(ABC):
	"""Abstract base class for post-processing operations."""
	
	def __init__(self, name: str):
		self.name = name
	
	@abstractmethod
	def process(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Apply post-processing to binary image.
		Args:
			binary_image: Input binary image (0/255)
			**params: Operation-specific parameters
		Returns:
			Processed binary image
		"""
		pass


class TextEnhancer(ABC):
	"""Abstract base class for text enhancement operations."""
	
	def __init__(self, name: str):
		self.name = name
	
	@abstractmethod
	def enhance(self, binary_image: np.ndarray, **params) -> np.ndarray:
		"""
		Apply text enhancement to binary image.
		Args:
			binary_image: Input binary image (0/255)
			**params: Enhancement-specific parameters
		Returns:
			Enhanced binary image
		"""
		pass


def ensure_binary(image: np.ndarray) -> np.ndarray:
	"""
	Ensure image is binary (0/255) uint8.
	Args:
		image: Input image
	Returns:
		Binary image with values 0 and 255
	"""
	if image.dtype != np.uint8:
		image = image.astype(np.uint8)
	
	# Threshold to ensure only 0 and 255
	return np.where(image > 127, 255, 0).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
	"""
	Normalize any image to uint8 range [0, 255].
	Args:
		image: Input image of any type		
	Returns:
		Normalized uint8 image
	"""
	if image.dtype == np.uint8:
		return image
	
	# Handle float images
	if image.dtype in [np.float32, np.float64]:
		if image.max() <= 1.0:
			return (image * 255).astype(np.uint8)
		else:
			# Normalize to 0-255 range
			img_min, img_max = image.min(), image.max()
			if img_max > img_min:
				return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
			else:
				return np.zeros_like(image, dtype=np.uint8)
	
	# Handle uint16
	if image.dtype == np.uint16:
		return (image // 256).astype(np.uint8)
	
	# Default conversion
	return image.astype(np.uint8)