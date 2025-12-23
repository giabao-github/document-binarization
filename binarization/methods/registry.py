"""
Algorithm registry system for automatic discovery and registration.
This module provides a centralized registry for all binarization algorithms,
allowing easy discovery and instantiation by name.
"""

from typing import Dict, List, Type, Optional
import logging

from ..core.base import BinarizationAlgorithm

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
	"""Central registry for binarization algorithms."""
	
	def __init__(self):
		self._algorithms: Dict[str, BinarizationAlgorithm] = {}
		self._algorithm_classes: Dict[str, Type[BinarizationAlgorithm]] = {}
	
	def register(
		self,
		name: str,
		algorithm: BinarizationAlgorithm,
		override: bool = False
	) -> None:
		"""
		Register an algorithm instance.
		Args:
			name: Unique identifier for the algorithm
			algorithm: Algorithm instance
			override: Whether to override existing registration    
		Raises:
			ValueError: If name already registered and override=False
		"""
		if name in self._algorithms and not override:
			raise ValueError(
				f"Algorithm '{name}' already registered. "
				"Use override=True to replace."
			)
		
		self._algorithms[name] = algorithm
		logger.debug(f"Registered algorithm: {name}")

	def register_class(
		self,
		algorithm_class: Type[BinarizationAlgorithm],
		override: bool = False
	) -> None:
		"""
		Register an algorithm class (will be instantiated on get).
		Args:
			algorithm_class: Algorithm class
			override: Whether to override existing registration
		"""
		# Instantiate to get name
		instance = algorithm_class()
		name = instance.name
		
		if name in self._algorithm_classes and not override:
			raise ValueError(
				f"Algorithm class '{name}' already registered. "
				"Use override=True to replace."
			)
		
		self._algorithm_classes[name] = algorithm_class
		logger.debug(f"Registered algorithm class: {name}")
	
	def get(self, name: str) -> BinarizationAlgorithm:
		"""
		Get algorithm instance by name.
		Args:
			name: Algorithm identifier    
		Returns:
			Algorithm instance    
		Raises:
			KeyError: If algorithm not found
		"""
		# Check instances first
		if name in self._algorithms:
			return self._algorithms[name]
		
		# Check classes (instantiate if needed)
		if name in self._algorithm_classes:
			algorithm = self._algorithm_classes[name]()
			# Cache the instance
			self._algorithms[name] = algorithm
			return algorithm
		
		raise KeyError(
			f"Algorithm '{name}' not found. "
			f"Available: {self.list_algorithms()}"
		)
	
	def list_algorithms(self) -> List[str]:
		"""
		List all registered algorithm names.
		Returns:
			List of algorithm names
		"""
		all_names = set(self._algorithms.keys()) | set(self._algorithm_classes.keys())
		return sorted(all_names)
	
	def get_info(self, name: str) -> Dict[str, any]:
		"""
		Get information about an algorithm.
		Args:
			name: Algorithm identifier    
		Returns:
			Dictionary with algorithm information
		"""
		algorithm = self.get(name)
		return {
			'name': algorithm.name,
			'description': algorithm.description,
			'default_params': algorithm.get_default_params(),
			'param_ranges': algorithm.get_param_ranges(),
			'class': algorithm.__class__.__name__
		}
	
	def clear(self) -> None:
		"""Clear all registrations."""
		self._algorithms.clear()
		self._algorithm_classes.clear()
		logger.debug("Cleared algorithm registry")


# Global registry instance
_global_registry = AlgorithmRegistry()


def get_registry() -> AlgorithmRegistry:
	"""
	Get the global algorithm registry.
	Returns:
		Global AlgorithmRegistry instance
	"""
	return _global_registry


def register_algorithm(
	name: str,
	algorithm: BinarizationAlgorithm,
	override: bool = False
) -> None:
	"""
	Register an algorithm in the global registry.
	Args:
		name: Algorithm identifier
		algorithm: Algorithm instance
		override: Whether to override existing registration
	"""
	_global_registry.register(name, algorithm, override)


def register_algorithm_class(
	algorithm_class: Type[BinarizationAlgorithm],
	override: bool = False
) -> None:
	"""Register an algorithm class in the global registry.
	Args:
		algorithm_class: Algorithm class
		override: Whether to override existing registration
	"""
	_global_registry.register_class(algorithm_class, override)


def get_algorithm(name: str) -> BinarizationAlgorithm:
	"""
	Get algorithm from global registry.
	Args:
		name: Algorithm identifier    
	Returns:
		Algorithm instance
	"""
	return _global_registry.get(name)


def list_algorithms() -> List[str]:
	"""
	List all registered algorithms.
	Returns:
		List of algorithm names
	"""
	return _global_registry.list_algorithms()


def register_default_algorithms() -> None:
	"""Register all default algorithms in the global registry."""
	from .global_methods import (
		ManualThreshold,
		OtsuThreshold,
		TriangleThreshold,
		EntropyThreshold,
		MinimumErrorThreshold
	)
	from .adaptive_methods import (
		MeanAdaptiveThreshold,
		GaussianAdaptiveThreshold,
		NiblackThreshold,
		SauvolaThreshold,
		WolfThreshold,
		BradleyThreshold
	)
	from .advanced_methods import (
		CLAHEThreshold,
		MultiScaleThreshold,
		GradientFusionThreshold,
		HybridThreshold
	)
	
	algorithms = [
		# Global methods
		ManualThreshold,
		OtsuThreshold,
		TriangleThreshold,
		EntropyThreshold,
		MinimumErrorThreshold,
		# Adaptive methods
		MeanAdaptiveThreshold,
		GaussianAdaptiveThreshold,
		NiblackThreshold,
		SauvolaThreshold,
		WolfThreshold,
		BradleyThreshold,
		# Advanced methods
		CLAHEThreshold,
		MultiScaleThreshold,
		GradientFusionThreshold,
		HybridThreshold
	]
	
	for algorithm_class in algorithms:
		try:
			register_algorithm_class(algorithm_class, override=True)
		except Exception as e:
			logger.error(f"Failed to register {algorithm_class.__name__}: {e}")


# Auto-register default algorithms on import
register_default_algorithms()