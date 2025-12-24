"""
Parameter optimization for binarization algorithms.
This module provides automated parameter tuning using various optimization strategies including grid search, random search, and Bayesian optimization.
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
import numpy as np
import time
from dataclasses import dataclass
import json
from pathlib import Path

try:
	import optuna
	OPTUNA_AVAILABLE = True
except ImportError:
	OPTUNA_AVAILABLE = False
	print("Warning: optuna not available. Bayesian optimization disabled.")


@dataclass
class OptimizationResult:
	"""
	Container for optimization results.
	Attributes:
		best_params: Best parameter configuration found
		best_score: Best objective score achieved
		all_trials: List of all trial results
		optimization_time: Total optimization time
		method: Optimization method used
		metadata: Additional information
	"""
	best_params: Dict[str, Any]
	best_score: float
	all_trials: List[Dict[str, Any]]
	optimization_time: float
	method: str
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			'best_params': self.best_params,
			'best_score': self.best_score,
			'num_trials': len(self.all_trials),
			'optimization_time': self.optimization_time,
			'method': self.method,
			**self.metadata
		}
	
	def save(self, path: Path):
		"""Save results to JSON file."""
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, 'w') as f:
			json.dump(self.to_dict(), f, indent=2)


class ParameterOptimizer:
	"""
	Base class for parameter optimization.
	Provides a unified interface for different optimization strategies.
	"""
	
	def __init__(
		self,
		objective_function: Callable,
		param_space: Dict[str, Tuple],
		maximize: bool = False
	):
		"""
		Initialize optimizer.
		Args:
			objective_function: Function to optimize (takes params dict, returns float)
			param_space: Dictionary mapping parameter names to (min, max) tuples
			maximize: If True, maximize objective; if False, minimize
		"""
		self.objective_function = objective_function
		self.param_space = param_space
		self.maximize = maximize
		self.trials = []
	
	def optimize(self, n_trials: int, **kwargs) -> OptimizationResult:
		"""
		Run optimization.
		Args:
			n_trials: Number of trials to run
			**kwargs: Additional optimizer-specific arguments	
		Returns:
			OptimizationResult with best parameters
		"""
		raise NotImplementedError


class GridSearchOptimizer(ParameterOptimizer):
	"""
	Grid search parameter optimization.
	Exhaustively searches all combinations of discrete parameter values.
	Example:
		>>> def objective(params):
		...   return evaluate_binarization(image, params)
		>>> 
		>>> optimizer = GridSearchOptimizer(
		...   objective_function=objective,
		...   param_space={
		...     'window_size': [11, 15, 21, 25],
		...     'k': [0.1, 0.2, 0.3, 0.4]
		...   }
		... )
		>>> result = optimizer.optimize()
	"""
	
	def __init__(
		self,
		objective_function: Callable,
		param_space: Dict[str, List],
		maximize: bool = False
	):
		"""
		Initialize grid search optimizer.
		Args:
			objective_function: Function to optimize
			param_space: Dictionary mapping parameter names to lists of values
			maximize: If True, maximize objective
		"""
		# Convert lists to tuples for parent class
		tuple_space = {k: (min(v), max(v)) for k, v in param_space.items()}
		super().__init__(objective_function, tuple_space, maximize)
		self.grid_space = param_space
	
	def optimize(self, verbose: bool = True) -> OptimizationResult:
		"""
		Run grid search.
		Args:
			verbose: Print progress	
		Returns:
			OptimizationResult
		"""
		start_time = time.time()
		
		# Generate all combinations
		import itertools
		param_names = list(self.grid_space.keys())
		param_values = [self.grid_space[name] for name in param_names]
		combinations = list(itertools.product(*param_values))
		
		best_score = float('-inf') if self.maximize else float('inf')
		best_params = None
		
		if verbose:
			print(f"Grid search: {len(combinations)} combinations")
		
		for i, values in enumerate(combinations):
			params = dict(zip(param_names, values))
			
			try:
				score = self.objective_function(params)
				
				self.trials.append({
					'params': params,
					'score': score,
					'trial_number': i
				})
				
				# Update best
				if self.maximize:
					if score > best_score:
						best_score = score
						best_params = params
				else:
					if score < best_score:
						best_score = score
						best_params = params
				
				if verbose and (i + 1) % max(1, len(combinations) // 10) == 0:
					print(f"Progress: {i+1}/{len(combinations)}, Best: {best_score:.4f}")
			
			except Exception as e:
				if verbose:
					print(f"Trial {i} failed: {e}")
		
		optimization_time = time.time() - start_time
		
		return OptimizationResult(
			best_params=best_params,
			best_score=best_score,
			all_trials=self.trials,
			optimization_time=optimization_time,
			method='grid_search',
			metadata={'total_combinations': len(combinations)}
		)


class RandomSearchOptimizer(ParameterOptimizer):
	"""
	Random search parameter optimization.
	Randomly samples parameter combinations from the search space.
	Often more efficient than grid search for high-dimensional spaces.
	Example:
		>>> optimizer = RandomSearchOptimizer(
		...   objective_function=objective,
		...   param_space={
		...     'window_size': (11, 51),
		...     'k': (0.1, 0.5)
		...   }
		... )
		>>> result = optimizer.optimize(n_trials=100)
	"""
	
	def optimize(
		self, 
		n_trials: int = 100, 
		seed: Optional[int] = None,
		verbose: bool = True
  ) -> OptimizationResult:
		"""
		Run random search.
		Args:
			n_trials: Number of random trials
			seed: Random seed for reproducibility
			verbose: Print progress	
		Returns:
			OptimizationResult
		"""
		if seed is not None:
			np.random.seed(seed)
		
		start_time = time.time()
		
		best_score = float('-inf') if self.maximize else float('inf')
		best_params = None
		
		if verbose:
			print(f"Random search: {n_trials} trials")
		
		for i in range(n_trials):
			# Sample random parameters
			params = {}
			for name, (min_val, max_val) in self.param_space.items():
				if isinstance(min_val, int) and isinstance(max_val, int):
					params[name] = np.random.randint(min_val, max_val + 1)
				else:
					params[name] = np.random.uniform(min_val, max_val)
			
			try:
				score = self.objective_function(params)
				
				self.trials.append({
					'params': params,
					'score': score,
					'trial_number': i
				})
				
				# Update best
				if self.maximize:
					if score > best_score:
						best_score = score
						best_params = params
				else:
					if score < best_score:
						best_score = score
						best_params = params
				
				if verbose and (i + 1) % max(1, n_trials // 10) == 0:
					print(f"Progress: {i+1}/{n_trials}, Best: {best_score:.4f}")
			
			except Exception as e:
				if verbose:
					print(f"Trial {i} failed: {e}")
		
		optimization_time = time.time() - start_time
		
		return OptimizationResult(
			best_params=best_params,
			best_score=best_score,
			all_trials=self.trials,
			optimization_time=optimization_time,
			method='random_search',
			metadata={'seed': seed}
		)


class BayesianOptimizer(ParameterOptimizer):
	"""
	Bayesian optimization using Optuna.
	Uses Tree-structured Parzen Estimator (TPE) for efficient parameter space exploration. Most efficient for expensive objective functions.
	Example:
		>>> optimizer = BayesianOptimizer(
		...   objective_function=objective,
		...   param_space={
		...     'window_size': (11, 51),
		...     'k': (0.1, 0.5)
		...   }
		... )
		>>> result = optimizer.optimize(n_trials=50)
	"""
	
	def __init__(
		self,
		objective_function: Callable,
		param_space: Dict[str, Tuple],
		maximize: bool = False,
		param_types: Optional[Dict[str, str]] = None
	):
		"""
		Initialize Bayesian optimizer.
		Args:
			objective_function: Function to optimize
			param_space: Dictionary mapping parameter names to (min, max) tuples
			maximize: If True, maximize objective
			param_types: Optional dict specifying 'int' or 'float' for each param
		"""
		if not OPTUNA_AVAILABLE:
			raise ImportError("optuna is required for Bayesian optimization")
		
		super().__init__(objective_function, param_space, maximize)
		self.param_types = param_types or {}
	
	def optimize(
		self,
		n_trials: int = 100,
		timeout: Optional[float] = None,
		seed: Optional[int] = None,
		verbose: bool = True
	) -> OptimizationResult:
		"""
		Run Bayesian optimization.
		Args:
			n_trials: Number of trials
			timeout: Maximum optimization time in seconds
			seed: Random seed
			verbose: Print progress	
		Returns:
			OptimizationResult
		"""
		start_time = time.time()
		
		# Create Optuna study
		direction = 'maximize' if self.maximize else 'minimize'
		sampler = optuna.samplers.TPESampler(seed=seed)
		
		study = optuna.create_study(
			direction=direction,
			sampler=sampler
		)
		
		# Define objective wrapper for Optuna
		def optuna_objective(trial):
			params = {}
			for name, (min_val, max_val) in self.param_space.items():
				param_type = self.param_types.get(name, 'float')
				
				if param_type == 'int':
					params[name] = trial.suggest_int(name, int(min_val), int(max_val))
				else:
					params[name] = trial.suggest_float(name, float(min_val), float(max_val))
			
			score = self.objective_function(params)
			
			self.trials.append({
				'params': params,
				'score': score,
				'trial_number': trial.number
			})
			
			return score
		
		# Run optimization
		if verbose:
			print(f"Bayesian optimization: {n_trials} trials")
		
		study.optimize(
			optuna_objective,
			n_trials=n_trials,
			timeout=timeout,
			show_progress_bar=verbose
		)
		
		optimization_time = time.time() - start_time
		
		return OptimizationResult(
			best_params=study.best_params,
			best_score=study.best_value,
			all_trials=self.trials,
			optimization_time=optimization_time,
			method='bayesian',
			metadata={
				'seed': seed,
				'timeout': timeout,
				'n_trials_completed': len(study.trials)
			}
		)


class BinarizationOptimizer:
	"""
	High-level optimizer for binarization algorithms.
	Provides a convenient interface for optimizing binarization parameters based on OCR accuracy or image quality metrics.
	Example:
		>>> from binarization.methods.adaptive_methods import SauvolaThreshold
		>>> from binarization.evaluation.ocr import TesseractOCR
		>>> 
		>>> optimizer = BinarizationOptimizer(
		...   algorithm=SauvolaThreshold(),
		...   images=[img1, img2, img3],
		...   ground_truth_texts=[gt1, gt2, gt3],
		...   objective='cer'  # Minimize Character Error Rate
		... )
		>>> 
		>>> result = optimizer.optimize(
		...   method='bayesian',
		...   n_trials=50
		... )
		>>> 
		>>> print(f"Best params: {result.best_params}")
		>>> print(f"Best CER: {result.best_score:.4f}")
	"""
	
	def __init__(
		self,
		algorithm,
		images: List[np.ndarray],
		ground_truth_texts: Optional[List[str]] = None,
		ground_truth_images: Optional[List[np.ndarray]] = None,
		objective: str = 'cer',
		ocr_lang: str = 'eng'
	):
		"""
		Initialize binarization optimizer.
		Args:
			algorithm: Binarization algorithm instance
			images: List of input images
			ground_truth_texts: Ground truth texts for OCR evaluation
			ground_truth_images: Ground truth binary images
			objective: Objective to optimize ('cer', 'wer', 'psnr', 'ssim', 'iou')
			ocr_lang: OCR language
		"""
		self.algorithm = algorithm
		self.images = images
		self.ground_truth_texts = ground_truth_texts
		self.ground_truth_images = ground_truth_images
		self.objective = objective
		self.ocr_lang = ocr_lang
		
		# Initialize OCR and metrics
		if ground_truth_texts and TESSERACT_AVAILABLE:
			from ..evaluation.ocr import TesseractOCR
			self.ocr = TesseractOCR(lang=ocr_lang)
		
		from ..evaluation.metrics import MetricsCalculator
		self.metrics_calculator = MetricsCalculator()
	
	def _objective_function(self, params: Dict[str, Any]) -> float:
		"""
		Evaluate parameters on dataset.
		Args:
			params: Parameter dictionary	
		Returns:
			Objective score (lower is better for errors, higher for quality)
		"""
		scores = []
		
		for i, image in enumerate(self.images):
			# Binarize with these parameters
			result = self.algorithm.binarize(image, **params)
			binary_image = result.binary_image
			
			# Compute metric
			if self.objective in ['cer', 'wer'] and self.ground_truth_texts:
				# OCR-based metrics
				ocr_result = self.ocr.extract_text(binary_image)
				
				if self.objective == 'cer':
					from ..evaluation.metrics import OCRMetrics
					score = OCRMetrics.character_error_rate(
						self.ground_truth_texts[i],
						ocr_result.text
					)
				else:  # wer
					from ..evaluation.metrics import OCRMetrics
					score = OCRMetrics.word_error_rate(
						self.ground_truth_texts[i],
						ocr_result.text
					)
				
				scores.append(score)
			
			elif self.objective in ['psnr', 'ssim', 'iou'] and self.ground_truth_images:
				# Image-based metrics
				from ..evaluation.metrics import ImageMetrics
				
				if self.objective == 'psnr':
					score = ImageMetrics.psnr(
						self.ground_truth_images[i],
						binary_image
					)
				elif self.objective == 'ssim':
					score = ImageMetrics.ssim(
						self.ground_truth_images[i],
						binary_image
					)
				else:  # iou
					score = ImageMetrics.iou(
						self.ground_truth_images[i],
						binary_image
					)
				
				scores.append(score)
		
		# Return mean score
		return float(np.mean(scores))
	
	def optimize(
		self,
		method: str = 'bayesian',
		n_trials: int = 50,
		param_space: Optional[Dict] = None,
		verbose: bool = True
	) -> OptimizationResult:
		"""
		Optimize algorithm parameters.
		Args:
			method: Optimization method ('grid', 'random', 'bayesian')
			n_trials: Number of trials
			param_space: Custom parameter space (None = use algorithm defaults)
			verbose: Print progress	
		Returns:
			OptimizationResult
		"""
		# Get parameter space
		if param_space is None:
			param_space = self.algorithm.get_param_ranges()
		
		# Determine if maximizing or minimizing
		maximize = self.objective in ['psnr', 'ssim', 'iou', 'confidence']
		
		# Create optimizer
		if method == 'grid':
			# Convert ranges to discrete values for grid search
			grid_space = {}
			for name, (min_val, max_val) in param_space.items():
				if isinstance(min_val, int):
					grid_space[name] = list(range(min_val, max_val + 1, max(1, (max_val - min_val) // 5)))
				else:
					grid_space[name] = list(np.linspace(min_val, max_val, 5))
			
			optimizer = GridSearchOptimizer(
				self._objective_function,
				grid_space,
				maximize
			)
			result = optimizer.optimize(verbose=verbose)
		
		elif method == 'random':
			optimizer = RandomSearchOptimizer(
				self._objective_function,
				param_space,
				maximize
			)
			result = optimizer.optimize(n_trials=n_trials, verbose=verbose)
		
		elif method == 'bayesian':
			# Infer parameter types
			param_types = {}
			for name, (min_val, max_val) in param_space.items():
				param_types[name] = 'int' if isinstance(min_val, int) else 'float'
			
			optimizer = BayesianOptimizer(
				self._objective_function,
				param_space,
				maximize,
				param_types
			)
			result = optimizer.optimize(n_trials=n_trials, verbose=verbose)
		
		else:
			raise ValueError(f"Unknown optimization method: {method}")
		
		return result