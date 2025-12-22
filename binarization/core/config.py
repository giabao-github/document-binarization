"""
Configuration management for the binarization system.
Handles loading, validation, and merging of configuration files.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import json
from copy import deepcopy


@dataclass
class MethodConfig:
	"""Configuration for a specific binarization method."""
	name: str
	enabled: bool = True
	parameters: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class PostProcessConfig:
	"""Configuration for post-processing operations."""
	enabled: bool = True
	morphology: Dict[str, Any] = field(default_factory=dict)
	component_filtering: Dict[str, Any] = field(default_factory=dict)
	border_removal: bool = False
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class EnhancementConfig:
	"""Configuration for text enhancement operations."""
	enabled: bool = False
	stroke_normalization: bool = False
	sharpening: bool = False
	broken_char_connection: bool = False
	touching_char_separation: bool = False
	parameters: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class EvaluationConfig:
	"""Configuration for evaluation."""
	enabled: bool = True
	ocr_enabled: bool = True
	tesseract_lang: str = "eng"
	tesseract_config: str = "--psm 6"
	compute_metrics: List[str] = field(default_factory=lambda: ["cer", "wer", "confidence"])
	ground_truth_path: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class PipelineConfig:
	"""Complete pipeline configuration."""
	# Input/output settings
	input_path: Optional[str] = None
	output_path: Optional[str] = None
	
	# Method selection
	method: str = "otsu"
	method_params: Dict[str, Any] = field(default_factory=dict)
	
	# Pipeline stages
	post_processing: PostProcessConfig = field(default_factory=PostProcessConfig)
	enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
	evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
	
	# Performance settings
	batch_size: int = 1
	num_workers: int = 1
	use_gpu: bool = False
	
	# Logging
	log_level: str = "INFO"
	save_intermediates: bool = False
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)
	
	@classmethod
	def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
		"""Create PipelineConfig from dictionary."""
		config_dict = deepcopy(config_dict)
		
		# Handle nested configs
		if 'post_processing' in config_dict:
			config_dict['post_processing'] = PostProcessConfig(**config_dict['post_processing'])
		
		if 'enhancement' in config_dict:
			config_dict['enhancement'] = EnhancementConfig(**config_dict['enhancement'])
		
		if 'evaluation' in config_dict:
			config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
		
		return cls(**config_dict)


class ConfigLoader:
	"""Load and manage configuration files."""
	
	@staticmethod
	def load_yaml(path: Path) -> Dict[str, Any]:
		"""
		Load configuration from YAML file.
		Args:
			path: Path to YAML file			
		Returns:
			Configuration dictionary
		"""
		with open(path, 'r') as f:
			return yaml.safe_load(f)
	
	@staticmethod
	def load_json(path: Path) -> Dict[str, Any]:
		"""
		Load configuration from JSON file.
		Args:
			path: Path to JSON file			
		Returns:
			Configuration dictionary
		"""
		with open(path, 'r') as f:
			return json.load(f)
	
	@staticmethod
	def save_yaml(config: Dict[str, Any], path: Path) -> None:
		"""
		Save configuration to YAML file.
		Args:
			config: Configuration dictionary
			path: Output path
		"""
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, 'w') as f:
			yaml.dump(config, f, default_flow_style=False, sort_keys=False)
	
	@staticmethod
	def save_json(config: Dict[str, Any], path: Path) -> None:
		"""
		Save configuration to JSON file.
		Args:
			config: Configuration dictionary
			path: Output path
		"""
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, 'w') as f:
			json.dump(config, f, indent=2)
	
	@classmethod
	def load_config(cls, path: Path) -> PipelineConfig:
		"""
		Load PipelineConfig from file.
		Args:
			path: Path to config file (YAML or JSON)			
		Returns:
			PipelineConfig object
		"""
		path = Path(path)
		
		if path.suffix in ['.yaml', '.yml']:
			config_dict = cls.load_yaml(path)
		elif path.suffix == '.json':
			config_dict = cls.load_json(path)
		else:
			raise ValueError(f"Unsupported config format: {path.suffix}")
		
		return PipelineConfig.from_dict(config_dict)
	
	@classmethod
	def merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Merge two configuration dictionaries.
		Args:
			base: Base configuration
			override: Override configuration			
		Returns:
			Merged configuration
		"""
		result = deepcopy(base)
		
		for key, value in override.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = cls.merge_configs(result[key], value)
			else:
				result[key] = deepcopy(value)
		
		return result


def get_default_config() -> PipelineConfig:
	"""
	Get default pipeline configuration.
	Returns:
		Default PipelineConfig
	"""
	return PipelineConfig(
		method="otsu",
		method_params={},
		post_processing=PostProcessConfig(
			enabled=True,
			morphology={
				"opening": {"enabled": True, "kernel_size": 3},
				"closing": {"enabled": True, "kernel_size": 3}
			},
			component_filtering={
				"min_area": 10,
				"max_area": 100000,
				"min_aspect_ratio": 0.1,
				"max_aspect_ratio": 10.0
			},
			border_removal=False
		),
		enhancement=EnhancementConfig(
			enabled=False
		),
		evaluation=EvaluationConfig(
			enabled=True,
			ocr_enabled=True,
			tesseract_lang="eng",
			compute_metrics=["cer", "wer", "confidence"]
		),
		batch_size=1,
		num_workers=1,
		log_level="INFO"
	)


# Example default configuration as YAML string
DEFAULT_CONFIG_YAML = """
# Default Binarization Pipeline Configuration

# Method selection
method: "otsu"  # Available: otsu, sauvola, niblack, clahe_otsu, etc.
method_params: {}

# Post-processing configuration
post_processing:
  enabled: true
  morphology:
	opening:
		enabled: true
		kernel_size: 3
		kernel_shape: "rect"  # rect, ellipse, cross
	closing:
		enabled: true
		kernel_size: 3
		kernel_shape: "rect"
  component_filtering:
	enabled: true
	min_area: 10
	max_area: 100000
	min_aspect_ratio: 0.1
	max_aspect_ratio: 10.0
  border_removal: false

# Text enhancement configuration
enhancement:
  enabled: false
  stroke_normalization: false
  sharpening: false
  broken_char_connection: false
  touching_char_separation: false
  parameters: {}

# Evaluation configuration
evaluation:
  enabled: true
  ocr_enabled: true
  tesseract_lang: "eng"
  tesseract_config: "--psm 6"
  compute_metrics:
	- cer
	- wer
	- confidence
  ground_truth_path: null

# Performance settings
batch_size: 1
num_workers: 1
use_gpu: false

# Logging
log_level: "INFO"
save_intermediates: false
"""