#!/usr/bin/env python3
"""
Command-line interface for document binarization.
Usage:
	python scripts/binarize.py --method otsu --input doc.jpg --output result.png
	python scripts/binarize.py --method otsu --input-dir images/ --output-dir results/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from binarization.methods.registry import get_algorithm, list_algorithms
from binarization.core.pipeline import BinarizationPipeline
from binarization.core.config import ConfigLoader, get_default_config


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Document Image Binarization Tool',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Single image with Otsu
  %(prog)s --method otsu --input document.jpg --output result.png
  
  # Batch processing
  %(prog)s --method otsu --input-dir images/ --output-dir results/
  
  # With specific parameters
  %(prog)s --method manual --input doc.jpg --output result.png --threshold 127
  
  # Using config file
  %(prog)s --config config.yaml --input doc.jpg --output result.png
  
  # List available methods
  %(prog)s --list-methods
		"""
	)
	
	# Input/output
	parser.add_argument('-i', '--input', type=str, help='Input image path')
	parser.add_argument('-o', '--output', type=str, help='Output image path')
	parser.add_argument('--input-dir', type=str, help='Input directory for batch processing')
	parser.add_argument('--output-dir', type=str, help='Output directory for batch processing')
	
	# Method selection
	parser.add_argument('-m', '--method', type=str, default='otsu', help='Binarization method (default: otsu)')
	parser.add_argument('--list-methods', action='store_true', help='List all available methods and exit')
	
	# Configuration
	parser.add_argument('-c', '--config', type=str, help='Configuration file (YAML)')
	parser.add_argument('--threshold', type=int, help='Threshold value for manual method (0-255)')
	
	# Post-processing
	parser.add_argument('--no-postprocess', action='store_true', help='Disable post-processing')
	parser.add_argument('--morphology', type=str, help='Morphological operations (comma-separated: opening,closing)')
	
	# Output options
	parser.add_argument('--save-intermediates', action='store_true', help='Save intermediate processing results')
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
	parser.add_argument('--show-threshold', action='store_true', help='Print computed threshold value')
	
	return parser.parse_args()


def list_methods_info():
	"""Print information about all available methods."""
	from binarization.methods.registry import get_registry
	
	registry = get_registry()
	methods = registry.list_algorithms()
	
	print("\n" + "="*60)
	print("Available Binarization Methods")
	print("="*60 + "\n")
	
	for method_name in methods:
		info = registry.get_info(method_name)
		print(f"• {method_name.upper()}")
		print(f"  Description: {info['description']}")
		
		if info['default_params']:
			print(f"  Default parameters:")
			for param, value in info['default_params'].items():
				print(f"    - {param}: {value}")
		
		print()


def process_single_image(
	input_path: Path,
	output_path: Path,
	method_name: str,
	params: dict,
	config: Optional[object] = None,
	verbose: bool = False
) -> dict:
	"""
	Process a single image.
	Args:
		input_path: Input image path
		output_path: Output image path
		method_name: Binarization method name
		params: Method parameters
		config: Pipeline configuration
		verbose: Print verbose output
	Returns:
		Result dictionary
	"""
	if verbose:
		print(f"Processing: {input_path.name}")
	
	# Read image
	image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		print(f"Error: Could not read image {input_path}")
		return None
	
	try:
		if config:
			# Use pipeline
			from binarization.methods.registry import get_registry
			registry = get_registry()
			pipeline = BinarizationPipeline(config, registry._algorithms)
			result = pipeline.run(image)
			binary_image = result['binary_image']
			threshold = result.get('threshold')
			processing_time = result['processing_time']
		else:
			# Use method directly
			method = get_algorithm(method_name)
			result = method.binarize(image, **params)
			binary_image = result.binary_image
			threshold = result.threshold
			processing_time = result.processing_time
		
		# Save result
		output_path.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(output_path), binary_image)
		
		if verbose:
			print(f"  ✓ Saved to: {output_path}")
			print(f"  Threshold: {threshold:.1f}")
			print(f"  Time: {processing_time:.4f}s")
		
		return {
			'input': str(input_path),
			'output': str(output_path),
			'method': method_name,
			'threshold': threshold,
			'time': processing_time,
			'success': True
		}
		
	except Exception as e:
		print(f"Error processing {input_path}: {e}")
		return {
			'input': str(input_path),
			'success': False,
			'error': str(e)
		}


def process_batch(
	input_dir: Path,
	output_dir: Path,
	method_name: str,
	params: dict,
	config: Optional[object] = None,
	verbose: bool = False
) -> List[dict]:
	"""
	Process all images in a directory.
	Args:
		input_dir: Input directory
		output_dir: Output directory
		method_name: Binarization method name
		params: Method parameters
		config: Pipeline configuration
		verbose: Print verbose output
	Returns:
		List of result dictionaries
	"""
	# Find all image files
	extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
	image_paths = []
	for ext in extensions:
		image_paths.extend(input_dir.glob(f'*{ext}'))
		image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
	
	if not image_paths:
		print(f"No images found in {input_dir}")
		return []
	
	print(f"\nFound {len(image_paths)} images")
	print(f"Processing with method: {method_name}")
	print("-" * 60)
	
	results = []
	start_time = time.time()
	
	for i, input_path in enumerate(image_paths, 1):
		if verbose:
			print(f"\n[{i}/{len(image_paths)}] ", end='')
		
		output_path = output_dir / f"{input_path.stem}_binary{input_path.suffix}"
		result = process_single_image(
			input_path, output_path, method_name, params, config, verbose
		)
		
		if result:
			results.append(result)
	
	total_time = time.time() - start_time
	
	# Summary
	print("\n" + "="*60)
	print("Summary")
	print("="*60)
	successful = sum(1 for r in results if r.get('success', False))
	print(f"Processed: {successful}/{len(image_paths)} images")
	print(f"Total time: {total_time:.2f}s")
	print(f"Average time: {total_time/len(image_paths):.4f}s per image")
	
	if successful > 0:
		avg_threshold = np.mean([r['threshold'] for r in results if r.get('success')])
		print(f"Average threshold: {avg_threshold:.1f}")
	
	return results


def main():
	"""Main entry point."""
	args = parse_args()
	
	# List methods and exit
	if args.list_methods:
		list_methods_info()
		return 0
	
	# Validate input
	if not args.input and not args.input_dir:
		print("Error: Either --input or --input-dir must be specified")
		return 1
	
	if args.input and not args.output:
		print("Error: --output must be specified with --input")
		return 1
	
	if args.input_dir and not args.output_dir:
		print("Error: --output-dir must be specified with --input-dir")
		return 1
	
	# Load or create configuration
	if args.config:
		config = ConfigLoader.load_config(Path(args.config))
	else:
		config = get_default_config()
		config.method = args.method
		
		# Apply command-line overrides
		if args.no_postprocess:
			config.post_processing.enabled = False
		
		if args.morphology:
			ops = args.morphology.split(',')
			config.post_processing.morphology = {}
			for op in ops:
				op = op.strip()
				config.post_processing.morphology[op] = {
					'enabled': True,
					'kernel_size': 3
				}
	
	# Method parameters
	params = {}
	if args.threshold is not None:
		params['threshold'] = args.threshold
	
	# Process
	try:
		if args.input:
			# Single image
			input_path = Path(args.input)
			output_path = Path(args.output)
			
			if not input_path.exists():
				print(f"Error: Input file not found: {input_path}")
				return 1
			
			result = process_single_image(
				input_path,
				output_path,
				args.method,
				params,
				config if args.config else None,
				args.verbose or args.show_threshold
			)
			
			if result and result.get('success'):
				if args.show_threshold:
					print(f"\nComputed threshold: {result['threshold']:.1f}")
				return 0
			else:
				return 1
		
		else:
			# Batch processing
			input_dir = Path(args.input_dir)
			output_dir = Path(args.output_dir)
			
			if not input_dir.exists():
				print(f"Error: Input directory not found: {input_dir}")
				return 1
			
			results = process_batch(
				input_dir,
				output_dir,
				args.method,
				params,
				config if args.config else None,
				args.verbose
			)
			
			# Check if any failed
			failed = sum(1 for r in results if not r.get('success', False))
			return 1 if failed > 0 else 0
	
	except KeyboardInterrupt:
		print("\n\nInterrupted by user")
		return 130
	except Exception as e:
		print(f"\nError: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		return 1


if __name__ == '__main__':
	sys.exit(main())