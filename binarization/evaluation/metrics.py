"""
Comprehensive metrics for evaluating binarization quality.
This module provides various metrics including OCR-based metrics, image quality metrics, and structural similarity measures.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings


class ImageQualityMetrics:
	"""
	Image quality metrics for binarization evaluation.
	Provides PSNR, SSIM, MSE, and other image quality measures.
	Example:
		>>> metrics = ImageQualityMetrics()
		>>> quality = metrics.compute_all(binary_result, ground_truth)
		>>> print(f"PSNR: {quality['psnr']:.2f} dB")
	"""
	
	def __init__(self):
		pass
	
	def compute_all(self, image: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
		"""
		Compute all image quality metrics.
		Args:
			image: Result binary image
			reference: Ground truth binary image
		Returns:
			Dictionary with all metrics
		"""
		# Ensure same size
		if image.shape != reference.shape:
			warnings.warn("Image shapes don't match, resizing reference", stacklevel=2)
			reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
		
		metrics = {
			'psnr': self.compute_psnr(image, reference),
			'ssim': self.compute_ssim(image, reference),
			'mse': self.compute_mse(image, reference),
			'mae': self.compute_mae(image, reference),
			'iou': self.compute_iou(image, reference),
			'dice': self.compute_dice(image, reference),
			'pixel_accuracy': self.compute_pixel_accuracy(image, reference)
		}
		
		return metrics
	
	def compute_psnr(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Peak Signal-to-Noise Ratio (PSNR).
		Higher is better. Typically 20-50 dB for good quality.
		Args:
			image: Result image
			reference: Ground truth image	
		Returns:
			PSNR in decibels
		"""
		try:
			return float(psnr(reference, image, data_range=255))
		except (ValueError, ZeroDivisionError) as e:
			warnings.warn(f"PSNR computation failed: {e}", stacklevel=2)
			return 0.0
	
	def compute_ssim(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Structural Similarity Index (SSIM).
		Range: [-1, 1], higher is better. 1 = identical.
		Args:
			image: Result image
			reference: Ground truth image	
		Returns:
			SSIM score
		"""
		try:
			return float(ssim(reference, image, data_range=255))
		except (ValueError, RuntimeError) as e:
			warnings.warn(f"SSIM computation failed: {e}", stacklevel=2)
			return 0.0
	
	def compute_mse(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Mean Squared Error (MSE).
		Lower is better. 0 = identical.
		Args:
			image: Result image
			reference: Ground truth image
		Returns:
			MSE value
		"""
		return float(np.mean((image.astype(float) - reference.astype(float)) ** 2))
	
	def compute_mae(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Mean Absolute Error (MAE).
		Lower is better. 0 = identical.
		Args:
			image: Result image
			reference: Ground truth image	
		Returns:
			MAE value
		"""
		return float(np.mean(np.abs(image.astype(float) - reference.astype(float))))
	
	def compute_iou(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Intersection over Union (IoU) for foreground.
		Range: [0, 1], higher is better. 1 = perfect overlap.
		Also known as Jaccard index.
		Args:
			image: Result binary image
			reference: Ground truth binary image	
		Returns:
			IoU score
		"""
		# Foreground is black (0)
		pred_fg = (image == 0)
		ref_fg = (reference == 0)
		
		intersection = np.logical_and(pred_fg, ref_fg).sum()
		union = np.logical_or(pred_fg, ref_fg).sum()
		
		if union == 0:
			return 1.0 if intersection == 0 else 0.0
		
		return float(intersection / union)
	
	def compute_dice(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute Dice coefficient (F1 score at pixel level).
		Range: [0, 1], higher is better. 1 = perfect match.
		Args:
			image: Result binary image
			reference: Ground truth binary image	
		Returns:
			Dice coefficient
		"""
		# Foreground is black (0)
		pred_fg = (image == 0)
		ref_fg = (reference == 0)
		
		intersection = np.logical_and(pred_fg, ref_fg).sum()
		
		if pred_fg.sum() + ref_fg.sum() == 0:
			return 1.0
		
		return float(2.0 * intersection / (pred_fg.sum() + ref_fg.sum()))
	
	def compute_pixel_accuracy(self, image: np.ndarray, reference: np.ndarray) -> float:
		"""
		Compute pixel-wise accuracy.
		Range: [0, 1], higher is better. 1 = all pixels match.
		Args:
			image: Result image
			reference: Ground truth image
		Returns:
			Pixel accuracy
		"""
		correct = (image == reference).sum()
		total = image.size
		return float(correct / total)


class BinarizationMetrics:
	"""
	Metrics specifically for binarization quality.
	Includes foreground/background specific metrics and
	text-focused quality measures.
	Example:
		>>> metrics = BinarizationMetrics()
		>>> quality = metrics.evaluate(binary_result, ground_truth)
	"""
	
	def __init__(self):
		self.image_metrics = ImageQualityMetrics()
	
	def evaluate(self, image: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
		"""
		Evaluate binarization quality.
		Args:
			image: Result binary image
			reference: Ground truth binary image
		Returns:
			Dictionary with evaluation metrics
		"""
		# Basic image quality metrics
		basic_metrics = self.image_metrics.compute_all(image, reference)

		# Ensure same size for subsequent metrics (reference may have been resized above)
		if image.shape != reference.shape:
			reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
		
		# Foreground/background specific metrics
		fg_bg_metrics = self._compute_fg_bg_metrics(image, reference)
		
		# Edge preservation
		edge_metrics = self._compute_edge_metrics(image, reference)
		
		# Combine all metrics
		return {
			**basic_metrics,
			**fg_bg_metrics,
			**edge_metrics
		}
	
	def _compute_fg_bg_metrics(self, image: np.ndarray,	reference: np.ndarray) -> Dict[str, float]:
		"""
		Compute foreground/background specific metrics.
		Args:
			image: Result image
			reference: Ground truth image	
		Returns:
			Dictionary with FG/BG metrics
		"""
		# Foreground is black (0), background is white (255)
		pred_fg = (image == 0)
		pred_bg = (image == 255)
		ref_fg = (reference == 0)
		ref_bg = (reference == 255)
		
		# True positives, false positives, etc.
		tp_fg = np.logical_and(pred_fg, ref_fg).sum()
		fp_fg = np.logical_and(pred_fg, ref_bg).sum()
		tn_fg = np.logical_and(pred_bg, ref_bg).sum()
		fn_fg = np.logical_and(pred_bg, ref_fg).sum()
		
		# Precision, recall, F1 for foreground
		precision_fg = tp_fg / (tp_fg + fp_fg) if (tp_fg + fp_fg) > 0 else 0.0
		recall_fg = tp_fg / (tp_fg + fn_fg) if (tp_fg + fn_fg) > 0 else 0.0
		f1_fg = 2 * (precision_fg * recall_fg) / (precision_fg + recall_fg) \
				if (precision_fg + recall_fg) > 0 else 0.0
		
		return {
			'foreground_precision': float(precision_fg),
			'foreground_recall': float(recall_fg),
			'foreground_f1': float(f1_fg),
			'false_positive_rate': float(fp_fg / max(fp_fg + tn_fg, 1)),
			'false_negative_rate': float(fn_fg / max(fn_fg + tp_fg, 1))
		}
	
	def _compute_edge_metrics(self, image: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
		"""
		Compute edge preservation metrics.
		Args:
			image: Result image
			reference: Ground truth image	
		Returns:
			Dictionary with edge metrics
		"""
		# Compute edges
		pred_edges = cv2.Canny(image, 50, 150)
		ref_edges = cv2.Canny(reference, 50, 150)
		
		# Edge overlap
		edge_tp = np.logical_and(pred_edges > 0, ref_edges > 0).sum()
		edge_fp = np.logical_and(pred_edges > 0, ref_edges == 0).sum()
		edge_fn = np.logical_and(pred_edges == 0, ref_edges > 0).sum()
		
		edge_precision = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
		edge_recall = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
		edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall) \
			if (edge_precision + edge_recall) > 0 else 0.0
		
		return {
			'edge_precision': float(edge_precision),
			'edge_recall': float(edge_recall),
			'edge_f1': float(edge_f1)
		}


class PerformanceMetrics:
	"""
	Performance metrics for algorithms.
	Tracks processing time, memory usage, and computational efficiency.
	Example:
		>>> metrics = PerformanceMetrics()
		>>> stats = metrics.compute(processing_time, image_size)
	"""
	
	def __init__(self):
		pass
	
	def compute(self, processing_time: float, image_shape: Tuple[int, int]) -> Dict[str, Any]:
		"""
		Compute performance metrics.
		Args:
			processing_time: Time taken in seconds
			image_shape: (height, width) of image
		Returns:
			Dictionary with performance metrics
		"""
		h, w = image_shape
		num_pixels = h * w
		megapixels = num_pixels / 1_000_000
		
		return {
			'processing_time_seconds': processing_time,
			'processing_time_ms': processing_time * 1000,
			'megapixels': megapixels,
			'pixels_per_second': num_pixels / max(processing_time, 0.001),
			'megapixels_per_second': megapixels / max(processing_time, 0.001),
			'time_per_megapixel_ms': (processing_time * 1000) / max(megapixels, 0.001)
		}


class CompositeMetric:
	"""
	Composite metric combining multiple evaluation criteria.
	Provides a single score by weighting different metric types.
	Example:
		>>> metric = CompositeMetric(ocr_weight=0.5, image_weight=0.3, speed_weight=0.2)
		>>> score = metric.compute(ocr_metrics, image_metrics, perf_metrics)
	"""
	
	def __init__(self, ocr_weight: float = 0.5, image_weight: float = 0.3, speed_weight: float = 0.2):
		"""
		Initialize composite metric.
		Args:
			ocr_weight: Weight for OCR accuracy (0-1)
			image_weight: Weight for image quality (0-1)
			speed_weight: Weight for processing speed (0-1)
		"""
		total = ocr_weight + image_weight + speed_weight
		if total == 0:
			raise ValueError("At least one weight must be non-zero")
		self.ocr_weight = ocr_weight / total
		self.image_weight = image_weight / total
		self.speed_weight = speed_weight / total
	
	def compute(
		self, 
		ocr_metrics: Optional[Dict[str, float]] = None,
		image_metrics: Optional[Dict[str, float]] = None,
		performance_metrics: Optional[Dict[str, Any]] = None
	) -> Dict[str, float]:
		"""
		Compute composite score.
		Args:
			ocr_metrics: OCR evaluation metrics
			image_metrics: Image quality metrics
			performance_metrics: Performance metrics
		Returns:
			Dictionary with composite scores
		"""
		scores = []
		weights = []
		components = {}
		
		# OCR score (higher character accuracy is better)
		if ocr_metrics and self.ocr_weight > 0:
			ocr_score = ocr_metrics.get('character_accuracy', 0.0)
			scores.append(ocr_score)
			weights.append(self.ocr_weight)
			components['ocr_component'] = ocr_score * self.ocr_weight
		
		# Image quality score (average of normalized metrics)
		if image_metrics and self.image_weight > 0:
			# Normalize metrics to [0, 1] range
			iou = image_metrics.get('iou', 0.0)
			ssim_norm = (image_metrics.get('ssim', 0.0) + 1) / 2  # SSIM is [-1, 1]
			image_score = (iou + ssim_norm) / 2
			scores.append(image_score)
			weights.append(self.image_weight)
			components['image_component'] = image_score * self.image_weight
		
		# Speed score (faster is better, normalized)
		if performance_metrics and self.speed_weight > 0:
			# Assume 1 second per megapixel is baseline (score=0.5)
			time_per_mp = performance_metrics.get('time_per_megapixel_ms', 1000)
			speed_score = np.clip(1.0 - (time_per_mp / 2000), 0, 1)
			scores.append(speed_score)
			weights.append(self.speed_weight)
			components['speed_component'] = speed_score * self.speed_weight
		
		# Weighted average
		if scores:
			total_weight = sum(weights)
			composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
		else:
			composite = 0.0
		
		return {
			'composite_score': composite,
			**components
		}