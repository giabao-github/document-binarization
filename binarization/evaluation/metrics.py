"""
Metrics for evaluating binarization quality.
This module provides various metrics to assess binarization quality including OCR accuracy metrics and image quality metrics.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
from dataclasses import dataclass
import difflib


@dataclass
class EvaluationMetrics:
	"""
	Container for evaluation metrics.
	Attributes:
		cer: Character Error Rate (0-1, lower is better)
		wer: Word Error Rate (0-1, lower is better)
		precision: Precision (0-1, higher is better)
		recall: Recall (0-1, higher is better)
		f1_score: F1 score (0-1, higher is better)
		ocr_confidence: Mean OCR confidence (0-100)
		psnr: Peak Signal-to-Noise Ratio (higher is better)
		ssim: Structural Similarity Index (0-1, higher is better)
		iou: Intersection over Union (0-1, higher is better)
		metadata: Additional metrics
	"""
	cer: Optional[float] = None
	wer: Optional[float] = None
	precision: Optional[float] = None
	recall: Optional[float] = None
	f1_score: Optional[float] = None
	ocr_confidence: Optional[float] = None
	psnr: Optional[float] = None
	ssim: Optional[float] = None
	iou: Optional[float] = None
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			'cer': self.cer,
			'wer': self.wer,
			'precision': self.precision,
			'recall': self.recall,
			'f1_score': self.f1_score,
			'ocr_confidence': self.ocr_confidence,
			'psnr': self.psnr,
			'ssim': self.ssim,
			'iou': self.iou,
			**self.metadata
		}


class OCRMetrics:
	"""
	OCR-based accuracy metrics.
	Computes Character Error Rate (CER), Word Error Rate (WER), and related metrics by comparing OCR output to ground truth text.
	"""
	
	@staticmethod
	def character_error_rate(reference: str, hypothesis: str) -> float:
		"""
		Compute Character Error Rate (CER).
		CER = (Substitutions + Deletions + Insertions) / Total Characters
		Args:
			reference: Ground truth text
			hypothesis: OCR extracted text
		Returns:
			CER value (0-1, lower is better)
		"""
		if len(reference) == 0:
			return 0.0 if len(hypothesis) == 0 else 1.0
		
		# Use Levenshtein distance (edit distance)
		distance = OCRMetrics._levenshtein_distance(reference, hypothesis)
		cer = distance / len(reference)
		
		return min(cer, 1.0)
	
	@staticmethod
	def word_error_rate(reference: str, hypothesis: str) -> float:
		"""
		Compute Word Error Rate (WER).
		WER = (Substitutions + Deletions + Insertions) / Total Words
		Args:
			reference: Ground truth text
			hypothesis: OCR extracted text
		Returns:
			WER value (0-1, lower is better)
		"""
		ref_words = reference.split()
		hyp_words = hypothesis.split()
		
		if len(ref_words) == 0:
			return 0.0 if len(hyp_words) == 0 else 1.0
		
		distance = OCRMetrics._levenshtein_distance(ref_words, hyp_words)
		wer = distance / len(ref_words)
		
		return min(wer, 1.0)
	
	@staticmethod
	def accuracy(reference: str, hypothesis: str) -> float:
		"""
		Compute recognition accuracy.
		Accuracy = 1 - CER
		Args:
			reference: Ground truth text
			hypothesis: OCR extracted text		
		Returns:
			Accuracy (0-1, higher is better)
		"""
		cer = OCRMetrics.character_error_rate(reference, hypothesis)
		return max(0.0, 1.0 - cer)
	
	@staticmethod
	def sequence_alignment(reference: str, hypothesis: str) -> Tuple[int, int, int]:
		"""
		Align sequences and count operations.
		Args:
			reference: Ground truth text
			hypothesis: OCR extracted text		
		Returns:
			Tuple of (substitutions, deletions, insertions)
		"""
		sm = difflib.SequenceMatcher(None, reference, hypothesis)
		
		substitutions = 0
		deletions = 0
		insertions = 0
		
		for tag, i1, i2, j1, j2 in sm.get_opcodes():
			if tag == 'replace':
				substitutions += max(i2 - i1, j2 - j1)
			elif tag == 'delete':
				deletions += i2 - i1
			elif tag == 'insert':
				insertions += j2 - j1
		
		return substitutions, deletions, insertions
	
	@staticmethod
	def _levenshtein_distance(s1, s2) -> int:
		"""
		Compute Levenshtein distance between two sequences.
		Args:
			s1: First sequence (string or list)
			s2: Second sequence (string or list)		
		Returns:
			Edit distance
		"""
		if len(s1) < len(s2):
			return OCRMetrics._levenshtein_distance(s2, s1)
		
		if len(s2) == 0:
			return len(s1)
		
		previous_row = range(len(s2) + 1)
		for i, c1 in enumerate(s1):
			current_row = [i + 1]
			for j, c2 in enumerate(s2):
				# Cost of insertions, deletions, substitutions
				insertions = previous_row[j + 1] + 1
				deletions = current_row[j] + 1
				substitutions = previous_row[j] + (c1 != c2)
				current_row.append(min(insertions, deletions, substitutions))
			previous_row = current_row
		
		return previous_row[-1]


class ImageMetrics:
	"""
	Image quality metrics for binary images.
	Computes PSNR, SSIM, IoU, and pixel-level accuracy metrics by comparing binary image to ground truth.
	"""
	
	@staticmethod
	def psnr(reference: np.ndarray, hypothesis: np.ndarray) -> float:
		"""
		Compute Peak Signal-to-Noise Ratio (PSNR).
		PSNR = 20 * log10(MAX) - 10 * log10(MSE)
		Args:
			reference: Ground truth binary image
			hypothesis: Binarized image to evaluate
		Returns:
			PSNR in dB (higher is better, inf = perfect match)
		"""
		mse = np.mean((reference.astype(float) - hypothesis.astype(float)) ** 2)
		
		if mse == 0:
			return float('inf')
		
		max_pixel = 255.0
		psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
		
		return float(psnr)
	
	@staticmethod
	def ssim(reference: np.ndarray, hypothesis: np.ndarray) -> float:
		"""Compute Structural Similarity Index (SSIM).
		Args:
			reference: Ground truth binary image
			hypothesis: Binarized image to evaluate		
		Returns:
			SSIM value (0-1, higher is better)
		"""
		from skimage.metrics import structural_similarity
		
		return structural_similarity(reference, hypothesis, data_range=255)
	
	@staticmethod
	def iou(reference: np.ndarray, hypothesis: np.ndarray, foreground_value: int = 0) -> float:
		"""
		Compute Intersection over Union (IoU) for foreground.
		Also known as Jaccard Index.
		Args:
			reference: Ground truth binary image
			hypothesis: Binarized image to evaluate
			foreground_value: Pixel value representing foreground (default: 0)		
		Returns:
			IoU value (0-1, higher is better)
		"""
		ref_fg = (reference == foreground_value).astype(bool)
		hyp_fg = (hypothesis == foreground_value).astype(bool)
		
		intersection = np.sum(ref_fg & hyp_fg)
		union = np.sum(ref_fg | hyp_fg)
		
		if union == 0:
			return 1.0 if intersection == 0 else 0.0
		
		return float(intersection / union)
	
	@staticmethod
	def pixel_accuracy(reference: np.ndarray, hypothesis: np.ndarray) -> float:
		"""
		Compute pixel-level accuracy.
		Args:
			reference: Ground truth binary image
			hypothesis: Binarized image to evaluate		
		Returns:
			Accuracy (0-1, higher is better)
		"""
		correct = np.sum(reference == hypothesis)
		total = reference.size
		
		return float(correct / total)
	
	@staticmethod
	def foreground_metrics(
		reference: np.ndarray, 
		hypothesis: np.ndarray,
		foreground_value: int = 0
  ) -> Dict[str, float]:
		"""
		Compute precision, recall, F1 for foreground pixels.
		Args:
			reference: Ground truth binary image
			hypothesis: Binarized image to evaluate
			foreground_value: Pixel value representing foreground		
		Returns:
			Dictionary with precision, recall, f1_score
		"""
		ref_fg = (reference == foreground_value).astype(bool)
		hyp_fg = (hypothesis == foreground_value).astype(bool)
		
		true_positive = np.sum(ref_fg & hyp_fg)
		false_positive = np.sum((~ref_fg) & hyp_fg)
		false_negative = np.sum(ref_fg & (~hyp_fg))
		
		precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
		recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
		f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
		
		return {
			'precision': float(precision),
			'recall': float(recall),
			'f1_score': float(f1_score)
		}


class MetricsCalculator:
	"""
	Unified metrics calculator combining OCR and image metrics.
	Provides a single interface to compute all available metrics.
	Example:
		>>> calculator = MetricsCalculator()
		>>> metrics = calculator.compute_all(
		...   binary_image=result,
		...   ground_truth_image=gt_image,
		...   ocr_text=ocr_result.text,
		...   ground_truth_text=gt_text
		... )
	"""
	
	def __init__(self):
		self.ocr_metrics = OCRMetrics()
		self.image_metrics = ImageMetrics()
	
	def compute_all(
		self,
		binary_image: Optional[np.ndarray] = None,
		ground_truth_image: Optional[np.ndarray] = None,
		ocr_text: Optional[str] = None,
		ground_truth_text: Optional[str] = None,
		ocr_confidence: Optional[float] = None
	) -> EvaluationMetrics:
		"""
		Compute all available metrics.
		Args:
			binary_image: Binarized image to evaluate
			ground_truth_image: Ground truth binary image
			ocr_text: OCR extracted text
			ground_truth_text: Ground truth text
			ocr_confidence: OCR confidence score		
		Returns:
			EvaluationMetrics with all computed metrics
		"""
		metrics = EvaluationMetrics()
		
		# OCR metrics (if text is available)
		if ocr_text is not None and ground_truth_text is not None:
			metrics.cer = self.ocr_metrics.character_error_rate(
				ground_truth_text, 
				ocr_text
			)
			metrics.wer = self.ocr_metrics.word_error_rate(
				ground_truth_text, 
				ocr_text
			)
		
		# Image metrics (if images are available)
		if binary_image is not None and ground_truth_image is not None:
			# Ensure same size
			if binary_image.shape != ground_truth_image.shape:
				binary_image = cv2.resize(
					binary_image, 
					(ground_truth_image.shape[1], 
					ground_truth_image.shape[0])
				)
			
			metrics.psnr = self.image_metrics.psnr(ground_truth_image, binary_image)
			metrics.ssim = self.image_metrics.ssim(ground_truth_image, binary_image)
			metrics.iou = self.image_metrics.iou(ground_truth_image, binary_image)
			
			fg_metrics = self.image_metrics.foreground_metrics(
				ground_truth_image, 
				binary_image
			)
			metrics.precision = fg_metrics['precision']
			metrics.recall = fg_metrics['recall']
			metrics.f1_score = fg_metrics['f1_score']
		
		# OCR confidence
		if ocr_confidence is not None:
			metrics.ocr_confidence = ocr_confidence
		
		return metrics
	
	def compute_batch(
		self,
		binary_images: list,
		ground_truth_images: list,
		ocr_texts: Optional[list] = None,
		ground_truth_texts: Optional[list] = None,
		ocr_confidences: Optional[list] = None
	) -> Tuple[List[EvaluationMetrics], Dict[str, float]]:
		"""
		Compute metrics for batch of images.
		Args:
			binary_images: List of binarized images
			ground_truth_images: List of ground truth images
			ocr_texts: List of OCR texts (optional)
			ground_truth_texts: List of ground truth texts (optional)
			ocr_confidences: List of OCR confidences (optional)		
		Returns:
			Tuple of (per-image metrics list, aggregate statistics dict)
		"""
		if len(binary_images) != len(ground_truth_images):
			raise ValueError("binary_images and ground_truth_images must have the same length")
	
		per_image_metrics = []
		
		for i in range(len(binary_images)):
			metrics = self.compute_all(
				binary_image=binary_images[i],
				ground_truth_image=ground_truth_images[i],
				ocr_text=ocr_texts[i] if ocr_texts else None,
				ground_truth_text=ground_truth_texts[i] if ground_truth_texts else None,
				ocr_confidence=ocr_confidences[i] if ocr_confidences else None
			)
			per_image_metrics.append(metrics)
		
		# Compute aggregate statistics
		aggregate = self._aggregate_metrics(per_image_metrics)
		
		return per_image_metrics, aggregate
	
	def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
		"""
		Aggregate metrics from multiple images.
		Args:
			metrics_list: List of EvaluationMetrics	
		Returns:
			Dictionary with mean, std, min, max for each metric
		"""
		aggregate = {}
		
		metric_names = ['cer', 'wer', 'precision', 'recall', 'f1_score', 'ocr_confidence', 'psnr', 'ssim', 'iou']
		
		for name in metric_names:
			values = [getattr(m, name) for m in metrics_list if getattr(m, name) is not None]
			
			if values:
				aggregate[f'{name}_mean'] = float(np.mean(values))
				aggregate[f'{name}_std'] = float(np.std(values))
				aggregate[f'{name}_min'] = float(np.min(values))
				aggregate[f'{name}_max'] = float(np.max(values))
				aggregate[f'{name}_median'] = float(np.median(values))
		
		return aggregate