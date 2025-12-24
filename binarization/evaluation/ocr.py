"""
OCR integration for evaluating binarization quality.
This module provides integration with Tesseract OCR for extracting text and evaluating binarization results based on OCR accuracy.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from dataclasses import dataclass, field
import warnings

try:
	import pytesseract
	TESSERACT_AVAILABLE = True
except ImportError:
	TESSERACT_AVAILABLE = False
	warnings.warn("pytesseract not available. OCR functionality will be limited.", stacklevel=2)


@dataclass
class OCRResult:
	"""
	Container for OCR results.
	Attributes:
		text: Extracted text
		confidence: Overall confidence score (0-100)
		word_confidences: Confidence per word
		char_confidences: Confidence per character
		boxes: Bounding boxes for detected text
		metadata: Additional OCR metadata
	"""
	text: str
	confidence: float
	word_confidences: List[float] = field(default_factory=list)
	char_confidences: List[float] = field(default_factory=list)
	boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def __post_init__(self):
		"""Validate OCR result."""
		if not 0 <= self.confidence <= 100:
			warnings.warn(f"Confidence {self.confidence} outside [0, 100] range", stacklevel=2)


class TesseractOCR:
	"""
	Tesseract OCR wrapper for text extraction.
	Provides convenient interface to Tesseract OCR with various output formats and configuration options.
	Example:
		>>> ocr = TesseractOCR(lang='eng')
		>>> result = ocr.recognize(binary_image)
		>>> print(f"Text: {result.text}")
		>>> print(f"Confidence: {result.confidence:.2f}%")
	"""
	
	def __init__(self, lang: str = 'eng', config: str = '--psm 6'):
		"""
		Initialize Tesseract OCR.
		Args:
			lang: Language code(s), e.g., 'eng', 'fra', 'eng+fra'
			config: Tesseract configuration string
				PSM modes:
				0 = Orientation and script detection only
				1 = Automatic page segmentation with OSD
				3 = Fully automatic page segmentation (default)
				4 = Single column of text
				6 = Uniform block of text (recommended for documents)
				7 = Single text line
				8 = Single word
				11 = Sparse text
				13 = Raw line
		"""
		if not TESSERACT_AVAILABLE:
			raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract")
		
		self.lang = lang
		self.config = config
	
	def recognize(self, image: np.ndarray, **kwargs) -> OCRResult:
		"""
		Recognize text in image.
		Args:
			image: Input image (grayscale or binary)
			**kwargs: Additional Tesseract parameters	
		Returns:
			OCRResult with extracted text and confidence
		"""
		# Ensure grayscale
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# Get custom config if provided
		config = kwargs.get('config', self.config)
		lang = kwargs.get('lang', self.lang)
		
		# Extract text
		text = pytesseract.image_to_string(image, lang=lang, config=config)
		
		# Get detailed data
		data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
		
		# Calculate overall confidence
		confidences = [conf for conf in data['conf'] if conf != -1]
		overall_confidence = float(np.mean(confidences)) if confidences else 0.0
		
		# Extract word-level information
		word_confidences = []
		boxes = []
		
		n_boxes = len(data['text'])
		for i in range(n_boxes):
			if data['text'][i].strip():  # Non-empty text
				conf = data['conf'][i]
				if conf != -1:
					word_confidences.append(float(conf))
					
					# Bounding box
					x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
					boxes.append((x, y, w, h))
		
		return OCRResult(
			text=text.strip(),
			confidence=overall_confidence,
			word_confidences=word_confidences,
			boxes=boxes,
			metadata={
				'lang': lang,
				'config': config,
				'num_words': len(word_confidences)
			}
		)
	
	def recognize_with_boxes(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
		"""
		Recognize text with detailed bounding boxes.
		Args:
			image: Input image	
		Returns:
			Tuple of (text, list of word info dicts)
		"""
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		data = pytesseract.image_to_data(
			image, 
			lang=self.lang, 
			config=self.config, 
			output_type=pytesseract.Output.DICT
		)
		
		text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
		
		words = []
		n_boxes = len(data['text'])
		for i in range(n_boxes):
			if data['text'][i].strip():
				words.append({
					'text': data['text'][i],
					'confidence': data['conf'][i],
					'box': {
						'x': data['left'][i],
						'y': data['top'][i],
						'w': data['width'][i],
						'h': data['height'][i]
					},
					'level': data['level'][i]
				})
		
		return text.strip(), words
	
	def get_confidence_distribution(self, image: np.ndarray) -> Dict[str, Any]:
		"""
		Get confidence score distribution.
		Args:
			image: Input image	
		Returns:
			Dictionary with confidence statistics
		"""
		result = self.recognize(image)
		
		if not result.word_confidences:
			return {
				'mean': 0.0,
				'median': 0.0,
				'std': 0.0,
				'min': 0.0,
				'max': 0.0,
				'distribution': []
			}
		
		confidences = np.array(result.word_confidences)
		
		return {
			'mean': float(np.mean(confidences)),
			'median': float(np.median(confidences)),
			'std': float(np.std(confidences)),
			'min': float(np.min(confidences)),
			'max': float(np.max(confidences)),
			'q25': float(np.percentile(confidences, 25)),
			'q75': float(np.percentile(confidences, 75)),
			'distribution': confidences.tolist()
		}


class OCRComparator:
	"""
	Compare OCR results against ground truth. 
	Provides various metrics for evaluating OCR accuracy including character error rate, word error rate, and accuracy. 
	Example:
		>>> comparator = OCRComparator()
		>>> metrics = comparator.compare(predicted_text, ground_truth_text)
		>>> print(f"CER: {metrics['cer']:.2%}")
	"""
	
	def __init__(self):
		pass
	
	def compare(self, predicted: str, ground_truth: str) -> Dict[str, float]:
		"""
		Compare predicted text with ground truth.
		Args:
			predicted: Predicted text from OCR
			ground_truth: Ground truth text	
		Returns:
			Dictionary with comparison metrics
		"""
		# Character-level metrics
		cer = self.character_error_rate(predicted, ground_truth)
		
		# Word-level metrics
		wer = self.word_error_rate(predicted, ground_truth)
		word_acc = 1.0 - wer
		
		# Exact match
		exact_match = float(predicted.strip() == ground_truth.strip())
		
		# Length comparison
		len_ratio = len(predicted) / max(len(ground_truth), 1)
		
		return {
			'cer': cer,
			'wer': wer,
			'character_accuracy': 1.0 - cer,
			'word_accuracy': word_acc,
			'exact_match': exact_match,
			'length_ratio': len_ratio,
			'predicted_length': len(predicted),
			'ground_truth_length': len(ground_truth)
		}
	
	def character_error_rate(self, predicted: str, ground_truth: str) -> float:
		"""
		Calculate Character Error Rate (CER).
		CER = (insertions + deletions + substitutions) / len(ground_truth)
		Args:
			predicted: Predicted text
			ground_truth: Ground truth text
		Returns:
			Character error rate (0-1)
		"""
		distance = self._levenshtein_distance(predicted, ground_truth)
		
		if len(ground_truth) == 0:
			return 0.0 if len(predicted) == 0 else 1.0
		
		return min(1.0, distance / len(ground_truth))
	
	def word_error_rate(self, predicted: str, ground_truth: str) -> float:
		"""
		Calculate Word Error Rate (WER).
		WER = (insertions + deletions + substitutions) / num_words_gt
		Args:
			predicted: Predicted text
			ground_truth: Ground truth text	
		Returns:
			Word error rate (0-1)
		"""
		pred_words = predicted.split()
		gt_words = ground_truth.split()
		
		distance = self._levenshtein_distance(pred_words, gt_words)
		
		if len(gt_words) == 0:
			return 0.0 if len(pred_words) == 0 else 1.0
		
		return min(1.0, distance / len(gt_words))
	
	def _levenshtein_distance(self, s1: any, s2: any) -> int:
		"""
		Calculate Levenshtein distance between two sequences.
		Args:
			s1: First sequence (string or list)
			s2: Second sequence (string or list)	
		Returns:
			Edit distance
		"""
		if len(s1) < len(s2):
			return self._levenshtein_distance(s2, s1)
		
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
	
	def precision_recall_f1(self, predicted: str, ground_truth: str) -> Dict[str, float]:
		"""
		Calculate precision, recall, and F1 score at character level.
		Args:
			predicted: Predicted text
			ground_truth: Ground truth text	
		Returns:
			Dictionary with precision, recall, F1
		"""
		pred_chars = set(predicted)
		gt_chars = set(ground_truth)
		
		if len(pred_chars) == 0 and len(gt_chars) == 0:
			return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
		
		if len(pred_chars) == 0:
			return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
		
		if len(gt_chars) == 0:
			return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
		
		true_positives = len(pred_chars & gt_chars)
		
		precision = true_positives / len(pred_chars) if pred_chars else 0.0
		recall = true_positives / len(gt_chars) if gt_chars else 0.0
		
		if precision + recall > 0:
			f1 = 2 * (precision * recall) / (precision + recall)
		else:
			f1 = 0.0
		
		return {
			'precision': precision,
			'recall': recall,
			'f1': f1
		}


class BatchOCREvaluator:
	"""
	Evaluate OCR performance on batches of images.
	Processes multiple images and computes aggregate statistics.
	Example:
		>>> evaluator = BatchOCREvaluator()
		>>> results = evaluator.evaluate_batch(images, ground_truths)
		>>> print(f"Mean CER: {results['mean_cer']:.2%}")
	"""
	
	def __init__(self, ocr: Optional[TesseractOCR] = None):
		"""
		Initialize batch evaluator.
		Args:
			ocr: TesseractOCR instance (creates default if None)
		"""
		self.ocr = ocr or TesseractOCR()
		self.comparator = OCRComparator()
	
	def evaluate_batch(self, images: List[np.ndarray], ground_truths: List[str]) -> Dict[str, Any]:
		"""
		Evaluate OCR on batch of images.
		Args:
			images: List of input images
			ground_truths: List of ground truth texts
		Returns:
			Dictionary with aggregate metrics
		"""
		if len(images) != len(ground_truths):
			raise ValueError("Number of images must match number of ground truths")
		
		results = []
		
		for image, gt in zip(images, ground_truths):
			# Run OCR
			ocr_result = self.ocr.recognize(image)
			
			# Compare with ground truth
			comparison = self.comparator.compare(ocr_result.text, gt)
			
			results.append({
				'predicted': ocr_result.text,
				'ground_truth': gt,
				'confidence': ocr_result.confidence,
				**comparison
			})
		
		# Compute aggregate statistics
		cers = [r['cer'] for r in results]
		wers = [r['wer'] for r in results]
		confidences = [r['confidence'] for r in results]
		
		return {
			'num_images': len(images),
			'mean_cer': float(np.mean(cers)),
			'median_cer': float(np.median(cers)),
			'std_cer': float(np.std(cers)),
			'mean_wer': float(np.mean(wers)),
			'median_wer': float(np.median(wers)),
			'std_wer': float(np.std(wers)),
			'mean_confidence': float(np.mean(confidences)),
			'median_confidence': float(np.median(confidences)),
			'per_image_results': results
		}