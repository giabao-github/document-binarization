"""Evaluation utilities for binarization."""
from .metrics import compute_metrics
from .ocr import ocr_text

__all__ = ["compute_metrics", "ocr_text"]
