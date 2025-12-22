"""Post-processing utilities for binarized images."""
from .morphology import opening, closing
from .components import filter_components

__all__ = ["opening", "closing", "filter_components"]
