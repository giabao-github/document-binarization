"""Binarization methods package.

Contains global, adaptive and advanced algorithm implementations and a registry.
"""
from .registry import METHODS  # expose registry

__all__ = ["METHODS"]
