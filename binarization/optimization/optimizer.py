"""Simple optimizer interface (stub)."""
from typing import Any


class Optimizer:
    def __init__(self, config: Any = None):
        self.config = config

    def optimize(self, objective, n_trials: int = 50):
        raise NotImplementedError("optimize is not implemented")
