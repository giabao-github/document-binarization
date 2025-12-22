"""Simple registry for binarization methods."""
METHODS = {}


def register(name: str, func):
    METHODS[name] = func
    return func


def get(name: str):
    return METHODS.get(name)
