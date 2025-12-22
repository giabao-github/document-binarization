"""Logging helper."""
import logging


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger
