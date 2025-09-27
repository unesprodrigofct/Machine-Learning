"""Logging utilities for the machine learning package."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
