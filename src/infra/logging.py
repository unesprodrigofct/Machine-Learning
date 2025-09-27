"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
from typing import Any

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog with sensible defaults."""

    logging.basicConfig(format="%(message)s", stream=None, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_traceback,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: dict[str, Any]) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger with optional context."""

    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger
