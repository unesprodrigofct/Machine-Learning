"""Backward-compatible re-export of regression trainer."""

from src.core.trainers.regression import LinearRegressionTrainer, RegressionConfig

__all__ = ["RegressionConfig", "LinearRegressionTrainer"]
