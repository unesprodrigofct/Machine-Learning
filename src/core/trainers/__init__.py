"""Trainer implementations for supervised models."""

from .regression import LinearRegressionTrainer, RegressionConfig
from .xgboost import XGBoostConfig, XGBoostTrainer

__all__ = [
    "LinearRegressionTrainer",
    "RegressionConfig",
    "XGBoostTrainer",
    "XGBoostConfig",
]
