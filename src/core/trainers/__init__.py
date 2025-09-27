"""Trainer implementations for supervised models."""

from .regression import LinearRegressionTrainer, RegressionConfig
from .xgboost import XGBoostTrainer, XGBoostConfig

__all__ = [
    "LinearRegressionTrainer",
    "RegressionConfig",
    "XGBoostTrainer",
    "XGBoostConfig",
]
