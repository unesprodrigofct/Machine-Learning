"""Backward-compatible re-export of XGBoost trainer."""

from src.core.trainers.xgboost import XGBoostConfig, XGBoostTrainer

__all__ = ["XGBoostConfig", "XGBoostTrainer"]
