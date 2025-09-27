"""Data preprocessing utilities for machine learning pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class PreprocessingConfig:
    """Configuration holder for preprocessing pipelines."""

    features_to_scale: Iterable[str]
    missing_strategy: str = "drop"
    fill_value: Any = 0


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Professional preprocessing pipeline compatible with scikit-learn."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig(features_to_scale=[])
        self._fitted = False
        self._feature_means: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """Fit preprocessing statistics on the provided data."""

        self._feature_means = {
            feature: X[feature].mean() for feature in self.config.features_to_scale if feature in X
        }
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to the data."""

        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        processed = self._handle_missing(X.copy())
        processed = self._scale_features(processed)
        return processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the preprocessor and transform the data in a single step."""

        return self.fit(X, y).transform(X)

    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.missing_strategy
        if strategy == "drop":
            return X.dropna()
        if strategy == "fill":
            return X.fillna(self.config.fill_value)
        raise ValueError(f"Unsupported missing value strategy: {strategy}")

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature, mean_value in self._feature_means.items():
            if feature in X:
                X[feature] = (X[feature] - mean_value) / (np.std(X[feature]) + 1e-8)
        return X


__all__ = ["PreprocessingConfig", "DataPreprocessor"]
