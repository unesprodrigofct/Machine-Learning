"""Data preprocessing utilities for machine learning pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig(features_to_scale=[])
        self._fitted = False
        self._feature_means: dict[str, float] = {}

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None
    ) -> DataPreprocessor:
        """Fit preprocessing statistics on the provided data."""
        self._feature_means = {
            feature: x[feature].mean()
            for feature in self.config.features_to_scale
            if feature in x
        }
        self._fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to the data."""
        if not self._fitted:
            msg = "Preprocessor must be fitted before calling transform()."
            raise RuntimeError(msg)

        processed = self._handle_missing(x.copy())
        processed = self._scale_features(processed)
        return processed

    def fit_transform(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit the preprocessor and transform the data in a single step."""
        return self.fit(x, y).transform(x)

    def _handle_missing(self, x: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.missing_strategy
        if strategy == "drop":
            return x.dropna()
        if strategy == "fill":
            return x.fillna(self.config.fill_value)
        msg = f"Unsupported missing value strategy: {strategy}"
        raise ValueError(msg)

    def _scale_features(self, x: pd.DataFrame) -> pd.DataFrame:
        for feature, mean_value in self._feature_means.items():
            if feature in x:
                x[feature] = (x[feature] - mean_value) / (np.std(x[feature]) + 1e-8)
        return x


__all__ = ["PreprocessingConfig", "DataPreprocessor"]
