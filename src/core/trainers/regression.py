"""Linear regression training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.base import ModelMetadata, PersistableModel


@dataclass
class RegressionConfig:
    test_size: float = 0.3
    random_state: Optional[int] = None


class LinearRegressionTrainer(PersistableModel):
    """Encapsulated workflow for linear regression training and evaluation."""

    def __init__(self, config: RegressionConfig, metadata: Optional[ModelMetadata] = None) -> None:
        super().__init__(metadata)
        self.config = config
        self.model = linear_model.LinearRegression()

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = datasets.load_diabetes()
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target, name="target")
        return X, y

    def train(self) -> Tuple[float, float]:
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def export(self) -> None:
        self.save(self.model)


__all__ = ["RegressionConfig", "LinearRegressionTrainer"]
