"""XGBoost training utilities for binary classification problems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .base import ModelMetadata, PersistableModel


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost training pipeline."""

    test_size: float = 0.3
    random_state: int = 1337
    params_grid: Optional[Dict[str, Any]] = None


class XGBoostTrainer(PersistableModel):
    """Encapsulates an XGBoost training workflow with grid search."""

    def __init__(self, config: XGBoostConfig, metadata: Optional[ModelMetadata] = None) -> None:
        super().__init__(metadata)
        self.config = config
        self.model = xgb.XGBClassifier()

    def load_data(self, path: Path, target_column: str) -> pd.DataFrame:
        dataset = pd.read_csv(path)
        if target_column not in dataset:
            raise ValueError(f"Target column '{target_column}' not present in dataset")
        return dataset

    def train(self, dataset: pd.DataFrame, target_column: str) -> float:
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        search_space = self.config.params_grid or {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "min_child_weight": [1, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "n_estimators": [100],
        }

        grid_search = GridSearchCV(
            self.model,
            param_grid=search_space,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state),
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return auc

    def export(self) -> Path:
        return self.save(self.model)

    @staticmethod
    def load_artifact(path: Path) -> xgb.XGBClassifier:
        return joblib.load(path)


__all__ = ["XGBoostConfig", "XGBoostTrainer"]
