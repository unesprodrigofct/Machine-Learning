import pandas as pd
from sklearn import datasets

from src.core.base import ModelMetadata
from src.core.trainers import LinearRegressionTrainer, RegressionConfig


def test_linear_regression_trainer_produces_metrics(tmp_path):
    trainer = LinearRegressionTrainer(
        config=RegressionConfig(test_size=0.2, random_state=42),
        metadata=ModelMetadata(name="linear_diabetes", artifact_path=tmp_path),
    )

    mse, r2 = trainer.train()
    assert mse > 0
    assert -1.0 <= r2 <= 1.0

    trainer.export()
    saved = tmp_path / "linear_diabetes.pkl"
    assert saved.exists()


def test_dataset_shape_matches_sklearn_diabetes():
    trainer = LinearRegressionTrainer(config=RegressionConfig())
    X, y = trainer.load_data()
    data = datasets.load_diabetes()
    assert X.shape == data.data.shape
    assert y.shape[0] == data.target.shape[0]
