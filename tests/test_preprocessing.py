import pandas as pd
import numpy as np
import pytest

from src.data.preprocessing import DataPreprocessor, PreprocessingConfig


def test_drop_strategy_removes_missing_values():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    preprocessor = DataPreprocessor(PreprocessingConfig(features_to_scale=["b"]))
    result = preprocessor.fit_transform(df)
    assert result.isna().sum().sum() == 0


def test_fill_strategy_uses_configured_value():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    preprocessor = DataPreprocessor(
        PreprocessingConfig(features_to_scale=[], missing_strategy="fill", fill_value=42)
    )
    result = preprocessor.fit_transform(df)
    assert (result["a"] == pd.Series([1, 42, 3])).all()


def test_scaling_normalizes_feature():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    preprocessor = DataPreprocessor(PreprocessingConfig(features_to_scale=["a"], missing_strategy="fill"))
    transformed = preprocessor.fit_transform(df)
    assert np.isclose(transformed["a"].mean(), 0.0, atol=1e-6)


def test_raises_when_transform_without_fit():
    df = pd.DataFrame({"a": [1, 2, 3]})
    preprocessor = DataPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(df)
