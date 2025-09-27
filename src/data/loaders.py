"""Reusable data loading utilities for ML projects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


@dataclass
class CSVLoaderConfig:
    """Configuration for loading tabular data from CSV files."""

    path: Path
    sep: str = ","
    encoding: str = "utf-8"
    usecols: Iterable[str] | None = None


def load_csv_dataset(config: CSVLoaderConfig) -> pd.DataFrame:
    """Load a CSV file into a DataFrame according to the provided config."""

    return pd.read_csv(
        config.path,
        sep=config.sep,
        encoding=config.encoding,
        usecols=config.usecols,
    )


@dataclass
class SQLLoaderConfig:
    """Configuration for loading data via SQL queries."""

    connection_uri: str
    query: str


def load_sql_dataset(config: SQLLoaderConfig) -> pd.DataFrame:
    """Execute a SQL query and return the results as a DataFrame."""

    engine = create_engine(config.connection_uri)
    with engine.connect() as connection:
        result = connection.execute(text(config.query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    engine.dispose()
    return df


def load_text_corpus(path: Path, encoding: str = "utf-8") -> list[str]:
    """Load a text corpus where each line is treated as a separate document."""

    content = path.read_text(encoding=encoding)
    return [line.strip() for line in content.splitlines() if line.strip()]


def load_titanic_for_xgboost(cache_dir: Path | None = None) -> pd.DataFrame:
    """Download and preprocess the Titanic dataset for XGBoost training."""

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_cache = cache_dir / "titanic_raw.csv"
        if raw_cache.exists():
            df = pd.read_csv(raw_cache)
        else:
            df = pd.read_csv(url)
            df.to_csv(raw_cache, index=False)
    else:
        df = pd.read_csv(url)

    processed = df[["Pclass", "Age", "Fare", "SibSp", "Parch", "Sex"]].copy()
    processed["Sex"] = processed["Sex"].map({"male": 0, "female": 1}).fillna(0)
    processed["Age"] = processed["Age"].fillna(processed["Age"].median())
    processed["Fare"] = processed["Fare"].fillna(processed["Fare"].median())
    processed = processed.fillna(0)
    processed.rename(columns={
        "Pclass": "FEATURE_PCLASS",
        "Age": "FEATURE_AGE",
        "Fare": "FEATURE_FARE",
        "SibSp": "FEATURE_SIBSP",
        "Parch": "FEATURE_PARCH",
        "Sex": "FEATURE_SEX",
    }, inplace=True)
    processed["TARGET"] = df["Survived"].astype(int)

    if cache_dir is not None:
        prepared_cache = cache_dir / "titanic_xgboost.csv"
        processed.to_csv(prepared_cache, index=False)

    return processed


__all__ = [
    "CSVLoaderConfig",
    "SQLLoaderConfig",
    "load_csv_dataset",
    "load_sql_dataset",
    "load_text_corpus",
    "load_titanic_for_xgboost",
]
