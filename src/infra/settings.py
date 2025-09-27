"""Application settings loaded from environment variables and config files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime configuration for the ML system."""

    environment: str = "dev"
    artifacts_path: Path = Path("artifacts")
    text_corpus_path: Path = Path("data/text_corpus.txt")
    mlflow_tracking_uri: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        env_file_encoding="utf-8"
    )


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
