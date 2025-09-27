"""Core abstractions shared across training and inference layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class ModelMetadata:
    """Metadata container describing model training context."""

    name: str
    version: str = "1.0.0"
    artifact_path: Path = field(default_factory=lambda: Path("artifacts"))
    extra: dict[str, Any] = field(default_factory=dict)


class PersistableModel:
    """Mixin adding persistence helpers for scikit-learn compatible models."""

    def __init__(self, metadata: ModelMetadata | None = None) -> None:
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)

    def save(self, model: BaseEstimator) -> Path:
        """Persist the provided model instance to disk."""

        self.metadata.artifact_path.mkdir(parents=True, exist_ok=True)
        output_path = self.metadata.artifact_path / f"{self.metadata.name}.pkl"
        joblib.dump(model, output_path)
        logger.info("model.persisted", path=str(output_path))
        return output_path

    @staticmethod
    def load(path: Path) -> BaseEstimator:
        """Load a persisted model artifact."""

        logger.info("model.loaded", path=str(path))
        return joblib.load(path)


__all__ = ["ModelMetadata", "PersistableModel"]
