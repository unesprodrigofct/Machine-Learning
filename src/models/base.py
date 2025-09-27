"""Base classes and interfaces for machine learning models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


@dataclass
class ModelMetadata:
    """Metadata container describing model training context."""

    name: str
    version: str = "1.0.0"
    artifact_path: Path = field(default_factory=lambda: Path("artifacts"))
    extra: Dict[str, Any] = field(default_factory=dict)


class PersistableModel:
    """Mixin that adds persistence helpers for scikit-learn compatible models."""

    def __init__(self, metadata: Optional[ModelMetadata] = None) -> None:
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)

    def save(self, model: Any) -> Path:
        """Persist the provided model instance to disk."""

        self.metadata.artifact_path.mkdir(parents=True, exist_ok=True)
        output_path = self.metadata.artifact_path / f"{self.metadata.name}.pkl"
        joblib.dump(model, output_path)
        return output_path

    @staticmethod
    def load(path: Path) -> Any:
        """Load a persisted model artifact."""

        return joblib.load(path)


__all__ = ["ModelMetadata", "PersistableModel"]
