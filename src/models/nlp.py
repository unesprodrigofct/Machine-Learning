"""Natural Language Processing utilities and pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


@dataclass
class TextClusteringConfig:
    """Configuration for the text clustering pipeline."""

    n_clusters: int = 5
    max_features: int = 5_000
    ngram_range: tuple[int, int] = (1, 2)
    random_state: Optional[int] = 42


class TextClusteringPipeline:
    """Pipeline that transforms raw documents and fits a clustering model."""

    def __init__(self, config: Optional[TextClusteringConfig] = None) -> None:
        self.config = config or TextClusteringConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
        )
        self.model = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state)
        self._fitted = False

    def fit(self, documents: Iterable[str]) -> None:
        matrix = self.vectorizer.fit_transform(documents)
        self.model.fit(matrix)
        self._fitted = True

    def predict(self, documents: Iterable[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before predicting clusters.")
        matrix = self.vectorizer.transform(documents)
        return self.model.predict(matrix)

    def fit_predict(self, documents: Iterable[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(documents)
        labels = self.model.fit_predict(matrix)
        self._fitted = True
        return labels

    def silhouette(self, documents: Iterable[str]) -> float:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before computing silhouette score.")
        matrix = self.vectorizer.transform(documents)
        return silhouette_score(matrix, self.model.predict(matrix))

    def top_terms_per_cluster(self, top_n: int = 5) -> List[List[str]]:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before extracting top terms.")
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        centroids = self.model.cluster_centers_
        top_indices = np.argsort(centroids, axis=1)[:, ::-1][:, :top_n]
        return feature_names[top_indices].tolist()


__all__ = ["TextClusteringConfig", "TextClusteringPipeline"]
