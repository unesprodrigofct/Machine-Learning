"""Backward-compatible re-export of NLP pipelines."""

from src.core.pipelines.text import TextClusteringConfig, TextClusteringPipeline

__all__ = ["TextClusteringConfig", "TextClusteringPipeline"]
