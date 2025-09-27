"""CLI script to run the text clustering pipeline."""

from pathlib import Path

from src.core.pipelines import TextClusteringConfig, TextClusteringPipeline
from src.data.loaders import load_text_corpus
from src.infra.logging import configure_logging, get_logger
from src.infra.settings import get_settings


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    corpus_path = Path(settings.text_corpus_path)
    documents = load_text_corpus(corpus_path)

    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2))
    labels = pipeline.fit_predict(documents)
    terms = pipeline.top_terms_per_cluster(top_n=5)

    logger.info("clusters.built", documents=len(documents))
    for doc, label in zip(documents, labels):
        logger.info("clusters.assignment", label=int(label), document_preview=doc[:80])

    for idx, cluster_terms in enumerate(terms):
        logger.info("clusters.top_terms", cluster=idx, terms=cluster_terms)


if __name__ == "__main__":
    main()
