"""CLI script to run the text clustering pipeline."""

from pathlib import Path

from src.data.loaders import load_text_corpus
from src.models.nlp import TextClusteringConfig, TextClusteringPipeline


def main() -> None:
    corpus_path = Path("data/text_corpus.txt")
    documents = load_text_corpus(corpus_path)

    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2))
    labels = pipeline.fit_predict(documents)
    terms = pipeline.top_terms_per_cluster(top_n=5)

    print("Cluster assignments:")
    for doc, label in zip(documents, labels):
        print(f"- [{label}] {doc[:80]}")

    print("\nTop terms per cluster:")
    for idx, cluster_terms in enumerate(terms):
        print(f"Cluster {idx}: {', '.join(cluster_terms)}")


if __name__ == "__main__":
    main()
