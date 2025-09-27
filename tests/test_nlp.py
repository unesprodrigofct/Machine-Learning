import pytest

from src.core.pipelines import TextClusteringConfig, TextClusteringPipeline


@pytest.fixture
def sample_documents():
    return [
        "Machine learning enables predictive analytics",
        "Machine learning improves decision making",
        "Cooking recipes share similar ingredients",
        "Baking relies on precise ingredient ratios",
    ]


def test_pipeline_fit_predict(sample_documents):
    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2, random_state=42))
    pipeline.fit(sample_documents)
    labels = pipeline.predict(sample_documents)
    assert len(labels) == len(sample_documents)
    assert set(labels) <= {0, 1}


def test_fit_predict_returns_labels(sample_documents):
    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2, random_state=42))
    labels = pipeline.fit_predict(sample_documents)
    assert len(labels) == len(sample_documents)


def test_top_terms_requires_fit(sample_documents):
    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2, random_state=42))
    with pytest.raises(RuntimeError):
        pipeline.top_terms_per_cluster()

    pipeline.fit(sample_documents)
    terms = pipeline.top_terms_per_cluster(top_n=2)
    assert len(terms) == 2
    for cluster_terms in terms:
        assert len(cluster_terms) == 2
