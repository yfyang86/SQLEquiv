"""Tests for the embedding similarity stack."""

import numpy as np
import pytest

from sql_equivalence.equivalence.embedding_similarity import EmbeddingSimilarityChecker
from sql_equivalence.models import HashingVectorEncoder
from sql_equivalence.parser.sql_parser import SQLParser
from sql_equivalence.representations.embedding.graph_embedding import GraphEmbedding


@pytest.fixture(scope='module')
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


# --------------------------------------------------- EmbeddingSimilarity
def test_cosine_is_in_zero_one_range() -> None:
    checker = EmbeddingSimilarityChecker()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])  # anti-parallel
    sim = checker.compute_similarity(a, b)
    assert 0.0 <= sim <= 1.0
    assert sim == pytest.approx(0.0, abs=1e-6)


def test_cosine_of_identical_vectors_is_one() -> None:
    checker = EmbeddingSimilarityChecker()
    v = np.array([1.0, 2.0, 3.0])
    assert checker.compute_similarity(v, v) == pytest.approx(1.0)


def test_cosine_zero_vector_does_not_error() -> None:
    checker = EmbeddingSimilarityChecker()
    zero = np.zeros(4)
    # Two zeros are trivially equal.
    assert checker.compute_similarity(zero, zero) == pytest.approx(1.0)
    # Zero vs non-zero falls back to 0.0 instead of NaN.
    assert checker.compute_similarity(zero, np.array([1.0, 0.0, 0.0, 0.0])) == 0.0


def test_parsed_query_is_accepted_directly(parser: SQLParser) -> None:
    parsed = parser.parse("SELECT id FROM t")
    checker = EmbeddingSimilarityChecker()
    result = checker.check_equivalence(parsed, parsed)
    assert result.is_equivalent is True


def test_threshold_applies(parser: SQLParser) -> None:
    parsed1 = parser.parse("SELECT id FROM t")
    parsed2 = parser.parse("SELECT name FROM other WHERE x > 10 GROUP BY y")

    strict = EmbeddingSimilarityChecker(similarity_threshold=0.99)
    lenient = EmbeddingSimilarityChecker(similarity_threshold=0.0)

    r_strict = strict.check_equivalence(parsed1, parsed2)
    r_lenient = lenient.check_equivalence(parsed1, parsed2)
    # The similarity score itself must not depend on the threshold.
    assert r_strict.confidence == pytest.approx(r_lenient.confidence)
    # But the binary verdict does.
    assert r_lenient.is_equivalent is True
    assert r_strict.is_equivalent is False


def test_ensemble_returns_all_metrics() -> None:
    checker = EmbeddingSimilarityChecker(use_ensemble=True)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.5, 3.0, 4.0])
    result = checker.check_equivalence(a, b)
    ensemble = result.details.get('ensemble_scores')
    assert ensemble is not None
    assert {'cosine', 'euclidean', 'manhattan', 'pearson'} <= set(ensemble)


def test_batch_compute_similarity_shape() -> None:
    checker = EmbeddingSimilarityChecker()
    batch1 = [np.random.randn(16) for _ in range(3)]
    batch2 = [np.random.randn(16) for _ in range(4)]
    mat = checker.batch_compute_similarity(batch1, batch2)
    assert mat.shape == (3, 4)
    assert np.all(mat >= -1.0) and np.all(mat <= 1.0)


# --------------------------------------------------- HashingVectorEncoder
def test_hashing_encoder_handles_parsed_query(parser: SQLParser) -> None:
    parsed = parser.parse("SELECT id FROM users")
    encoder = HashingVectorEncoder(dim=64)
    vec = encoder.encode(parsed)
    assert vec.shape == (64,)


def test_hashing_encoder_rejects_unsupported_type() -> None:
    encoder = HashingVectorEncoder(dim=32)
    with pytest.raises(TypeError):
        encoder.encode(42)


def test_hashing_encoder_ngram_range_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        HashingVectorEncoder(dim=16, ngram_range=(3, 1))
    with pytest.raises(ValueError):
        HashingVectorEncoder(dim=16, ngram_range=(0, 2))


def test_hashing_encoder_unigram_vs_bigram_differ() -> None:
    uni = HashingVectorEncoder(dim=256, ngram_range=(1, 1))
    bi = HashingVectorEncoder(dim=256, ngram_range=(1, 2))
    sql = "SELECT id FROM users WHERE age > 18"
    assert not np.array_equal(uni.encode(sql), bi.encode(sql))


# --------------------------------------------------- GraphEmbedding
def test_graph_embedding_rejects_unknown_method(parser: SQLParser) -> None:
    parsed = parser.parse("SELECT id FROM t")
    with pytest.raises(ValueError):
        GraphEmbedding(parsed, method='nonsense')


def test_graph_embedding_pooling_methods_produce_different_vectors(
    parser: SQLParser,
) -> None:
    parsed = parser.parse("SELECT id FROM users WHERE age > 10")
    mean = GraphEmbedding(parsed, embedding_dim=32, method='mean_pool')
    sum_ = GraphEmbedding(parsed, embedding_dim=32, method='sum_pool')
    mean.build()
    sum_.build()
    # Sum is exactly count * mean; on a non-empty graph they must differ.
    assert not np.allclose(mean.embedding, sum_.embedding)


def test_graph_embedding_on_empty_graph_returns_zero(parser: SQLParser) -> None:
    """An unparseable or trivial query yields the zero vector."""
    class _FakeGraph:
        node_attributes = {}

        class graph:  # noqa: D401
            @staticmethod
            def nodes():
                return []

    class _FakeQuery:
        def to_graph(self):
            return _FakeGraph()

    embedding = GraphEmbedding(_FakeQuery(), embedding_dim=32)
    vec = embedding.encode()
    assert np.allclose(vec, 0.0)
