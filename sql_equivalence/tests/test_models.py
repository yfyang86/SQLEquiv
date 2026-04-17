"""Tests for the HashingVectorEncoder baseline."""

import numpy as np
import pytest

from sql_equivalence import SQLEquivalenceAnalyzer
from sql_equivalence.models import HashingVectorEncoder


def test_encoder_is_deterministic() -> None:
    encoder = HashingVectorEncoder(dim=64)
    v1 = encoder.encode("SELECT id FROM users WHERE id > 1")
    v2 = encoder.encode("SELECT id FROM users WHERE id > 1")
    np.testing.assert_array_equal(v1, v2)


def test_encoder_differs_for_different_queries() -> None:
    encoder = HashingVectorEncoder(dim=64)
    v1 = encoder.encode("SELECT id FROM users")
    v2 = encoder.encode("DROP TABLE users")
    assert not np.array_equal(v1, v2)


def test_encoder_output_is_l2_normalized() -> None:
    encoder = HashingVectorEncoder(dim=64)
    vec = encoder.encode("SELECT * FROM t")
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


def test_analyzer_can_use_custom_encoder() -> None:
    encoder = HashingVectorEncoder(dim=128)
    analyzer = SQLEquivalenceAnalyzer(embedding_model=encoder)
    result = analyzer.analyze(
        "SELECT id FROM users", "SELECT id FROM users", methods=['embedding']
    )
    assert result.method_results['embedding']['is_equivalent'] is True
