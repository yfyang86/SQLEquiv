"""Integration tests for the top-level SQLEquivalenceAnalyzer."""

import pytest

from sql_equivalence import SQLEquivalenceAnalyzer


@pytest.fixture(scope='module')
def analyzer() -> SQLEquivalenceAnalyzer:
    return SQLEquivalenceAnalyzer()


def test_identical_queries_all_methods_agree(analyzer: SQLEquivalenceAnalyzer) -> None:
    sql = "SELECT id, name FROM users WHERE id > 1"
    result = analyzer.analyze(sql, sql)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)
    for method, outcome in result.method_results.items():
        assert outcome['is_equivalent'] is True, (
            f"Method {method} failed to recognize identical queries"
        )


def test_clearly_different_queries(analyzer: SQLEquivalenceAnalyzer) -> None:
    result = analyzer.analyze(
        "SELECT id FROM t",
        "SELECT a, b, c FROM users WHERE age > 10 GROUP BY a",
    )
    assert result.is_equivalent is False
    assert result.confidence < 1.0


def test_unknown_method_raises(analyzer: SQLEquivalenceAnalyzer) -> None:
    with pytest.raises(ValueError, match='Unknown analysis method'):
        analyzer.analyze("SELECT 1", "SELECT 1", methods=['nope'])


def test_single_method_scope(analyzer: SQLEquivalenceAnalyzer) -> None:
    result = analyzer.analyze(
        "SELECT id FROM t",
        "SELECT id FROM t",
        methods=['algebraic'],
    )
    assert set(result.method_results) == {'algebraic'}
    assert result.is_equivalent is True


def test_detailed_exposes_canonical_form(analyzer: SQLEquivalenceAnalyzer) -> None:
    result = analyzer.analyze(
        "SELECT id FROM t", "SELECT id FROM t", methods=['algebraic'], detailed=True
    )
    alg = result.method_results['algebraic']
    assert 'canonical_form1' in alg
    assert 'canonical_form2' in alg
