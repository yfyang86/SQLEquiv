"""End-to-end corpus tests.

Exercises the full :class:`SQLEquivalenceAnalyzer` pipeline against the
shared fixture corpus. These act as the phase-B regression gate: if a rule
change silently breaks a previously-recognized equivalence, this test catches
it.
"""

import pytest

from sql_equivalence import SQLEquivalenceAnalyzer

from .fixtures import (
    EQUIVALENT_PAIRS,
    KNOWN_GAP_NON_EQUIVALENT_PAIRS,
    NON_EQUIVALENT_PAIRS,
)


@pytest.fixture(scope='module')
def analyzer() -> SQLEquivalenceAnalyzer:
    return SQLEquivalenceAnalyzer()


@pytest.mark.parametrize('sql1,sql2', EQUIVALENT_PAIRS)
def test_equivalent_pairs(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    assert result.method_results['algebraic']['is_equivalent'] is True


@pytest.mark.parametrize('sql1,sql2', NON_EQUIVALENT_PAIRS)
def test_non_equivalent_pairs(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    assert result.method_results['algebraic']['is_equivalent'] is False


@pytest.mark.xfail(
    reason="Algebraic rules do not yet distinguish projection/predicate edits",
    strict=True,
)
@pytest.mark.parametrize('sql1,sql2', KNOWN_GAP_NON_EQUIVALENT_PAIRS)
def test_known_gap_non_equivalent_pairs(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    assert result.method_results['algebraic']['is_equivalent'] is False
