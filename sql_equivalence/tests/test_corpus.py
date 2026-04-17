"""End-to-end corpus tests.

Exercises the full :class:`SQLEquivalenceAnalyzer` pipeline against the
shared fixture corpus. Two modes:

- Under the ``algebraic`` scope (``test_*_algebraic``): pins the
  algebra-only verdict. This catches regressions in the rule-based path
  without being confused by graph/embedding noise.
- Under the ``full`` scope (``test_*_overall``): pins the unanimous
  analyzer verdict (all three methods vote). This is the headline user
  contract.
"""

import pytest

from sql_equivalence import SQLEquivalenceAnalyzer

from .fixtures import EQUIVALENT_PAIRS, KNOWN_GAP_PAIRS, NON_EQUIVALENT_PAIRS


@pytest.fixture(scope='module')
def analyzer() -> SQLEquivalenceAnalyzer:
    return SQLEquivalenceAnalyzer()


# ---------------------------------------------------------------- algebraic
@pytest.mark.parametrize('sql1,sql2', EQUIVALENT_PAIRS)
def test_equivalent_pairs_algebraic(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    assert result.method_results['algebraic']['is_equivalent'] is True


# ------------------------------------------------------------------ overall
@pytest.mark.parametrize('sql1,sql2', EQUIVALENT_PAIRS)
def test_equivalent_pairs_overall(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)


@pytest.mark.parametrize('sql1,sql2', NON_EQUIVALENT_PAIRS)
def test_non_equivalent_pairs_overall(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    """The unanimous analyzer verdict must be *not* equivalent.

    We intentionally check the headline verdict rather than any one method
    because this is the contract users see. Individual methods may vote
    differently (they should -- their signals are complementary).
    """
    result = analyzer.analyze(sql1, sql2)
    assert result.is_equivalent is False


# --------------------------------------------------------------- known gaps
@pytest.mark.xfail(
    reason=(
        "Checkers do not yet distinguish: projection/predicate edits, "
        "JOIN kinds, UNION vs UNION ALL, DISTINCT, LIMIT values, or "
        "different aggregate functions."
    ),
    strict=True,
)
@pytest.mark.parametrize('sql1,sql2', KNOWN_GAP_PAIRS)
def test_known_gap_pairs_overall(
    analyzer: SQLEquivalenceAnalyzer, sql1: str, sql2: str
) -> None:
    result = analyzer.analyze(sql1, sql2)
    assert result.is_equivalent is False
