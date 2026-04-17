"""Tests for the algebraic representation / equivalence path."""

import pytest

from sql_equivalence.equivalence.algebraic_equivalence import (
    AlgebraicEquivalenceChecker,
)
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture(scope='module')
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


@pytest.fixture(scope='module')
def checker() -> AlgebraicEquivalenceChecker:
    return AlgebraicEquivalenceChecker()


# --------------------------------------------------- build
def test_algebraic_expression_builds(parser: SQLParser) -> None:
    expr = parser.parse("SELECT id FROM t WHERE id > 1").to_algebraic()
    assert expr is not None
    assert expr.to_canonical_form()


def test_algebraic_expression_caches_on_parsed_query(parser: SQLParser) -> None:
    """Asking for ``to_algebraic`` twice should return the same object."""
    parsed = parser.parse("SELECT id FROM t")
    first = parsed.to_algebraic()
    second = parsed.to_algebraic()
    assert first is second


# --------------------------------------------------- identity
def test_identical_queries_are_exactly_equivalent(
    parser: SQLParser, checker: AlgebraicEquivalenceChecker
) -> None:
    q = "SELECT a, b FROM t WHERE a > 1"
    expr1 = parser.parse(q).to_algebraic()
    expr2 = parser.parse(q).to_algebraic()
    result = checker.check_equivalence(expr1, expr2)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)


def test_identity_has_proof_step(
    parser: SQLParser, checker: AlgebraicEquivalenceChecker
) -> None:
    expr1 = parser.parse("SELECT id FROM t").to_algebraic()
    expr2 = parser.parse("SELECT id FROM t").to_algebraic()
    result = checker.check_equivalence(expr1, expr2)
    assert result.proof_steps


# --------------------------------------------------- non-equivalent
def test_structurally_different_queries_are_not_equivalent(
    parser: SQLParser, checker: AlgebraicEquivalenceChecker
) -> None:
    """Queries whose expression trees have different shapes (extra GROUP BY,
    extra JOIN, ...) should be flagged by the algebraic checker. The
    checker cannot yet distinguish queries that only differ in column or
    table *names* -- that gap is tracked in
    :data:`~sql_equivalence.tests.fixtures.KNOWN_GAP_PAIRS`.
    """
    expr1 = parser.parse("SELECT id FROM t").to_algebraic()
    expr2 = parser.parse(
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 1"
    ).to_algebraic()
    result = checker.check_equivalence(expr1, expr2)
    assert result.is_equivalent is False


def test_similarity_is_bounded(
    parser: SQLParser, checker: AlgebraicEquivalenceChecker
) -> None:
    expr1 = parser.parse("SELECT id FROM t").to_algebraic()
    expr2 = parser.parse(
        "SELECT a, b, c FROM other WHERE x > 1 GROUP BY a"
    ).to_algebraic()
    similarity = checker.compute_similarity(expr1, expr2)
    assert 0.0 <= similarity <= 1.0
