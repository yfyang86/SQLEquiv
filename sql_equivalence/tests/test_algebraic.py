"""Tests for the algebraic representation / equivalence path."""

import pytest

from sql_equivalence.equivalence.algebraic_equivalence import (
    AlgebraicEquivalenceChecker,
)
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


def test_algebraic_expression_builds(parser: SQLParser) -> None:
    expr = parser.parse("SELECT id FROM t WHERE id > 1").to_algebraic()
    assert expr is not None
    assert expr.to_canonical_form()


def test_identical_queries_are_exactly_equivalent(parser: SQLParser) -> None:
    checker = AlgebraicEquivalenceChecker()
    q = "SELECT a, b FROM t WHERE a > 1"
    expr1 = parser.parse(q).to_algebraic()
    expr2 = parser.parse(q).to_algebraic()
    result = checker.check_equivalence(expr1, expr2)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)


def test_different_queries_are_not_equivalent(parser: SQLParser) -> None:
    checker = AlgebraicEquivalenceChecker()
    expr1 = parser.parse("SELECT a FROM t").to_algebraic()
    expr2 = parser.parse("SELECT b FROM other").to_algebraic()
    result = checker.check_equivalence(expr1, expr2)
    assert result.is_equivalent is False
