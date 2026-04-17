"""Smoke tests for the SQL parser module."""

import pytest

from sql_equivalence.parser.normalizer import SQLNormalizer
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


def test_parse_simple_select(parser: SQLParser) -> None:
    parsed = parser.parse("SELECT id, name FROM users WHERE id > 10")
    assert parsed is not None
    assert parsed.sql.strip().lower().startswith("select")


def test_validate_sql_accepts_valid(parser: SQLParser) -> None:
    ok, err = parser.validate_sql("SELECT 1")
    assert ok is True
    assert err is None


def test_validate_sql_rejects_invalid(parser: SQLParser) -> None:
    ok, err = parser.validate_sql("SELEKT ))) FROM")
    assert ok is False
    assert err


def test_normalizer_whitespace_and_case() -> None:
    normalizer = SQLNormalizer()
    out = normalizer.normalize("select  id , name   from    users")
    assert "SELECT" in out
    assert "  " not in out
