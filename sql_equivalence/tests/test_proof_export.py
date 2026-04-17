"""Tests for sql_equivalence.proof_export."""

import json

from sql_equivalence import SQLEquivalenceAnalyzer
from sql_equivalence.proof_export import to_json, to_markdown


def test_markdown_contains_verdict_and_method_sections() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    result = analyzer.analyze(
        "SELECT id FROM t", "SELECT id FROM t", methods=['algebraic'], detailed=True
    )
    md = to_markdown(result)
    assert "Verdict" in md
    assert "algebraic" in md
    assert "SELECT id FROM t" in md


def test_json_roundtrips_to_dict() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    result = analyzer.analyze("SELECT 1", "SELECT 1", methods=['algebraic'])
    serialized = to_json(result)
    loaded = json.loads(serialized)
    assert loaded['is_equivalent'] is True
    assert 'method_results' in loaded


def test_markdown_flags_non_equivalent() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    result = analyzer.analyze(
        "SELECT id FROM t", "SELECT id FROM customers", methods=['algebraic']
    )
    md = to_markdown(result)
    assert "NOT EQUIVALENT" in md
