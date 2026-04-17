"""Tests for the QueryGraph / LogicalQueryTree / GraphEquivalence stack."""

import pytest

from sql_equivalence.equivalence.graph_equivalence import GraphEquivalenceChecker
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


def _graphs(parser: SQLParser, sql1: str, sql2: str):
    return parser.parse(sql1).to_graph(), parser.parse(sql2).to_graph()


def test_query_graph_has_nodes_after_build(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users WHERE id > 1").to_graph()
    assert graph.graph.number_of_nodes() > 0
    assert graph.root_node is not None


def test_query_graph_histogram_has_select(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users").to_graph()
    histogram = graph.node_type_histogram()
    assert histogram.get('select', 0) >= 1


def test_isomorphic_identical_queries(parser: SQLParser) -> None:
    checker = GraphEquivalenceChecker()
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    result = checker.check_equivalence(g1, g2)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)


def test_non_isomorphic_queries_are_not_equivalent(parser: SQLParser) -> None:
    checker = GraphEquivalenceChecker()
    g1, g2 = _graphs(
        parser,
        "SELECT id FROM t",
        "SELECT a, b, c FROM users WHERE age > 10 GROUP BY a",
    )
    result = checker.check_equivalence(g1, g2)
    assert result.is_equivalent is False
    assert 0.0 <= result.confidence <= 1.0


def test_logical_query_tree_has_root(parser: SQLParser) -> None:
    lqt = parser.parse("SELECT id FROM users").to_lqt()
    assert lqt.root_node is not None
    assert lqt.tree.number_of_nodes() > 0
    assert lqt.get_height() >= 1


def test_subgraph_isomorphism_trivial(parser: SQLParser) -> None:
    checker = GraphEquivalenceChecker()
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    assert checker.check_subgraph_isomorphism(g1, g2) is True
