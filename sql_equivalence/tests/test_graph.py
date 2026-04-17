"""Deep tests for the QueryGraph / LogicalQueryTree / GraphEquivalence stack."""

import pytest

from sql_equivalence.equivalence.graph_equivalence import GraphEquivalenceChecker
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture(scope='module')
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


@pytest.fixture(scope='module')
def checker() -> GraphEquivalenceChecker:
    return GraphEquivalenceChecker()


def _graphs(parser: SQLParser, sql1: str, sql2: str):
    return parser.parse(sql1).to_graph(), parser.parse(sql2).to_graph()


# ------------------------------------------------------- QueryGraph.build
def test_query_graph_has_nodes_after_build(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users WHERE id > 1").to_graph()
    assert graph.graph.number_of_nodes() > 0
    assert graph.root_node is not None


def test_query_graph_nodes_carry_type_attribute(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users").to_graph()
    for attrs in graph.node_attributes.values():
        assert 'type' in attrs


def test_query_graph_histogram_is_well_formed(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id, name FROM users WHERE age > 10").to_graph()
    histogram = graph.node_type_histogram()
    assert histogram.get('select', 0) >= 1
    # Sum of histogram == node count
    assert sum(histogram.values()) == graph.graph.number_of_nodes()


def test_iter_nodes_of_type_returns_only_matching(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users WHERE age > 10").to_graph()
    where_nodes = list(graph.iter_nodes_of_type('where'))
    for node_id in where_nodes:
        assert graph.node_attributes[node_id]['type'] == 'where'


def test_query_graph_complex_query_produces_more_nodes(parser: SQLParser) -> None:
    simple = parser.parse("SELECT id FROM t").to_graph()
    complex_ = parser.parse(
        "SELECT a.id, COUNT(*) AS c FROM a JOIN b ON a.id = b.aid "
        "WHERE age > 18 GROUP BY a.id HAVING COUNT(*) > 3 "
        "ORDER BY c DESC LIMIT 10"
    ).to_graph()
    assert complex_.graph.number_of_nodes() > simple.graph.number_of_nodes()


def test_to_adjacency_matrix_dimensions(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id FROM users").to_graph()
    n = graph.graph.number_of_nodes()
    mat = graph.to_adjacency_matrix()
    assert mat.shape == (n, n)


def test_get_subgraph_is_proper_subset(parser: SQLParser) -> None:
    graph = parser.parse("SELECT id, name FROM users WHERE age > 10").to_graph()
    nodes = list(graph.graph.nodes())[: max(1, len(graph.graph.nodes()) // 2)]
    sub = graph.get_subgraph(nodes)
    assert sub.graph.number_of_nodes() <= graph.graph.number_of_nodes()
    for n in sub.graph.nodes():
        assert n in graph.graph.nodes()


# ------------------------------------------------------- quick_reject
def test_quick_reject_catches_size_difference(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT a, b, c FROM u WHERE a>1")
    assert checker._quick_reject(g1, g2) is True


def test_quick_reject_passes_identical(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    assert checker._quick_reject(g1, g2) is False


# ------------------------------------------------------- isomorphism
def test_isomorphic_identical_queries(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    result = checker.check_equivalence(g1, g2)
    assert result.is_equivalent is True
    assert result.confidence == pytest.approx(1.0)


def test_non_isomorphic_queries(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(
        parser,
        "SELECT id FROM t",
        "SELECT a, b, c FROM users WHERE age > 10 GROUP BY a",
    )
    result = checker.check_equivalence(g1, g2)
    assert result.is_equivalent is False


def test_isomorphism_disregards_table_name_when_node_attrs_off(
    parser: SQLParser,
) -> None:
    """Turning off node-attribute matching should ignore table names.

    This documents an intentional lever: ``use_node_attributes=False`` makes
    isomorphism structural-only. Two queries with the same *shape* but
    different tables are then considered isomorphic.
    """
    lenient = GraphEquivalenceChecker(
        use_node_attributes=False, use_edge_attributes=False
    )
    g1, g2 = _graphs(parser, "SELECT id FROM users", "SELECT id FROM customers")
    assert lenient._check_isomorphism(g1, g2) is True


# ------------------------------------------------------- similarity
def test_similarity_is_symmetric(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(
        parser,
        "SELECT id FROM a WHERE x > 1",
        "SELECT name FROM b WHERE y < 2 GROUP BY x",
    )
    assert checker.compute_similarity(g1, g2) == pytest.approx(
        checker.compute_similarity(g2, g1)
    )


def test_similarity_is_bounded(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM a", "SELECT name FROM b WHERE y < 2")
    sim = checker.compute_similarity(g1, g2)
    assert 0.0 <= sim <= 1.0


def test_similarity_of_identical_is_one(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    assert checker.compute_similarity(g1, g2) == pytest.approx(1.0)


def test_similarity_monotonicity_under_diff(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    """A single-edit difference should score closer than a total rewrite."""
    base = parser.parse("SELECT id FROM users WHERE age > 18").to_graph()
    near = parser.parse("SELECT id FROM users WHERE age > 21").to_graph()
    far = parser.parse("SELECT name FROM products").to_graph()
    assert checker.compute_similarity(base, near) > checker.compute_similarity(
        base, far
    )


# ------------------------------------------------------- GED
def test_graph_edit_distance_of_identical_is_zero(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    # Use a short timeout; identical graphs should return 0 immediately.
    distance = checker.compute_graph_edit_distance(g1, g2)
    assert distance == 0.0


def test_subgraph_isomorphism_holds_for_identical(
    parser: SQLParser, checker: GraphEquivalenceChecker
) -> None:
    g1, g2 = _graphs(parser, "SELECT id FROM t", "SELECT id FROM t")
    assert checker.check_subgraph_isomorphism(g1, g2) is True


# ------------------------------------------------------- LogicalQueryTree
def test_logical_query_tree_has_root(parser: SQLParser) -> None:
    lqt = parser.parse("SELECT id FROM users").to_lqt()
    assert lqt.root_node is not None
    assert lqt.tree.number_of_nodes() > 0


def test_lqt_height_at_least_one(parser: SQLParser) -> None:
    lqt = parser.parse("SELECT id FROM t WHERE x > 1").to_lqt()
    assert lqt.get_height() >= 1


def test_lqt_leaves_have_out_degree_zero(parser: SQLParser) -> None:
    lqt = parser.parse("SELECT a, b FROM users WHERE age > 18").to_lqt()
    for leaf in lqt.get_leaves():
        assert lqt.tree.out_degree(leaf) == 0


def test_lqt_iter_subtree_visits_all_from_root(parser: SQLParser) -> None:
    lqt = parser.parse("SELECT id FROM t").to_lqt()
    visited = set(lqt.iter_subtree())
    assert visited == set(lqt.tree.nodes())
