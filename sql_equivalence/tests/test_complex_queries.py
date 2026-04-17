"""Integration tests for complex SQL (CTEs, windows, set ops, subqueries).

These exist to (a) catch regressions in the parser/AST builder when
sqlglot's API shifts (we already burned once on ``Window.partition_by``),
and (b) confirm that the end-to-end analyzer pipeline doesn't crash on the
kinds of queries users actually write.
"""

import pytest

from sql_equivalence import SQLEquivalenceAnalyzer
from sql_equivalence.parser.sql_parser import SQLParser


@pytest.fixture(scope='module')
def analyzer() -> SQLEquivalenceAnalyzer:
    return SQLEquivalenceAnalyzer()


@pytest.fixture(scope='module')
def parser() -> SQLParser:
    return SQLParser(dialect='postgres')


# ---------------------------------------------------------------- CTEs
CTE_IDENTICAL = (
    "WITH adults AS (SELECT id, name FROM users WHERE age >= 18) "
    "SELECT id FROM adults",
    "WITH adults AS (SELECT id, name FROM users WHERE age >= 18) "
    "SELECT id FROM adults",
)

CTE_DIFFERENT_TABLE = (
    "WITH adults AS (SELECT id FROM users WHERE age >= 18) SELECT id FROM adults",
    "WITH adults AS (SELECT id FROM customers WHERE age >= 18) SELECT id FROM adults",
)


def test_cte_identical_parses(parser: SQLParser) -> None:
    parsed = parser.parse(CTE_IDENTICAL[0])
    assert parsed is not None
    assert parsed.to_graph().graph.number_of_nodes() > 0


def test_cte_identical_overall_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*CTE_IDENTICAL)
    assert result.is_equivalent is True


@pytest.mark.xfail(
    reason=(
        "CTE scaffolding nodes dominate the graph-similarity metric, so "
        "swapping just the base table name stays above the 0.9 threshold. "
        "Tightening requires weighting 'table' nodes higher or adding "
        "table-set comparison as a first-class signal."
    ),
    strict=True,
)
def test_cte_different_base_table_not_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*CTE_DIFFERENT_TABLE)
    assert result.is_equivalent is False


# ---------------------------------------------------------------- windows
WINDOW_IDENTICAL = (
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
    "FROM emp",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
    "FROM emp",
)

WINDOW_DIFFERENT_PARTITION = (
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) FROM emp",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY region ORDER BY salary) FROM emp",
)


def test_window_function_parses(parser: SQLParser) -> None:
    parsed = parser.parse(WINDOW_IDENTICAL[0])
    graph = parsed.to_graph()
    assert graph.graph.number_of_nodes() > 0


def test_window_identical_overall_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*WINDOW_IDENTICAL)
    assert result.is_equivalent is True


@pytest.mark.xfail(
    reason=(
        "PARTITION BY column names live deep in the AST; the value-histogram "
        "signal they contribute is drowned out by the rest of the window "
        "scaffolding. Tracked gap."
    ),
    strict=True,
)
def test_window_different_partition_not_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*WINDOW_DIFFERENT_PARTITION, methods=['graph'])
    assert result.method_results['graph']['is_equivalent'] is False


# ---------------------------------------------------------------- set ops
UNION_IDENTICAL = (
    "SELECT id FROM a UNION SELECT id FROM b",
    "SELECT id FROM a UNION SELECT id FROM b",
)

UNION_VS_INTERSECT = (
    "SELECT id FROM a UNION SELECT id FROM b",
    "SELECT id FROM a INTERSECT SELECT id FROM b",
)


def test_union_parses(parser: SQLParser) -> None:
    parsed = parser.parse(UNION_IDENTICAL[0])
    assert parsed.to_graph().graph.number_of_nodes() > 0


def test_union_identical_overall_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*UNION_IDENTICAL)
    assert result.is_equivalent is True


def test_union_vs_intersect_graph_distinguishes(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    """The graph method should flag UNION vs INTERSECT because the top
    node-type histogram differs."""
    result = analyzer.analyze(*UNION_VS_INTERSECT, methods=['graph'])
    assert result.method_results['graph']['is_equivalent'] is False


# ---------------------------------------------------------------- subqueries
SUBQUERY_IN_WHERE = (
    "SELECT id FROM a WHERE id IN (SELECT aid FROM b)",
    "SELECT id FROM a WHERE id IN (SELECT aid FROM b)",
)

SUBQUERY_EXISTS = (
    "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.aid = a.id)",
    "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.aid = a.id)",
)


def test_subquery_in_where_parses(parser: SQLParser) -> None:
    parsed = parser.parse(SUBQUERY_IN_WHERE[0])
    graph = parsed.to_graph()
    assert graph.graph.number_of_nodes() > 0


def test_subquery_in_where_identical(analyzer: SQLEquivalenceAnalyzer) -> None:
    result = analyzer.analyze(*SUBQUERY_IN_WHERE)
    assert result.is_equivalent is True


def test_subquery_exists_parses(parser: SQLParser) -> None:
    parsed = parser.parse(SUBQUERY_EXISTS[0])
    graph = parsed.to_graph()
    assert graph.graph.number_of_nodes() > 0


# ---------------------------------------------------------------- joins
MULTI_JOIN_IDENTICAL = (
    "SELECT a.id, b.name, c.price "
    "FROM a JOIN b ON a.id = b.aid "
    "JOIN c ON b.cid = c.id "
    "WHERE c.price > 100",
    "SELECT a.id, b.name, c.price "
    "FROM a JOIN b ON a.id = b.aid "
    "JOIN c ON b.cid = c.id "
    "WHERE c.price > 100",
)

EXTRA_JOIN = (
    "SELECT a.id FROM a JOIN b ON a.id = b.aid JOIN c ON b.cid = c.id",
    "SELECT a.id FROM a JOIN b ON a.id = b.aid",
)


def test_three_way_join_parses(parser: SQLParser) -> None:
    parsed = parser.parse(MULTI_JOIN_IDENTICAL[0])
    graph = parsed.to_graph()
    assert graph.graph.number_of_nodes() > 0


def test_three_way_join_identical_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*MULTI_JOIN_IDENTICAL)
    assert result.is_equivalent is True


def test_extra_join_overall_not_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    """Adding a JOIN changes the structure enough for graph/embedding to
    reject, even if the algebraic fallback misses it."""
    result = analyzer.analyze(*EXTRA_JOIN)
    assert result.is_equivalent is False


# ---------------------------------------------------------------- aggregation
AGG_IDENTICAL = (
    "SELECT dept, COUNT(*) AS c FROM emp GROUP BY dept HAVING COUNT(*) > 5",
    "SELECT dept, COUNT(*) AS c FROM emp GROUP BY dept HAVING COUNT(*) > 5",
)

AGG_WITHOUT_HAVING = (
    "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5",
    "SELECT dept, COUNT(*) FROM emp GROUP BY dept",
)


def test_aggregation_with_having_parses(parser: SQLParser) -> None:
    parsed = parser.parse(AGG_IDENTICAL[0])
    graph = parsed.to_graph()
    hist = graph.node_type_histogram()
    assert 'group_by' in hist
    assert 'having' in hist


def test_aggregation_identical_equivalent(analyzer: SQLEquivalenceAnalyzer) -> None:
    result = analyzer.analyze(*AGG_IDENTICAL)
    assert result.is_equivalent is True


def test_aggregation_with_vs_without_having_not_equivalent(
    analyzer: SQLEquivalenceAnalyzer,
) -> None:
    result = analyzer.analyze(*AGG_WITHOUT_HAVING)
    assert result.is_equivalent is False
