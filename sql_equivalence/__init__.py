# sql_equivalence/__init__.py
"""
SQL Equivalence Analysis Library

A comprehensive library for analyzing SQL query equivalence using algebraic,
graph-based, and embedding-based approaches.
"""

__version__ = "0.1.0"
__author__ = "Yifan Yang"
__email__ = "yifan.yang@transwarp.io"

# Core components
# Main analyzer
from .analyzer import SQLEquivalenceAnalyzer

# Equivalence checkers
from .equivalence.algebraic_equivalence import AlgebraicEquivalenceChecker
from .equivalence.embedding_similarity import EmbeddingSimilarityChecker
from .equivalence.graph_equivalence import GraphEquivalenceChecker
from .operators.aggregate_functions import (
    AvgFunction,
    CountFunction,
    MaxFunction,
    MinFunction,
    SumFunction,
)

# Operators
from .operators.relational_operators import (
    FromOperator,
    GroupByOperator,
    JoinOperator,
    OrderByOperator,
    SelectOperator,
    WhereOperator,
)
from .operators.set_operators import ExceptOperator, IntersectOperator, UnionOperator
from .operators.window_functions import (
    DenseRankFunction,
    LagFunction,
    LeadFunction,
    RankFunction,
    RowNumberFunction,
)
from .parser.ast_builder import ASTBuilder
from .parser.normalizer import SQLNormalizer
from .parser.sql_parser import ParsedQuery, SQLParser

# Plugin registry for custom analysis methods
from .plugins import register_method, registered_names, unregister_method
from .representations.algebraic.expression_tree import ExpressionTree

# Representations
from .representations.algebraic.relational_algebra import AlgebraicExpression
from .representations.embedding.encoder import QueryEncoder
from .representations.embedding.graph_embedding import GraphEmbedding
from .representations.graph.lqt import LogicalQueryTree
from .representations.graph.query_graph import QueryGraph

# Utilities
from .utils.sql_utils import format_sql, validate_sql
from .utils.visualization import visualize_expression_tree, visualize_query_graph

__all__ = [
    # Version info
    "__version__",

    # Core classes
    "SQLParser",
    "ParsedQuery",
    "SQLEquivalenceAnalyzer",

    # Representations
    "AlgebraicExpression",
    "QueryGraph",
    "LogicalQueryTree",
    "GraphEmbedding",

    # Equivalence checkers
    "AlgebraicEquivalenceChecker",
    "GraphEquivalenceChecker",
    "EmbeddingSimilarityChecker",

    # Utilities
    "format_sql",
    "validate_sql",
    "visualize_query_graph",
    "visualize_expression_tree",

    # Plugin registry
    "register_method",
    "unregister_method",
    "registered_names",
]
