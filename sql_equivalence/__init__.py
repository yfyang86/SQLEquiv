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
from .parser.sql_parser import SQLParser, ParsedQuery
from .parser.ast_builder import ASTBuilder
from .parser.normalizer import SQLNormalizer

# Representations
from .representations.algebraic.relational_algebra import AlgebraicExpression
from .representations.algebraic.expression_tree import ExpressionTree
from .representations.graph.query_graph import QueryGraph
from .representations.graph.lqt import LogicalQueryTree
from .representations.embedding.encoder import QueryEncoder
from .representations.embedding.graph_embedding import GraphEmbedding

# Equivalence checkers
from .equivalence.algebraic_equivalence import AlgebraicEquivalenceChecker
from .equivalence.graph_equivalence import GraphEquivalenceChecker
from .equivalence.embedding_similarity import EmbeddingSimilarityChecker

# Main analyzer
from .analyzer import SQLEquivalenceAnalyzer

# Operators
from .operators.relational_operators import (
    SelectOperator, FromOperator, WhereOperator, 
    JoinOperator, GroupByOperator, OrderByOperator
)
from .operators.set_operators import UnionOperator, IntersectOperator, ExceptOperator
from .operators.aggregate_functions import (
    SumFunction, CountFunction, AvgFunction, 
    MinFunction, MaxFunction
)
from .operators.window_functions import (
    RowNumberFunction, RankFunction, DenseRankFunction,
    LeadFunction, LagFunction
)

# Utilities
from .utils.sql_utils import format_sql, validate_sql
from .utils.visualization import visualize_query_graph, visualize_expression_tree

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
]