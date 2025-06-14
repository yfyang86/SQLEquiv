# sql_equivalence/utils/__init__.py
"""Utility functions for SQL equivalence analysis."""

from .sql_utils import (
    format_sql, validate_sql, extract_tables_from_sql,
    extract_columns_from_sql, sql_to_lowercase_normalized,
    split_sql_statements, is_select_query, get_sql_type,
    standardize_sql, remove_sql_comments
)

from .graph_utils import (
    graph_to_adjacency_matrix, adjacency_matrix_to_graph,
    compute_graph_hash, find_common_subgraph,
    compute_graph_statistics, is_dag, topological_sort_graph,
    find_strongly_connected_components, compute_pagerank
)

from .algebra_utils import (
    simplify_algebraic_expression, evaluate_predicate,
    merge_predicates, split_conjunctive_predicate,
    is_predicate_satisfiable, normalize_predicate,
    get_predicate_tables, get_predicate_columns
)

from .visualization import (
    visualize_query_graph, visualize_expression_tree,
    visualize_algebraic_expression, create_query_comparison_plot,
    save_visualization, create_interactive_graph
)

__all__ = [
    # SQL utilities
    'format_sql', 'validate_sql', 'extract_tables_from_sql',
    'extract_columns_from_sql', 'sql_to_lowercase_normalized',
    'split_sql_statements', 'is_select_query', 'get_sql_type',
    'standardize_sql', 'remove_sql_comments',
    
    # Graph utilities
    'graph_to_adjacency_matrix', 'adjacency_matrix_to_graph',
    'compute_graph_hash', 'find_common_subgraph',
    'compute_graph_statistics', 'is_dag', 'topological_sort_graph',
    'find_strongly_connected_components', 'compute_pagerank',
    
    # Algebra utilities
    'simplify_algebraic_expression', 'evaluate_predicate',
    'merge_predicates', 'split_conjunctive_predicate',
    'is_predicate_satisfiable', 'normalize_predicate',
    'get_predicate_tables', 'get_predicate_columns',
    
    # Visualization utilities
    'visualize_query_graph', 'visualize_expression_tree',
    'visualize_algebraic_expression', 'create_query_comparison_plot',
    'save_visualization', 'create_interactive_graph',
]