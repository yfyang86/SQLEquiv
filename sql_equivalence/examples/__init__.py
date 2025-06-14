# sql_equivalence/examples/__init__.py
"""Example usage of SQL equivalence analysis library."""

from .basic_examples import (
    example_simple_equivalence,
    example_join_reordering,
    example_predicate_pushdown,
    example_projection_elimination,
    example_set_operations,
    example_aggregate_queries
)

from .advanced_examples import (
    example_complex_joins,
    example_subqueries,
    example_window_functions,
    example_cte_analysis,
    example_batch_analysis,
    example_visualization
)

__all__ = [
    # Basic examples
    'example_simple_equivalence',
    'example_join_reordering',
    'example_predicate_pushdown',
    'example_projection_elimination',
    'example_set_operations',
    'example_aggregate_queries',
    
    # Advanced examples
    'example_complex_joins',
    'example_subqueries',
    'example_window_functions',
    'example_cte_analysis',
    'example_batch_analysis',
    'example_visualization',
]