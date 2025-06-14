# sql_equivalence/transformations/__init__.py
"""Query transformation rules and optimizations."""

from .algebraic_rules import (
    AlgebraicRule,
    SelectionPushdown,
    ProjectionPushdown,
    JoinCommutativity,
    JoinAssociativity,
    SelectionSplit,
    SelectionCombine,
    ProjectionCascade,
    UnionCommutativity,
    UnionAssociativity,
    PredicateSimplification,
    RedundantJoinElimination,
    get_all_algebraic_rules,
    apply_algebraic_rules
)

from .graph_transformations import (
    GraphTransformation,
    NodeMerging,
    EdgeReduction,
    SubgraphExtraction,
    GraphNormalization,
    PathSimplification,
    CycleDetection,
    get_all_graph_transformations,
    apply_graph_transformations
)

from .optimization_rules import (
    OptimizationRule,
    IndexSuggestion,
    JoinOrderOptimization,
    SubqueryUnnesting,
    ViewMerging,
    PredicatePushdown,
    PartitionPruning,
    MaterializedViewRewriting,
    CostBasedOptimizer,
    get_optimization_rules,
    optimize_query
)

__all__ = [
    # Algebraic rules
    'AlgebraicRule',
    'SelectionPushdown',
    'ProjectionPushdown',
    'JoinCommutativity',
    'JoinAssociativity',
    'SelectionSplit',
    'SelectionCombine',
    'ProjectionCascade',
    'UnionCommutativity',
    'UnionAssociativity',
    'PredicateSimplification',
    'RedundantJoinElimination',
    'get_all_algebraic_rules',
    'apply_algebraic_rules',
    
    # Graph transformations
    'GraphTransformation',
    'NodeMerging',
    'EdgeReduction',
    'SubgraphExtraction',
    'GraphNormalization',
    'PathSimplification',
    'CycleDetection',
    'get_all_graph_transformations',
    'apply_graph_transformations',
    
    # Optimization rules
    'OptimizationRule',
    'IndexSuggestion',
    'JoinOrderOptimization',
    'SubqueryUnnesting',
    'ViewMerging',
    'PredicatePushdown',
    'PartitionPruning',
    'MaterializedViewRewriting',
    'CostBasedOptimizer',
    'get_optimization_rules',
    'optimize_query',
]