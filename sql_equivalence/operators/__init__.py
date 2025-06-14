# sql_equivalence/operators/__init__.py
"""SQL operators and functions module."""

from .base_operator import (
    BaseOperator, OperatorCategory, OperatorProperties,
    BinaryOperator, UnaryOperator, FunctionOperator
)

from .relational_operators import (
    SelectOperator, FromOperator, WhereOperator,
    JoinOperator, GroupByOperator, HavingOperator,
    OrderByOperator, LimitOperator
)

from .set_operators import (
    UnionOperator, IntersectOperator, ExceptOperator,
    SetOperator
)

from .aggregate_functions import (
    SumFunction, CountFunction, AvgFunction,
    MinFunction, MaxFunction, AggregateFunction
)

from .window_functions import (
    RowNumberFunction, RankFunction, DenseRankFunction,
    NtileFunction, LeadFunction, LagFunction,
    WindowFunction
)

from .scalar_functions import (
    UpperFunction, LowerFunction, TrimFunction,
    SubstringFunction, LengthFunction,
    ExpFunction, LogFunction, AbsFunction,
    RoundFunction, CeilFunction, FloorFunction,
    ScalarFunction
)

__all__ = [
    # Base classes
    'BaseOperator', 'OperatorCategory', 'OperatorProperties',
    'BinaryOperator', 'UnaryOperator', 'FunctionOperator',
    
    # Relational operators
    'SelectOperator', 'FromOperator', 'WhereOperator',
    'JoinOperator', 'GroupByOperator', 'HavingOperator',
    'OrderByOperator', 'LimitOperator',
    
    # Set operators
    'UnionOperator', 'IntersectOperator', 'ExceptOperator',
    'SetOperator',
    
    # Aggregate functions
    'SumFunction', 'CountFunction', 'AvgFunction',
    'MinFunction', 'MaxFunction', 'AggregateFunction',
    
    # Window functions
    'RowNumberFunction', 'RankFunction', 'DenseRankFunction',
    'NtileFunction', 'LeadFunction', 'LagFunction',
    'WindowFunction',
    
    # Scalar functions
    'UpperFunction', 'LowerFunction', 'TrimFunction',
    'SubstringFunction', 'LengthFunction',
    'ExpFunction', 'LogFunction', 'AbsFunction',
    'RoundFunction', 'CeilFunction', 'FloorFunction',
    'ScalarFunction',
]