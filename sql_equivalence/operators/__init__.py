# sql_equivalence/operators/__init__.py
"""SQL operators and functions module."""

from .aggregate_functions import (
    AggregateFunction,
    AvgFunction,
    CountFunction,
    MaxFunction,
    MinFunction,
    SumFunction,
)
from .base_operator import (
    BaseOperator,
    BinaryOperator,
    FunctionOperator,
    OperatorCategory,
    OperatorProperties,
    UnaryOperator,
)
from .relational_operators import (
    FromOperator,
    GroupByOperator,
    HavingOperator,
    JoinOperator,
    LimitOperator,
    OrderByOperator,
    SelectOperator,
    WhereOperator,
)
from .scalar_functions import (
    AbsFunction,
    CeilFunction,
    ExpFunction,
    FloorFunction,
    LengthFunction,
    LogFunction,
    LowerFunction,
    RoundFunction,
    ScalarFunction,
    SubstringFunction,
    TrimFunction,
    UpperFunction,
)
from .set_operators import ExceptOperator, IntersectOperator, SetOperator, UnionOperator
from .window_functions import (
    DenseRankFunction,
    LagFunction,
    LeadFunction,
    NtileFunction,
    RankFunction,
    RowNumberFunction,
    WindowFunction,
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
