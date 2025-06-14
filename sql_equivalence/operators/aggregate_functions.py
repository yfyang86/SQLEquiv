# sql_equivalence/operators/aggregate_functions.py
"""Aggregate function operators."""

from typing import Any, Dict, List, Optional, Set

from .base_operator import (
    FunctionOperator, OperatorCategory, OperatorProperties,
    register_operator
)

class AggregateFunction(FunctionOperator):
    """Base class for aggregate functions."""
    
    def __init__(self, properties: OperatorProperties,
                 arguments: Optional[List[Any]] = None,
                 distinct: bool = False):
        properties.is_aggregate = True
        super().__init__(properties, arguments)
        self.distinct = distinct
        self.parameters['distinct'] = distinct
    
    def set_distinct(self, distinct: bool) -> None:
        """Set DISTINCT flag for aggregation."""
        self.distinct = distinct
        self.parameters['distinct'] = distinct
        self._hash = None
    
    def get_output_schema(self, input_schemas: List[Set[str]]) -> Set[str]:
        """Aggregate functions produce a single value."""
        # The output column name is typically the function call itself
        # or an alias if provided
        return {self.to_sql()}

class SumFunction(AggregateFunction):
    """SUM aggregate function."""
    
    def __init__(self, column: Optional[Any] = None, distinct: bool = False):
        properties = OperatorProperties(
            name="SUM",
            category=OperatorCategory.AGGREGATE,
            arity=1,
            is_aggregate=True,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        arguments = [column] if column is not None else None
        super().__init__(properties, arguments, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "SUM()"
        
        arg_sql = self.arguments[0].to_sql(dialect) if hasattr(self.arguments[0], 'to_sql') else str(self.arguments[0])
        distinct_str = "DISTINCT " if self.distinct else ""
        return f"SUM({distinct_str}{arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'SUM',
            'distinct': self.distinct,
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'SumFunction':
        arg_clone = None
        if self.arguments:
            arg_clone = self.arguments[0].clone() if hasattr(self.arguments[0], 'clone') else self.arguments[0]
        return SumFunction(arg_clone, self.distinct)

class CountFunction(AggregateFunction):
    """COUNT aggregate function."""
    
    def __init__(self, column: Optional[Any] = None, distinct: bool = False):
        properties = OperatorProperties(
            name="COUNT",
            category=OperatorCategory.AGGREGATE,
            arity=1,
            is_aggregate=True,
            is_deterministic=True,
            output_type="INTEGER"
        )
        arguments = [column] if column is not None else None
        super().__init__(properties, arguments, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "COUNT(*)"
        
        arg = self.arguments[0]
        if arg == '*':
            return "COUNT(*)"
        
        arg_sql = arg.to_sql(dialect) if hasattr(arg, 'to_sql') else str(arg)
        distinct_str = "DISTINCT " if self.distinct else ""
        return f"COUNT({distinct_str}{arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'COUNT',
            'distinct': self.distinct,
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'CountFunction':
        arg_clone = None
        if self.arguments:
            arg = self.arguments[0]
            arg_clone = arg.clone() if hasattr(arg, 'clone') else arg
        return CountFunction(arg_clone, self.distinct)

class AvgFunction(AggregateFunction):
    """AVG aggregate function."""
    
    def __init__(self, column: Optional[Any] = None, distinct: bool = False):
        properties = OperatorProperties(
            name="AVG",
            category=OperatorCategory.AGGREGATE,
            arity=1,
            is_aggregate=True,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        arguments = [column] if column is not None else None
        super().__init__(properties, arguments, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "AVG()"
        
        arg_sql = self.arguments[0].to_sql(dialect) if hasattr(self.arguments[0], 'to_sql') else str(self.arguments[0])
        distinct_str = "DISTINCT " if self.distinct else ""
        return f"AVG({distinct_str}{arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'AVG',
            'distinct': self.distinct,
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'AvgFunction':
        arg_clone = None
        if self.arguments:
            arg_clone = self.arguments[0].clone() if hasattr(self.arguments[0], 'clone') else self.arguments[0]
        return AvgFunction(arg_clone, self.distinct)

class MinFunction(AggregateFunction):
    """MIN aggregate function."""
    
    def __init__(self, column: Optional[Any] = None):
        properties = OperatorProperties(
            name="MIN",
            category=OperatorCategory.AGGREGATE,
            arity=1,
            is_aggregate=True,
            is_deterministic=True,
            output_type="ANY"  # Same type as input
        )
        arguments = [column] if column is not None else None
        # MIN doesn't support DISTINCT
        super().__init__(properties, arguments, distinct=False)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "MIN()"
        
        arg_sql = self.arguments[0].to_sql(dialect) if hasattr(self.arguments[0], 'to_sql') else str(self.arguments[0])
        return f"MIN({arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'MIN',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'MinFunction':
        arg_clone = None
        if self.arguments:
            arg_clone = self.arguments[0].clone() if hasattr(self.arguments[0], 'clone') else self.arguments[0]
        return MinFunction(arg_clone)

class MaxFunction(AggregateFunction):
    """MAX aggregate function."""
    
    def __init__(self, column: Optional[Any] = None):
        properties = OperatorProperties(
            name="MAX",
            category=OperatorCategory.AGGREGATE,
            arity=1,
            is_aggregate=True,
            is_deterministic=True,
            output_type="ANY"  # Same type as input
        )
        arguments = [column] if column is not None else None
        # MAX doesn't support DISTINCT
        super().__init__(properties, arguments, distinct=False)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "MAX()"
        
        arg_sql = self.arguments[0].to_sql(dialect) if hasattr(self.arguments[0], 'to_sql') else str(self.arguments[0])
        return f"MAX({arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'MAX',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'MaxFunction':
        arg_clone = None
        if self.arguments:
            arg_clone = self.arguments[0].clone() if hasattr(self.arguments[0], 'clone') else self.arguments[0]
        return MaxFunction(arg_clone)

# Register aggregate functions
register_operator('SUM', SumFunction)
register_operator('COUNT', CountFunction)
register_operator('AVG', AvgFunction)
register_operator('MIN', MinFunction)
register_operator('MAX', MaxFunction)