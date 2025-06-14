# sql_equivalence/operators/window_functions.py
"""Window function operators."""

from typing import Any, Dict, List, Optional

from .base_operator import (
    FunctionOperator, OperatorCategory, OperatorProperties,
    register_operator
)

class WindowFunction(FunctionOperator):
    """Base class for window functions."""
    
    def __init__(self, properties: OperatorProperties,
                 arguments: Optional[List[Any]] = None):
        super().__init__(properties, arguments)
        self.partition_by: List[Any] = []
        self.order_by: List[Dict[str, Any]] = []
        self.frame_spec: Optional[Dict[str, Any]] = None
    
    def set_partition_by(self, columns: List[Any]) -> None:
        """Set PARTITION BY clause."""
        self.partition_by = columns
        self.parameters['partition_by'] = columns
        self._hash = None
    
    def set_order_by(self, columns: List[Dict[str, Any]]) -> None:
        """Set ORDER BY clause."""
        self.order_by = columns
        self.parameters['order_by'] = columns
        self._hash = None
    
    def set_frame(self, frame_spec: Dict[str, Any]) -> None:
        """Set window frame specification."""
        self.frame_spec = frame_spec
        self.parameters['frame_spec'] = frame_spec
        self._hash = None
    
    def _window_clause_sql(self, dialect: str = 'standard') -> str:
        """Generate OVER clause SQL."""
        parts = []
        
        if self.partition_by:
            partition_strs = []
            for col in self.partition_by:
                if hasattr(col, 'to_sql'):
                    partition_strs.append(col.to_sql(dialect))
                else:
                    partition_strs.append(str(col))
            parts.append(f"PARTITION BY {', '.join(partition_strs)}")
        
        if self.order_by:
            order_strs = []
            for col_spec in self.order_by:
                col_str = col_spec['column']
                if hasattr(col_spec['column'], 'to_sql'):
                    col_str = col_spec['column'].to_sql(dialect)
                
                if col_spec.get('direction', 'ASC').upper() == 'DESC':
                    col_str += " DESC"
                order_strs.append(col_str)
            parts.append(f"ORDER BY {', '.join(order_strs)}")
        
        if self.frame_spec:
            # Frame specification (e.g., ROWS BETWEEN ... AND ...)
            frame_str = self._build_frame_sql(self.frame_spec)
            if frame_str:
                parts.append(frame_str)
        
        return f"OVER ({' '.join(parts)})" if parts else "OVER ()"
    
    def _build_frame_sql(self, frame_spec: Dict[str, Any]) -> str:
        """Build frame specification SQL."""
        # Simplified implementation
        # Full implementation would handle all frame types
        return ""

class RowNumberFunction(WindowFunction):
    """ROW_NUMBER window function."""
    
    def __init__(self):
        properties = OperatorProperties(
            name="ROW_NUMBER",
            category=OperatorCategory.WINDOW,
            arity=0,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, arguments=[])
    
    def to_sql(self, dialect: str = 'standard') -> str:
        return f"ROW_NUMBER() {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ROW_NUMBER',
            'partition_by': self.partition_by,
            'order_by': self.order_by,
            'frame_spec': self.frame_spec
        }
    
    def clone(self) -> 'RowNumberFunction':
        import copy
        func = RowNumberFunction()
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        func.frame_spec = copy.deepcopy(self.frame_spec)
        return func

class RankFunction(WindowFunction):
    """RANK window function."""
    
    def __init__(self):
        properties = OperatorProperties(
            name="RANK",
            category=OperatorCategory.WINDOW,
            arity=0,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, arguments=[])
    
    def to_sql(self, dialect: str = 'standard') -> str:
        return f"RANK() {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'RANK',
            'partition_by': self.partition_by,
            'order_by': self.order_by
        }
    
    def clone(self) -> 'RankFunction':
        import copy
        func = RankFunction()
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        return func

class DenseRankFunction(WindowFunction):
    """DENSE_RANK window function."""
    
    def __init__(self):
        properties = OperatorProperties(
            name="DENSE_RANK",
            category=OperatorCategory.WINDOW,
            arity=0,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, arguments=[])
    
    def to_sql(self, dialect: str = 'standard') -> str:
        return f"DENSE_RANK() {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'DENSE_RANK',
            'partition_by': self.partition_by,
            'order_by': self.order_by
        }
    
    def clone(self) -> 'DenseRankFunction':
        import copy
        func = DenseRankFunction()
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        return func

class NtileFunction(WindowFunction):
    """NTILE window function."""
    
    def __init__(self, buckets: Optional[int] = None):
        properties = OperatorProperties(
            name="NTILE",
            category=OperatorCategory.WINDOW,
            arity=1,
            is_deterministic=True,
            output_type="INTEGER"
        )
        arguments = [buckets] if buckets is not None else []
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        buckets_str = str(self.arguments[0]) if self.arguments else "1"
        return f"NTILE({buckets_str}) {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'NTILE',
            'buckets': self.arguments[0] if self.arguments else None,
            'partition_by': self.partition_by,
            'order_by': self.order_by
        }
    
    def clone(self) -> 'NtileFunction':
        import copy
        func = NtileFunction(self.arguments[0] if self.arguments else None)
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        return func

class LeadFunction(WindowFunction):
    """LEAD window function."""
    
    def __init__(self, column: Optional[Any] = None, 
                 offset: int = 1, 
                 default: Optional[Any] = None):
        properties = OperatorProperties(
            name="LEAD",
            category=OperatorCategory.WINDOW,
            arity=-1,  # Variable arguments
            is_deterministic=True,
            output_type="ANY"
        )
        arguments = [column] if column is not None else []
        if offset != 1:
            arguments.append(offset)
        if default is not None:
            arguments.extend([offset, default] if len(arguments) == 1 else [default])
        
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        args = []
        if self.arguments:
            for arg in self.arguments:
                if hasattr(arg, 'to_sql'):
                    args.append(arg.to_sql(dialect))
                else:
                    args.append(str(arg))
        
        args_str = ', '.join(args) if args else ""
        return f"LEAD({args_str}) {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LEAD',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments],
            'partition_by': self.partition_by,
            'order_by': self.order_by
        }
    
    def clone(self) -> 'LeadFunction':
        import copy
        args = []
        for arg in self.arguments:
            if hasattr(arg, 'clone'):
                args.append(arg.clone())
            else:
                args.append(arg)
        
        func = LeadFunction(*args)
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        return func

class LagFunction(WindowFunction):
    """LAG window function."""
    
    def __init__(self, column: Optional[Any] = None, 
                 offset: int = 1, 
                 default: Optional[Any] = None):
        properties = OperatorProperties(
            name="LAG",
            category=OperatorCategory.WINDOW,
            arity=-1,  # Variable arguments
            is_deterministic=True,
            output_type="ANY"
        )
        arguments = [column] if column is not None else []
        if offset != 1:
            arguments.append(offset)
        if default is not None:
            arguments.extend([offset, default] if len(arguments) == 1 else [default])
        
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        args = []
        if self.arguments:
            for arg in self.arguments:
                if hasattr(arg, 'to_sql'):
                    args.append(arg.to_sql(dialect))
                else:
                    args.append(str(arg))
        
        args_str = ', '.join(args) if args else ""
        return f"LAG({args_str}) {self._window_clause_sql(dialect)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LAG',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments],
            'partition_by': self.partition_by,
            'order_by': self.order_by
        }
    
    def clone(self) -> 'LagFunction':
        import copy
        args = []
        for arg in self.arguments:
            if hasattr(arg, 'clone'):
                args.append(arg.clone())
            else:
                args.append(arg)
        
        func = LagFunction(*args)
        func.partition_by = copy.deepcopy(self.partition_by)
        func.order_by = copy.deepcopy(self.order_by)
        return func

# Register window functions
register_operator('ROW_NUMBER', RowNumberFunction)
register_operator('RANK', RankFunction)
register_operator('DENSE_RANK', DenseRankFunction)
register_operator('NTILE', NtileFunction)
register_operator('LEAD', LeadFunction)
register_operator('LAG', LagFunction)