# sql_equivalence/operators/relational_operators.py
"""Relational SQL operators."""

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

from .base_operator import (
    BaseOperator, OperatorCategory, OperatorProperties,
    UnaryOperator, BinaryOperator, register_operator
)

class SelectOperator(BaseOperator):
    """SELECT clause operator."""
    
    def __init__(self, columns: Optional[List[Union[str, Dict[str, str]]]] = None):
        properties = OperatorProperties(
            name="SELECT",
            category=OperatorCategory.RELATIONAL,
            arity=-1,  # Variable number of columns
            is_deterministic=True
        )
        super().__init__(properties)
        self.columns = columns or []
        self.distinct = False
        self.parameters['columns'] = self.columns
        self.parameters['distinct'] = self.distinct
    
    def add_column(self, column: Union[str, Dict[str, str]]) -> None:
        """Add a column to the selection."""
        self.columns.append(column)
        self.parameters['columns'] = self.columns
        self._hash = None
    
    def set_distinct(self, distinct: bool = True) -> None:
        """Set DISTINCT flag."""
        self.distinct = distinct
        self.parameters['distinct'] = distinct
        self._hash = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.columns:
            return False, "SELECT requires at least one column"
        return True, None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        distinct_str = "DISTINCT " if self.distinct else ""
        
        column_strs = []
        for col in self.columns:
            if isinstance(col, dict):
                if 'alias' in col:
                    column_strs.append(f"{col['expr']} AS {col['alias']}")
                else:
                    column_strs.append(col['expr'])
            else:
                column_strs.append(col)
        
        return f"SELECT {distinct_str}{', '.join(column_strs)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'SELECT',
            'columns': self.columns,
            'distinct': self.distinct,
            'operands': [op.to_dict() if isinstance(op, BaseOperator) else op 
                        for op in self.operands]
        }
    
    def clone(self) -> 'SelectOperator':
        op = SelectOperator(columns=self.columns.copy())
        op.distinct = self.distinct
        for operand in self.operands:
            if isinstance(operand, BaseOperator):
                op.add_operand(operand.clone())
            else:
                op.add_operand(operand)
        return op
    
    def get_output_schema(self, input_schemas: List[Set[str]]) -> Set[str]:
        """Get output schema based on selected columns."""
        output_schema = set()
        
        for col in self.columns:
            if isinstance(col, dict):
                # Handle aliased columns
                if 'alias' in col:
                    output_schema.add(col['alias'])
                else:
                    output_schema.add(col['expr'])
            elif col == '*':
                # Select all columns from input
                if input_schemas:
                    output_schema.update(input_schemas[0])
            else:
                output_schema.add(col)
        
        return output_schema

class FromOperator(BaseOperator):
    """FROM clause operator."""
    
    def __init__(self, tables: Optional[List[Union[str, Dict[str, str]]]] = None):
        properties = OperatorProperties(
            name="FROM",
            category=OperatorCategory.RELATIONAL,
            arity=-1,
            is_deterministic=True
        )
        super().__init__(properties)
        self.tables = tables or []
        self.parameters['tables'] = self.tables
    
    def add_table(self, table: Union[str, Dict[str, str]]) -> None:
        """Add a table to FROM clause."""
        self.tables.append(table)
        self.parameters['tables'] = self.tables
        self._hash = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.tables:
            return False, "FROM requires at least one table"
        return True, None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        table_strs = []
        for table in self.tables:
            if isinstance(table, dict):
                if 'alias' in table:
                    table_strs.append(f"{table['name']} AS {table['alias']}")
                else:
                    table_strs.append(table['name'])
            else:
                table_strs.append(table)
        
        return f"FROM {', '.join(table_strs)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'FROM',
            'tables': self.tables
        }
    
    def clone(self) -> 'FromOperator':
        return FromOperator(tables=self.tables.copy())

class WhereOperator(UnaryOperator):
    """WHERE clause operator."""
    
    def __init__(self, condition: Optional[Any] = None):
        properties = OperatorProperties(
            name="WHERE",
            category=OperatorCategory.RELATIONAL,
            arity=1,
            is_deterministic=True
        )
        super().__init__(properties, condition)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            if isinstance(self.operand, BaseOperator):
                condition_str = self.operand.to_sql(dialect)
            else:
                condition_str = str(self.operand)
            return f"WHERE {condition_str}"
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'WHERE',
            'condition': self.operand.to_dict() if isinstance(self.operand, BaseOperator) else self.operand
        }
    
    def clone(self) -> 'WhereOperator':
        if self.operand and isinstance(self.operand, BaseOperator):
            return WhereOperator(self.operand.clone())
        return WhereOperator(self.operand)

class JoinOperator(BinaryOperator):
    """JOIN operator."""
    
    def __init__(self, join_type: str = 'INNER', 
                 left_table: Optional[Any] = None,
                 right_table: Optional[Any] = None,
                 condition: Optional[Any] = None):
        properties = OperatorProperties(
            name=f"{join_type}_JOIN",
            category=OperatorCategory.RELATIONAL,
            arity=2,
            is_commutative=(join_type == 'INNER'),  # Only INNER JOIN is commutative
            is_associative=(join_type == 'INNER'),
            is_deterministic=True
        )
        super().__init__(properties, left_table, right_table)
        self.join_type = join_type.upper()
        self.condition = condition
        self.parameters['join_type'] = self.join_type
        self.parameters['condition'] = condition
    
    def set_condition(self, condition: Any) -> None:
        """Set join condition."""
        self.condition = condition
        self.parameters['condition'] = condition
        self._hash = None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        left_sql = self.left.to_sql(dialect) if isinstance(self.left, BaseOperator) else str(self.left)
        right_sql = self.right.to_sql(dialect) if isinstance(self.right, BaseOperator) else str(self.right)
        
        join_str = f"{self.join_type} JOIN"
        
        if self.condition:
            condition_sql = self.condition.to_sql(dialect) if isinstance(self.condition, BaseOperator) else str(self.condition)
            return f"{left_sql} {join_str} {right_sql} ON {condition_sql}"
        else:
            return f"{left_sql} {join_str} {right_sql}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'JOIN',
            'join_type': self.join_type,
            'left': self.left.to_dict() if isinstance(self.left, BaseOperator) else self.left,
            'right': self.right.to_dict() if isinstance(self.right, BaseOperator) else self.right,
            'condition': self.condition.to_dict() if isinstance(self.condition, BaseOperator) else self.condition
        }
    
    def clone(self) -> 'JoinOperator':
        left_clone = self.left.clone() if isinstance(self.left, BaseOperator) else self.left
        right_clone = self.right.clone() if isinstance(self.right, BaseOperator) else self.right
        condition_clone = self.condition.clone() if isinstance(self.condition, BaseOperator) else self.condition
        
        return JoinOperator(
            join_type=self.join_type,
            left_table=left_clone,
            right_table=right_clone,
            condition=condition_clone
        )
    
    def get_output_schema(self, input_schemas: List[Set[str]]) -> Set[str]:
        """Join combines schemas from both inputs."""
        if len(input_schemas) >= 2:
            return input_schemas[0].union(input_schemas[1])
        return set()

class GroupByOperator(BaseOperator):
    """GROUP BY operator."""
    
    def __init__(self, columns: Optional[List[str]] = None):
        properties = OperatorProperties(
            name="GROUP_BY",
            category=OperatorCategory.RELATIONAL,
            arity=-1,
            is_deterministic=True
        )
        super().__init__(properties)
        self.columns = columns or []
        self.parameters['columns'] = self.columns
    
    def add_column(self, column: str) -> None:
        """Add a column to group by."""
        self.columns.append(column)
        self.parameters['columns'] = self.columns
        self._hash = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.columns:
            return False, "GROUP BY requires at least one column"
        return True, None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        return f"GROUP BY {', '.join(self.columns)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'GROUP_BY',
            'columns': self.columns
        }
    
    def clone(self) -> 'GroupByOperator':
        return GroupByOperator(columns=self.columns.copy())

class HavingOperator(UnaryOperator):
    """HAVING clause operator."""
    
    def __init__(self, condition: Optional[Any] = None):
        properties = OperatorProperties(
            name="HAVING",
            category=OperatorCategory.RELATIONAL,
            arity=1,
            is_deterministic=True
        )
        super().__init__(properties, condition)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            if isinstance(self.operand, BaseOperator):
                condition_str = self.operand.to_sql(dialect)
            else:
                condition_str = str(self.operand)
            return f"HAVING {condition_str}"
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'HAVING',
            'condition': self.operand.to_dict() if isinstance(self.operand, BaseOperator) else self.operand
        }
    
    def clone(self) -> 'HavingOperator':
        if self.operand and isinstance(self.operand, BaseOperator):
            return HavingOperator(self.operand.clone())
        return HavingOperator(self.operand)

class OrderByOperator(BaseOperator):
    """ORDER BY operator."""
    
    def __init__(self, columns: Optional[List[Dict[str, str]]] = None):
        properties = OperatorProperties(
            name="ORDER_BY",
            category=OperatorCategory.RELATIONAL,
            arity=-1,
            is_deterministic=True
        )
        super().__init__(properties)
        self.columns = columns or []
        self.parameters['columns'] = self.columns
    
    def add_column(self, column: str, direction: str = 'ASC') -> None:
        """Add a column to order by."""
        self.columns.append({'column': column, 'direction': direction})
        self.parameters['columns'] = self.columns
        self._hash = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if not self.columns:
            return False, "ORDER BY requires at least one column"
        return True, None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        order_strs = []
        for col in self.columns:
            order_str = col['column']
            if col.get('direction', 'ASC').upper() == 'DESC':
                order_str += " DESC"
            order_strs.append(order_str)
        
        return f"ORDER BY {', '.join(order_strs)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ORDER_BY',
            'columns': self.columns
        }
    
    def clone(self) -> 'OrderByOperator':
        import copy
        return OrderByOperator(columns=copy.deepcopy(self.columns))

class LimitOperator(BaseOperator):
    """LIMIT operator."""
    
    def __init__(self, limit: Optional[int] = None, offset: Optional[int] = None):
        properties = OperatorProperties(
            name="LIMIT",
            category=OperatorCategory.RELATIONAL,
            arity=0,
            is_deterministic=True
        )
        super().__init__(properties)
        self.limit = limit
        self.offset = offset
        self.parameters['limit'] = limit
        self.parameters['offset'] = offset
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if self.limit is None:
            return False, "LIMIT requires a limit value"
        if self.limit < 0:
            return False, "LIMIT value must be non-negative"
        if self.offset is not None and self.offset < 0:
            return False, "OFFSET value must be non-negative"
        return True, None
    
    def to_sql(self, dialect: str = 'standard') -> str:
        sql = f"LIMIT {self.limit}"
        if self.offset is not None:
            sql += f" OFFSET {self.offset}"
        return sql
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LIMIT',
            'limit': self.limit,
            'offset': self.offset
        }
    
    def clone(self) -> 'LimitOperator':
        return LimitOperator(limit=self.limit, offset=self.offset)

# Register operators
register_operator('SELECT', SelectOperator)
register_operator('FROM', FromOperator)
register_operator('WHERE', WhereOperator)
register_operator('JOIN', JoinOperator)
register_operator('INNER_JOIN', JoinOperator)
register_operator('LEFT_JOIN', JoinOperator)
register_operator('RIGHT_JOIN', JoinOperator)
register_operator('FULL_JOIN', JoinOperator)
register_operator('GROUP_BY', GroupByOperator)
register_operator('HAVING', HavingOperator)
register_operator('ORDER_BY', OrderByOperator)
register_operator('LIMIT', LimitOperator)