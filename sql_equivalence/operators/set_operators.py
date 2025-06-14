# sql_equivalence/operators/set_operators.py
"""Set operators for SQL queries."""

from typing import Any, Dict, List, Optional

from .base_operator import (
    BinaryOperator, OperatorCategory, OperatorProperties,
    register_operator
)

class SetOperator(BinaryOperator):
    """Base class for set operators."""
    
    def __init__(self, properties: OperatorProperties,
                 left: Optional[Any] = None,
                 right: Optional[Any] = None,
                 distinct: bool = True):
        super().__init__(properties, left, right)
        self.distinct = distinct
        self.parameters['distinct'] = distinct
    
    def set_distinct(self, distinct: bool) -> None:
        """Set whether to use DISTINCT."""
        self.distinct = distinct
        self.parameters['distinct'] = distinct
        self._hash = None

class UnionOperator(SetOperator):
    """UNION operator."""
    
    def __init__(self, left: Optional[Any] = None,
                 right: Optional[Any] = None,
                 distinct: bool = True):
        properties = OperatorProperties(
            name="UNION",
            category=OperatorCategory.SET,
            arity=2,
            is_commutative=True,
            is_associative=True,
            is_deterministic=True
        )
        super().__init__(properties, left, right, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        left_sql = self.left.to_sql(dialect) if hasattr(self.left, 'to_sql') else str(self.left)
        right_sql = self.right.to_sql(dialect) if hasattr(self.right, 'to_sql') else str(self.right)
        
        union_type = "UNION" if self.distinct else "UNION ALL"
        return f"({left_sql}) {union_type} ({right_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'UNION',
            'distinct': self.distinct,
            'left': self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
            'right': self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right
        }
    
    def clone(self) -> 'UnionOperator':
        left_clone = self.left.clone() if hasattr(self.left, 'clone') else self.left
        right_clone = self.right.clone() if hasattr(self.right, 'clone') else self.right
        return UnionOperator(left_clone, right_clone, self.distinct)

class IntersectOperator(SetOperator):
    """INTERSECT operator."""
    
    def __init__(self, left: Optional[Any] = None,
                 right: Optional[Any] = None,
                 distinct: bool = True):
        properties = OperatorProperties(
            name="INTERSECT",
            category=OperatorCategory.SET,
            arity=2,
            is_commutative=True,
            is_associative=True,
            is_deterministic=True
        )
        super().__init__(properties, left, right, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        left_sql = self.left.to_sql(dialect) if hasattr(self.left, 'to_sql') else str(self.left)
        right_sql = self.right.to_sql(dialect) if hasattr(self.right, 'to_sql') else str(self.right)
        
        intersect_type = "INTERSECT" if self.distinct else "INTERSECT ALL"
        return f"({left_sql}) {intersect_type} ({right_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'INTERSECT',
            'distinct': self.distinct,
            'left': self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
            'right': self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right
        }
    
    def clone(self) -> 'IntersectOperator':
        left_clone = self.left.clone() if hasattr(self.left, 'clone') else self.left
        right_clone = self.right.clone() if hasattr(self.right, 'clone') else self.right
        return IntersectOperator(left_clone, right_clone, self.distinct)

class ExceptOperator(SetOperator):
    """EXCEPT (MINUS) operator."""
    
    def __init__(self, left: Optional[Any] = None,
                 right: Optional[Any] = None,
                 distinct: bool = True):
        properties = OperatorProperties(
            name="EXCEPT",
            category=OperatorCategory.SET,
            arity=2,
            is_commutative=False,  # EXCEPT is not commutative
            is_associative=False,  # EXCEPT is not associative
            is_deterministic=True
        )
        super().__init__(properties, left, right, distinct)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        left_sql = self.left.to_sql(dialect) if hasattr(self.left, 'to_sql') else str(self.left)
        right_sql = self.right.to_sql(dialect) if hasattr(self.right, 'to_sql') else str(self.right)
        
        # Some dialects use MINUS instead of EXCEPT
        except_keyword = "MINUS" if dialect in ['oracle', 'db2'] else "EXCEPT"
        except_type = except_keyword if self.distinct else f"{except_keyword} ALL"
        
        return f"({left_sql}) {except_type} ({right_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'EXCEPT',
            'distinct': self.distinct,
            'left': self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
            'right': self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right
        }
    
    def clone(self) -> 'ExceptOperator':
        left_clone = self.left.clone() if hasattr(self.left, 'clone') else self.left
        right_clone = self.right.clone() if hasattr(self.right, 'clone') else self.right
        return ExceptOperator(left_clone, right_clone, self.distinct)

# Register operators
register_operator('UNION', UnionOperator)
register_operator('INTERSECT', IntersectOperator)
register_operator('EXCEPT', ExceptOperator)
register_operator('MINUS', ExceptOperator)  # Alias for EXCEPT