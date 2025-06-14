# sql_equivalence/representations/algebraic/operators.py (fixed GroupByOperator and others)
"""Relational algebra operators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
import json
import hashlib

class OperatorType(Enum):
    """Types of algebraic operators."""
    PROJECT = "π"  # Projection
    SELECT = "σ"   # Selection
    JOIN = "⋈"     # Join
    CROSS_PRODUCT = "×"  # Cross product
    UNION = "∪"    # Union
    INTERSECT = "∩"  # Intersection
    DIFFERENCE = "-"  # Difference (EXCEPT)
    RENAME = "ρ"   # Rename
    AGGREGATE = "γ"  # Aggregation
    GROUP_BY = "G"  # Group by
    ORDER_BY = "τ"  # Order by (tau)
    DISTINCT = "δ"  # Distinct
    RELATION = "R"  # Base relation (table)
 
@dataclass
class AlgebraicOperator(ABC):
    """Abstract base class for algebraic operators."""
    
    operator_type: OperatorType
    children: List['AlgebraicOperator'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Create a hash for the operator."""
        # Create a stable string representation for hashing
        hash_parts = [
            self.operator_type.value,
            str(sorted(self.attributes.items())),
            str([hash(child) for child in self.children])
        ]
        hash_string = "|".join(hash_parts)
        return hash(hash_string)
    
    def __eq__(self, other):
        """Check equality with another operator."""
        if not isinstance(other, AlgebraicOperator):
            return False
        
        if self.operator_type != other.operator_type:
            return False
        
        if self.attributes != other.attributes:
            return False
        
        if len(self.children) != len(other.children):
            return False
        
        for c1, c2 in zip(self.children, other.children):
            if c1 != c2:
                return False
        
        return True
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert operator to string notation."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert operator to dictionary representation."""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Set[str]:
        """Get the output schema (set of attributes) of this operator."""
        pass
    
    @abstractmethod
    def clone(self) -> 'AlgebraicOperator':
        """Create a deep copy of this operator."""
        pass
    
    def add_child(self, child: 'AlgebraicOperator') -> None:
        """Add a child operator."""
        self.children.append(child)
    
    def get_child(self, index: int = 0) -> Optional['AlgebraicOperator']:
        """Get child operator by index."""
        return self.children[index] if index < len(self.children) else None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.operator_type.value})"

@dataclass
class SelectOperator(AlgebraicOperator):
    """Selection operator σ."""
    
    def __init__(self, condition: Union[str, Dict[str, Any]]):
        """
        Initialize selection operator.
        
        Args:
            condition: Selection condition (WHERE clause)
        """
        super().__init__(operator_type=OperatorType.SELECT)
        self.condition = condition
        self.attributes['condition'] = condition
    
    def __hash__(self):
        """Create a hash for the selection operator."""
        # Convert condition to a stable string representation
        if isinstance(self.condition, dict):
            condition_str = json.dumps(self.condition, sort_keys=True)
        else:
            condition_str = str(self.condition)
        
        hash_parts = [
            self.operator_type.value,
            condition_str,
            str([hash(child) for child in self.children])
        ]
        hash_string = "|".join(hash_parts)
        return hash(hash_string)
    
    def to_string(self) -> str:
        condition_str = self.condition if isinstance(self.condition, str) else str(self.condition)
        child_str = self.children[0].to_string() if self.children else "?"
        return f"σ[{condition_str}]({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'condition': self.condition,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Selection doesn't change schema
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'SelectOperator':
        op = SelectOperator(condition=copy.deepcopy(self.condition))
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class ProjectOperator(AlgebraicOperator):
    """Projection operator π."""
    
    def __init__(self, columns: List[Union[str, Dict[str, Any]]]):
        """
        Initialize projection operator.
        
        Args:
            columns: List of columns to project
        """
        super().__init__(operator_type=OperatorType.PROJECT)
        self.columns = columns
        self.attributes['columns'] = columns
    
    def to_string(self) -> str:
        cols_str = ', '.join(str(c) for c in self.columns)
        child_str = self.children[0].to_string() if self.children else "?"
        return f"π[{cols_str}]({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'columns': self.columns,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Return projected columns
        schema = set()
        for col in self.columns:
            if isinstance(col, str):
                schema.add(col)
            elif isinstance(col, dict) and 'alias' in col:
                schema.add(col['alias'])
        return schema
    
    def clone(self) -> 'ProjectOperator':
        op = ProjectOperator(columns=copy.deepcopy(self.columns))
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class JoinOperator(AlgebraicOperator):
    """Join operator ⋈."""
    
    def __init__(self, join_type: str = 'INNER', condition: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize join operator.
        
        Args:
            join_type: Type of join (INNER, LEFT, RIGHT, FULL)
            condition: Join condition
        """
        super().__init__(operator_type=OperatorType.JOIN)
        self.join_type = join_type
        self.condition = condition
        self.attributes['join_type'] = join_type
        self.attributes['condition'] = condition
    
    def set_condition(self, condition: Union[str, Dict[str, Any]]) -> None:
        """Set join condition."""
        self.condition = condition
        self.attributes['condition'] = condition
    
    def to_string(self) -> str:
        left_str = self.children[0].to_string() if len(self.children) > 0 else "?"
        right_str = self.children[1].to_string() if len(self.children) > 1 else "?"
        cond_str = f"[{self.condition}]" if self.condition else ""
        return f"({left_str} ⋈{cond_str} {right_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'join_type': self.join_type,
            'condition': self.condition,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Union of schemas from both children
        schema = set()
        if len(self.children) > 0:
            schema.update(self.children[0].get_output_schema())
        if len(self.children) > 1:
            schema.update(self.children[1].get_output_schema())
        return schema
    
    def clone(self) -> 'JoinOperator':
        op = JoinOperator(join_type=self.join_type, condition=copy.deepcopy(self.condition))
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class RelationOperator(AlgebraicOperator):
    """Base relation (table) operator R."""
    
    def __init__(self, table_name: str, alias: Optional[str] = None):
        """
        Initialize relation operator.
        
        Args:
            table_name: Name of the table
            alias: Table alias
        """
        super().__init__(operator_type=OperatorType.RELATION)
        self.table_name = table_name
        self.alias = alias
        self.attributes['table_name'] = table_name
        self.attributes['alias'] = alias
    
    def to_string(self) -> str:
        if self.alias:
            return f"{self.table_name} AS {self.alias}"
        return self.table_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'table_name': self.table_name,
            'alias': self.alias
        }
    
    def get_output_schema(self) -> Set[str]:
        # Would need schema information to return actual columns
        return {'*'}  # Placeholder
    
    def clone(self) -> 'RelationOperator':
        return RelationOperator(table_name=self.table_name, alias=self.alias)

@dataclass
class GroupByOperator(AlgebraicOperator):
    """Group by operator G."""
    
    def __init__(self, group_by_columns: Optional[List[str]] = None):
        """
        Initialize group by operator.
        
        Args:
            group_by_columns: Columns to group by
        """
        super().__init__(operator_type=OperatorType.GROUP_BY)
        self.group_by_columns = group_by_columns or []
        self.attributes['group_by_columns'] = self.group_by_columns
    
    def to_string(self) -> str:
        cols_str = ', '.join(self.group_by_columns)
        child_str = self.children[0].to_string() if self.children else "?"
        return f"G[{cols_str}]({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'group_by_columns': self.group_by_columns,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Group by preserves grouping columns
        return set(self.group_by_columns)
    
    def clone(self) -> 'GroupByOperator':
        op = GroupByOperator(group_by_columns=copy.deepcopy(self.group_by_columns))
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class AggregateOperator(AlgebraicOperator):
    """Aggregate operator γ."""
    
    def __init__(self, aggregations: List[Dict[str, Any]], group_by: Optional[List[str]] = None):
        """
        Initialize aggregate operator.
        
        Args:
            aggregations: List of aggregation functions
            group_by: Group by columns
        """
        super().__init__(operator_type=OperatorType.AGGREGATE)
        self.aggregations = aggregations
        self.group_by = group_by or []
        self.attributes['aggregations'] = aggregations
        self.attributes['group_by'] = self.group_by
    
    def to_string(self) -> str:
        agg_str = ', '.join(f"{a['function']}({a.get('arguments', ['*'])[0]})" for a in self.aggregations)
        if self.group_by:
            agg_str += f"; GROUP BY {', '.join(self.group_by)}"
        child_str = self.children[0].to_string() if self.children else "?"
        return f"γ[{agg_str}]({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'aggregations': self.aggregations,
            'group_by': self.group_by,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        schema = set(self.group_by)
        for agg in self.aggregations:
            if 'alias' in agg:
                schema.add(agg['alias'])
        return schema
    
    def clone(self) -> 'AggregateOperator':
        op = AggregateOperator(
            aggregations=copy.deepcopy(self.aggregations),
            group_by=copy.deepcopy(self.group_by)
        )
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class OrderByOperator(AlgebraicOperator):
    """Order by operator τ."""
    
    def __init__(self, order_by_columns: Optional[List[Union[str, Dict[str, Any]]]] = None):
        """
        Initialize order by operator.
        
        Args:
            order_by_columns: Columns to order by
        """
        super().__init__(operator_type=OperatorType.ORDER_BY)
        self.order_by_columns = order_by_columns or []
        self.attributes['order_by_columns'] = self.order_by_columns
    
    def to_string(self) -> str:
        cols_str = ', '.join(
            f"{col['column']} {col.get('direction', 'ASC')}" if isinstance(col, dict) else str(col)
            for col in self.order_by_columns
        )
        child_str = self.children[0].to_string() if self.children else "?"
        return f"τ[{cols_str}]({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'order_by_columns': self.order_by_columns,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Order by doesn't change schema
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'OrderByOperator':
        op = OrderByOperator(order_by_columns=copy.deepcopy(self.order_by_columns))
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class UnionOperator(AlgebraicOperator):
    """Union operator ∪."""
    
    def __init__(self, distinct: bool = True):
        """
        Initialize union operator.
        
        Args:
            distinct: Whether to remove duplicates (UNION vs UNION ALL)
        """
        super().__init__(operator_type=OperatorType.UNION)
        self.distinct = distinct
        self.attributes['distinct'] = distinct
        self.attributes['is_commutative'] = True
    
    def to_string(self) -> str:
        left_str = self.children[0].to_string() if len(self.children) > 0 else "?"
        right_str = self.children[1].to_string() if len(self.children) > 1 else "?"
        op = "∪" if self.distinct else "∪ ALL"
        return f"({left_str} {op} {right_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'distinct': self.distinct,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # Should be same schema for both children
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'UnionOperator':
        op = UnionOperator(distinct=self.distinct)
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class IntersectOperator(AlgebraicOperator):
    """Intersect operator ∩."""
    
    def __init__(self, distinct: bool = True):
        super().__init__(operator_type=OperatorType.INTERSECT)
        self.distinct = distinct
        self.attributes['distinct'] = distinct
    
    def to_string(self) -> str:
        left_str = self.children[0].to_string() if len(self.children) > 0 else "?"
        right_str = self.children[1].to_string() if len(self.children) > 1 else "?"
        return f"({left_str} ∩ {right_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'distinct': self.distinct,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'IntersectOperator':
        op = IntersectOperator(distinct=self.distinct)
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class ExceptOperator(AlgebraicOperator):
    """Except (difference) operator -."""
    
    def __init__(self, distinct: bool = True):
        super().__init__(operator_type=OperatorType.DIFFERENCE)
        self.distinct = distinct
        self.attributes['distinct'] = distinct
    
    def to_string(self) -> str:
        left_str = self.children[0].to_string() if len(self.children) > 0 else "?"
        right_str = self.children[1].to_string() if len(self.children) > 1 else "?"
        return f"({left_str} - {right_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'distinct': self.distinct,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'ExceptOperator':
        op = ExceptOperator(distinct=self.distinct)
        op.children = [child.clone() for child in self.children]
        return op

@dataclass
class DistinctOperator(AlgebraicOperator):
    """DISTINCT operator δ."""
    
    def __init__(self):
        """Initialize DISTINCT operator."""
        super().__init__(operator_type=OperatorType.DISTINCT)
    
    def to_string(self) -> str:
        child_str = self.children[0].to_string() if self.children else "?"
        return f"δ({child_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.operator_type.value,
            'children': [child.to_dict() for child in self.children]
        }
    
    def get_output_schema(self) -> Set[str]:
        # DISTINCT doesn't change schema
        return self.children[0].get_output_schema() if self.children else set()
    
    def clone(self) -> 'DistinctOperator':
        op = DistinctOperator()
        op.children = [child.clone() for child in self.children]
        return op