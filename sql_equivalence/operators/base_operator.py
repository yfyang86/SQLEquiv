# sql_equivalence/operators/base_operator.py
"""Base operator classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

class OperatorCategory(Enum):
    """Categories of SQL operators."""
    RELATIONAL = "relational"      # SELECT, FROM, WHERE, etc.
    SET = "set"                    # UNION, INTERSECT, EXCEPT
    AGGREGATE = "aggregate"        # SUM, COUNT, AVG, etc.
    WINDOW = "window"              # ROW_NUMBER, RANK, etc.
    SCALAR = "scalar"              # String/Math functions
    LOGICAL = "logical"            # AND, OR, NOT
    COMPARISON = "comparison"      # =, !=, <, >, etc.
    ARITHMETIC = "arithmetic"      # +, -, *, /

@dataclass
class OperatorProperties:
    """Properties of an operator."""
    name: str
    category: OperatorCategory
    arity: int  # Number of operands (-1 for variable)
    is_commutative: bool = False
    is_associative: bool = False
    is_distributive: bool = False
    is_aggregate: bool = False
    is_deterministic: bool = True
    preserves_nulls: bool = True
    output_type: Optional[str] = None
    precedence: int = 0

class BaseOperator(ABC):
    """Abstract base class for all SQL operators."""
    
    def __init__(self, properties: OperatorProperties):
        self.properties = properties
        self.operands: List[Any] = []
        self.parameters: Dict[str, Any] = {}
        self._hash: Optional[int] = None
    
    @abstractmethod
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the operator configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def to_sql(self, dialect: str = 'standard') -> str:
        """
        Convert operator to SQL string.
        
        Args:
            dialect: SQL dialect to use
            
        Returns:
            SQL string representation
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert operator to dictionary representation."""
        pass
    
    @abstractmethod
    def clone(self) -> 'BaseOperator':
        """Create a deep copy of the operator."""
        pass
    
    def add_operand(self, operand: Any) -> None:
        """Add an operand to the operator."""
        self.operands.append(operand)
        self._hash = None  # Invalidate cached hash
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter for the operator."""
        self.parameters[name] = value
        self._hash = None  # Invalidate cached hash
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name, default)
    
    def is_equivalent_to(self, other: 'BaseOperator') -> bool:
        """Check if this operator is equivalent to another."""
        if type(self) != type(other):
            return False
        
        if self.properties.name != other.properties.name:
            return False
        
        # Check parameters
        if self.parameters != other.parameters:
            return False
        
        # For commutative operators, order doesn't matter
        if self.properties.is_commutative:
            return set(self._operand_hashes()) == set(other._operand_hashes())
        else:
            return self.operands == other.operands
    
    def _operand_hashes(self) -> List[int]:
        """Get hashes of all operands."""
        hashes = []
        for operand in self.operands:
            if isinstance(operand, BaseOperator):
                hashes.append(hash(operand))
            else:
                hashes.append(hash(str(operand)))
        return hashes
    
    def get_output_schema(self, input_schemas: List[Set[str]]) -> Set[str]:
        """
        Get the output schema based on input schemas.
        
        Args:
            input_schemas: List of input schemas (sets of column names)
            
        Returns:
            Output schema
        """
        # Default implementation - override in subclasses
        if input_schemas:
            return input_schemas[0]
        return set()
    
    def get_dependencies(self) -> Set[str]:
        """Get column dependencies of this operator."""
        deps = set()
        for operand in self.operands:
            if isinstance(operand, BaseOperator):
                deps.update(operand.get_dependencies())
            elif isinstance(operand, str):
                # Assume it's a column reference
                deps.add(operand)
        return deps
    
    def __hash__(self) -> int:
        """Compute hash for the operator."""
        if self._hash is None:
            # Create a stable hash based on operator type, parameters, and operands
            hash_data = {
                'type': type(self).__name__,
                'name': self.properties.name,
                'parameters': self.parameters,
                'operands': [
                    hash(op) if isinstance(op, BaseOperator) else str(op)
                    for op in self.operands
                ]
            }
            
            # Use JSON serialization for stable hashing
            hash_str = json.dumps(hash_data, sort_keys=True)
            self._hash = int(hashlib.md5(hash_str.encode()).hexdigest(), 16)
        
        return self._hash
    
    def __eq__(self, other: Any) -> bool:
        """Check equality with another operator."""
        if not isinstance(other, BaseOperator):
            return False
        return self.is_equivalent_to(other)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.properties.name})"

class UnaryOperator(BaseOperator):
    """Base class for unary operators."""
    
    def __init__(self, properties: OperatorProperties, operand: Any = None):
        super().__init__(properties)
        if operand is not None:
            self.add_operand(operand)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        if len(self.operands) != 1:
            return False, f"Unary operator {self.properties.name} requires exactly 1 operand"
        return True, None
    
    @property
    def operand(self) -> Any:
        """Get the single operand."""
        return self.operands[0] if self.operands else None

class BinaryOperator(BaseOperator):
    """Base class for binary operators."""
    
    def __init__(self, properties: OperatorProperties, 
                 left: Any = None, right: Any = None):
        super().__init__(properties)
        if left is not None:
            self.add_operand(left)
        if right is not None:
            self.add_operand(right)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        if len(self.operands) != 2:
            return False, f"Binary operator {self.properties.name} requires exactly 2 operands"
        return True, None
    
    @property
    def left(self) -> Any:
        """Get the left operand."""
        return self.operands[0] if len(self.operands) > 0 else None
    
    @property
    def right(self) -> Any:
        """Get the right operand."""
        return self.operands[1] if len(self.operands) > 1 else None

class FunctionOperator(BaseOperator):
    """Base class for function operators."""
    
    def __init__(self, properties: OperatorProperties, 
                 arguments: Optional[List[Any]] = None):
        super().__init__(properties)
        if arguments:
            for arg in arguments:
                self.add_operand(arg)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.properties.arity >= 0:
            if len(self.operands) != self.properties.arity:
                return False, (f"Function {self.properties.name} requires exactly "
                             f"{self.properties.arity} arguments")
        return True, None
    
    @property
    def arguments(self) -> List[Any]:
        """Get function arguments."""
        return self.operands

# Operator factory
_operator_registry: Dict[str, type] = {}

def register_operator(name: str, operator_class: type) -> None:
    """Register an operator class."""
    _operator_registry[name.upper()] = operator_class

def create_operator(name: str, **kwargs) -> BaseOperator:
    """Create an operator instance by name."""
    operator_class = _operator_registry.get(name.upper())
    if not operator_class:
        raise ValueError(f"Unknown operator: {name}")
    return operator_class(**kwargs)
