# sql_equivalence/operators/scalar_functions.py
"""Scalar function operators."""

from typing import Any, Dict, List, Optional
import math

from .base_operator import (
    FunctionOperator, UnaryOperator, OperatorCategory, OperatorProperties,
    register_operator
)

class ScalarFunction(FunctionOperator):
    """Base class for scalar functions."""
    
    def __init__(self, properties: OperatorProperties,
                 arguments: Optional[List[Any]] = None):
        super().__init__(properties, arguments)

# String Functions

class UpperFunction(UnaryOperator):
    """UPPER string function."""
    
    def __init__(self, string_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="UPPER",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="STRING"
        )
        super().__init__(properties, string_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            return f"UPPER({arg_sql})"
        return "UPPER()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'UPPER',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'UpperFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return UpperFunction(arg_clone)

class LowerFunction(UnaryOperator):
    """LOWER string function."""
    
    def __init__(self, string_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="LOWER",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="STRING"
        )
        super().__init__(properties, string_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            return f"LOWER({arg_sql})"
        return "LOWER()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LOWER',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'LowerFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return LowerFunction(arg_clone)

class TrimFunction(ScalarFunction):
    """TRIM string function."""
    
    def __init__(self, string_expr: Optional[Any] = None, 
                 trim_chars: Optional[str] = None,
                 trim_type: str = 'BOTH'):
        properties = OperatorProperties(
            name="TRIM",
            category=OperatorCategory.SCALAR,
            arity=-1,
            is_deterministic=True,
            output_type="STRING"
        )
        arguments = [string_expr] if string_expr is not None else []
        super().__init__(properties, arguments)
        self.trim_chars = trim_chars
        self.trim_type = trim_type.upper()  # LEADING, TRAILING, BOTH
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "TRIM()"
        
        arg_sql = self.arguments[0].to_sql(dialect) if hasattr(self.arguments[0], 'to_sql') else str(self.arguments[0])
        
        if self.trim_chars:
            if self.trim_type != 'BOTH':
                return f"TRIM({self.trim_type} '{self.trim_chars}' FROM {arg_sql})"
            return f"TRIM('{self.trim_chars}' FROM {arg_sql})"
        
        return f"TRIM({arg_sql})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'TRIM',
            'argument': self.arguments[0].to_dict() if self.arguments and hasattr(self.arguments[0], 'to_dict') else self.arguments[0] if self.arguments else None,
            'trim_chars': self.trim_chars,
            'trim_type': self.trim_type
        }
    
    def clone(self) -> 'TrimFunction':
        arg_clone = None
        if self.arguments:
            arg_clone = self.arguments[0].clone() if hasattr(self.arguments[0], 'clone') else self.arguments[0]
        return TrimFunction(arg_clone, self.trim_chars, self.trim_type)

class SubstringFunction(ScalarFunction):
    """SUBSTRING string function."""
    
    def __init__(self, string_expr: Optional[Any] = None,
                 start: Optional[int] = None,
                 length: Optional[int] = None):
        properties = OperatorProperties(
            name="SUBSTRING",
            category=OperatorCategory.SCALAR,
            arity=-1,
            is_deterministic=True,
            output_type="STRING"
        )
        arguments = []
        if string_expr is not None:
            arguments.append(string_expr)
        if start is not None:
            arguments.append(start)
        if length is not None:
            arguments.append(length)
        
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "SUBSTRING()"
        
        args_sql = []
        for arg in self.arguments:
            if hasattr(arg, 'to_sql'):
                args_sql.append(arg.to_sql(dialect))
            else:
                args_sql.append(str(arg))
        
        # Some dialects use SUBSTR
        func_name = "SUBSTR" if dialect in ['oracle', 'sqlite'] else "SUBSTRING"
        return f"{func_name}({', '.join(args_sql)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'SUBSTRING',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'SubstringFunction':
        args_clone = []
        for arg in self.arguments:
            if hasattr(arg, 'clone'):
                args_clone.append(arg.clone())
            else:
                args_clone.append(arg)
        return SubstringFunction(*args_clone)

class LengthFunction(UnaryOperator):
    """LENGTH string function."""
    
    def __init__(self, string_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="LENGTH",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, string_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            # Some dialects use LEN
            func_name = "LEN" if dialect in ['sqlserver', 'sybase'] else "LENGTH"
            return f"{func_name}({arg_sql})"
        return "LENGTH()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LENGTH',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'LengthFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return LengthFunction(arg_clone)

# Math Functions

class ExpFunction(UnaryOperator):
    """EXP (exponential) function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="EXP",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        super().__init__(properties, numeric_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            return f"EXP({arg_sql})"
        return "EXP()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'EXP',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'ExpFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return ExpFunction(arg_clone)

class LogFunction(ScalarFunction):
    """LOG (logarithm) function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None,
                 base: Optional[Any] = None):
        properties = OperatorProperties(
            name="LOG",
            category=OperatorCategory.SCALAR,
            arity=-1,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        arguments = []
        if numeric_expr is not None:
            arguments.append(numeric_expr)
        if base is not None:
            arguments.append(base)
        
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "LOG()"
        
        args_sql = []
        for arg in self.arguments:
            if hasattr(arg, 'to_sql'):
                args_sql.append(arg.to_sql(dialect))
            else:
                args_sql.append(str(arg))
        
        # Natural log is LN in some dialects
        if len(self.arguments) == 1 and dialect in ['postgres', 'oracle']:
            return f"LN({args_sql[0]})"
        
        return f"LOG({', '.join(args_sql)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'LOG',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'LogFunction':
        args_clone = []
        for arg in self.arguments:
            if hasattr(arg, 'clone'):
                args_clone.append(arg.clone())
            else:
                args_clone.append(arg)
        return LogFunction(*args_clone)

class AbsFunction(UnaryOperator):
    """ABS (absolute value) function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="ABS",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        super().__init__(properties, numeric_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            return f"ABS({arg_sql})"
        return "ABS()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ABS',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'AbsFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return AbsFunction(arg_clone)

class RoundFunction(ScalarFunction):
    """ROUND function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None,
                 precision: Optional[int] = None):
        properties = OperatorProperties(
            name="ROUND",
            category=OperatorCategory.SCALAR,
            arity=-1,
            is_deterministic=True,
            output_type="NUMERIC"
        )
        arguments = []
        if numeric_expr is not None:
            arguments.append(numeric_expr)
        if precision is not None:
            arguments.append(precision)
        
        super().__init__(properties, arguments)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if not self.arguments:
            return "ROUND()"
        
        args_sql = []
        for arg in self.arguments:
            if hasattr(arg, 'to_sql'):
                args_sql.append(arg.to_sql(dialect))
            else:
                args_sql.append(str(arg))
        
        return f"ROUND({', '.join(args_sql)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ROUND',
            'arguments': [arg.to_dict() if hasattr(arg, 'to_dict') else arg for arg in self.arguments]
        }
    
    def clone(self) -> 'RoundFunction':
        args_clone = []
        for arg in self.arguments:
            if hasattr(arg, 'clone'):
                args_clone.append(arg.clone())
            else:
                args_clone.append(arg)
        return RoundFunction(*args_clone)

class CeilFunction(UnaryOperator):
    """CEIL/CEILING function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="CEIL",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, numeric_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            # Some dialects use CEILING
            func_name = "CEILING" if dialect in ['sqlserver', 'postgres'] else "CEIL"
            return f"{func_name}({arg_sql})"
        return "CEIL()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'CEIL',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'CeilFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return CeilFunction(arg_clone)

class FloorFunction(UnaryOperator):
    """FLOOR function."""
    
    def __init__(self, numeric_expr: Optional[Any] = None):
        properties = OperatorProperties(
            name="FLOOR",
            category=OperatorCategory.SCALAR,
            arity=1,
            is_deterministic=True,
            output_type="INTEGER"
        )
        super().__init__(properties, numeric_expr)
    
    def to_sql(self, dialect: str = 'standard') -> str:
        if self.operand:
            arg_sql = self.operand.to_sql(dialect) if hasattr(self.operand, 'to_sql') else str(self.operand)
            return f"FLOOR({arg_sql})"
        return "FLOOR()"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'FLOOR',
            'argument': self.operand.to_dict() if hasattr(self.operand, 'to_dict') else self.operand
        }
    
    def clone(self) -> 'FloorFunction':
        arg_clone = self.operand.clone() if hasattr(self.operand, 'clone') else self.operand
        return FloorFunction(arg_clone)

# Register scalar functions
# String functions
register_operator('UPPER', UpperFunction)
register_operator('LOWER', LowerFunction)
register_operator('TRIM', TrimFunction)
register_operator('SUBSTRING', SubstringFunction)
register_operator('SUBSTR', SubstringFunction)  # Alias
register_operator('LENGTH', LengthFunction)
register_operator('LEN', LengthFunction)  # Alias
register_operator('CHAR_LENGTH', LengthFunction)  # Alias

# Math functions
register_operator('EXP', ExpFunction)
register_operator('LOG', LogFunction)
register_operator('LN', LogFunction)  # Natural log alias
register_operator('ABS', AbsFunction)
register_operator('ROUND', RoundFunction)
register_operator('CEIL', CeilFunction)
register_operator('CEILING', CeilFunction)  # Alias
register_operator('FLOOR', FloorFunction)