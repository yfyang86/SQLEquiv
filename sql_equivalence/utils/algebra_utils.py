# sql_equivalence/utils/algebra_utils.py
"""Algebraic operation utilities."""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PredicateType(Enum):
    """Types of predicates."""
    COMPARISON = "comparison"
    LOGICAL = "logical"
    IN = "in"
    BETWEEN = "between"
    LIKE = "like"
    EXISTS = "exists"
    IS_NULL = "is_null"

@dataclass
class Predicate:
    """Represents a logical predicate."""
    predicate_type: PredicateType
    operator: str
    operands: List[Any]
    negated: bool = False

def simplify_algebraic_expression(expression: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify an algebraic expression by applying algebraic laws.
    
    Args:
        expression: Dictionary representing algebraic expression
        
    Returns:
        Simplified expression
    """
    # This is a framework implementation
    # Real implementation would apply various simplification rules
    
    simplified = expression.copy()
    
    # Remove double negations
    if expression.get('type') == 'NOT' and expression.get('operand', {}).get('type') == 'NOT':
        simplified = expression['operand']['operand']
    
    # Simplify AND/OR with constants
    if expression.get('type') in ['AND', 'OR']:
        operands = expression.get('operands', [])
        
        # Remove TRUE from AND, FALSE from OR
        if expression['type'] == 'AND':
            operands = [op for op in operands if op != {'type': 'LITERAL', 'value': True}]
        else:  # OR
            operands = [op for op in operands if op != {'type': 'LITERAL', 'value': False}]
        
        # Short-circuit evaluation
        if expression['type'] == 'AND' and any(op == {'type': 'LITERAL', 'value': False} for op in operands):
            return {'type': 'LITERAL', 'value': False}
        if expression['type'] == 'OR' and any(op == {'type': 'LITERAL', 'value': True} for op in operands):
            return {'type': 'LITERAL', 'value': True}
        
        # Single operand
        if len(operands) == 1:
            return operands[0]
        
        simplified['operands'] = operands
    
    return simplified

def evaluate_predicate(predicate: Union[Dict[str, Any], Predicate], 
                      context: Dict[str, Any]) -> Optional[bool]:
    """
    Evaluate a predicate given a context.
    
    Args:
        predicate: Predicate to evaluate
        context: Dictionary mapping variables to values
        
    Returns:
        Boolean result or None if cannot evaluate
    """
    if isinstance(predicate, dict):
        pred_type = predicate.get('type')
        
        if pred_type == 'LITERAL':
            return predicate.get('value')
        
        elif pred_type == 'COLUMN':
            col_name = predicate.get('name')
            return context.get(col_name)
        
        elif pred_type == 'COMPARISON':
            left = evaluate_predicate(predicate.get('left'), context)
            right = evaluate_predicate(predicate.get('right'), context)
            op = predicate.get('operator')
            
            if left is None or right is None:
                return None
            
            if op == '=':
                return left == right
            elif op == '!=':
                return left != right
            elif op == '<':
                return left < right
            elif op == '>':
                return left > right
            elif op == '<=':
                return left <= right
            elif op == '>=':
                return left >= right
        
        elif pred_type == 'AND':
            results = []
            for operand in predicate.get('operands', []):
                result = evaluate_predicate(operand, context)
                if result is False:
                    return False
                if result is not None:
                    results.append(result)
            return all(results) if results else None
        
        elif pred_type == 'OR':
            results = []
            for operand in predicate.get('operands', []):
                result = evaluate_predicate(operand, context)
                if result is True:
                    return True
                if result is not None:
                    results.append(result)
            return any(results) if results else None
        
        elif pred_type == 'NOT':
            result = evaluate_predicate(predicate.get('operand'), context)
            return not result if result is not None else None
    
    return None

def merge_predicates(predicates: List[Dict[str, Any]], 
                    operator: str = 'AND') -> Dict[str, Any]:
    """
    Merge multiple predicates with a logical operator.
    
    Args:
        predicates: List of predicates
        operator: Logical operator ('AND' or 'OR')
        
    Returns:
        Merged predicate
    """
    if not predicates:
        return {'type': 'LITERAL', 'value': True if operator == 'AND' else False}
    
    if len(predicates) == 1:
        return predicates[0]
    
    # Flatten nested predicates of the same type
    flattened = []
    for pred in predicates:
        if pred.get('type') == operator:
            flattened.extend(pred.get('operands', []))
        else:
            flattened.append(pred)
    
    # Remove duplicates
    unique_preds = []
    seen = set()
    for pred in flattened:
        pred_str = str(pred)
        if pred_str not in seen:
            seen.add(pred_str)
            unique_preds.append(pred)
    
    return {
        'type': operator,
        'operands': unique_preds
    }

def split_conjunctive_predicate(predicate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a conjunctive predicate into individual predicates.
    
    Args:
        predicate: Predicate with AND operators
        
    Returns:
        List of individual predicates
    """
    if predicate.get('type') != 'AND':
        return [predicate]
    
    predicates = []
    for operand in predicate.get('operands', []):
        if operand.get('type') == 'AND':
            # Recursively split nested ANDs
            predicates.extend(split_conjunctive_predicate(operand))
        else:
            predicates.append(operand)
    
    return predicates

def is_predicate_satisfiable(predicate: Dict[str, Any]) -> bool:
    """
    Check if a predicate is satisfiable (not always false).
    
    Args:
        predicate: Predicate to check
        
    Returns:
        True if satisfiable
    """
    # Simplified implementation
    # Full implementation would use SAT solver or similar
    
    if predicate.get('type') == 'LITERAL':
        return predicate.get('value', False)
    
    # Check for contradictions like x = 1 AND x = 2
    if predicate.get('type') == 'AND':
        # Extract equality constraints
        equalities = {}
        for operand in predicate.get('operands', []):
            if operand.get('type') == 'COMPARISON' and operand.get('operator') == '=':
                left = operand.get('left', {})
                right = operand.get('right', {})
                
                if left.get('type') == 'COLUMN' and right.get('type') == 'LITERAL':
                    col_name = left.get('name')
                    value = right.get('value')
                    
                    if col_name in equalities and equalities[col_name] != value:
                        return False  # Contradiction
                    equalities[col_name] = value
    
    return True

def normalize_predicate(predicate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a predicate to canonical form.
    
    Args:
        predicate: Predicate to normalize
        
    Returns:
        Normalized predicate
    """
    normalized = predicate.copy()
    
    # Normalize comparison operators
    if normalized.get('type') == 'COMPARISON':
        op = normalized.get('operator')
        
        # Convert != to NOT =
        if op == '!=':
            normalized = {
                'type': 'NOT',
                'operand': {
                    'type': 'COMPARISON',
                    'operator': '=',
                    'left': normalized.get('left'),
                    'right': normalized.get('right')
                }
            }
        
        # Ensure consistent ordering (smaller value on left)
        elif op in ['<', '>', '<=', '>=']:
            left = normalized.get('left', {})
            right = normalized.get('right', {})
            
            # If left is literal and right is column, swap
            if left.get('type') == 'LITERAL' and right.get('type') == 'COLUMN':
                normalized['left'] = right
                normalized['right'] = left
                
                # Reverse operator
                op_map = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}
                normalized['operator'] = op_map[op]
    
    # Sort operands of commutative operators
    elif normalized.get('type') in ['AND', 'OR']:
        operands = normalized.get('operands', [])
        # Sort by string representation for consistency
        normalized['operands'] = sorted(operands, key=str)
    
    return normalized

def get_predicate_tables(predicate: Dict[str, Any]) -> Set[str]:
    """
    Extract table names referenced in a predicate.
    
    Args:
        predicate: Predicate expression
        
    Returns:
        Set of table names
    """
    tables = set()
    
    def extract_tables(expr):
        if isinstance(expr, dict):
            if expr.get('type') == 'COLUMN':
                table = expr.get('table')
                if table:
                    tables.add(table)
            
            # Recursively process nested expressions
            for key, value in expr.items():
                if key in ['left', 'right', 'operand', 'operands']:
                    if isinstance(value, list):
                        for item in value:
                            extract_tables(item)
                    else:
                        extract_tables(value)
    
    extract_tables(predicate)
    return tables

def get_predicate_columns(predicate: Dict[str, Any]) -> Set[str]:
    """
    Extract column names referenced in a predicate.
    
    Args:
        predicate: Predicate expression
        
    Returns:
        Set of column names
    """
    columns = set()
    
    def extract_columns(expr):
        if isinstance(expr, dict):
            if expr.get('type') == 'COLUMN':
                col_name = expr.get('name')
                if col_name:
                    columns.add(col_name)
            
            # Recursively process nested expressions
            for key, value in expr.items():
                if key in ['left', 'right', 'operand', 'operands']:
                    if isinstance(value, list):
                        for item in value:
                            extract_columns(item)
                    else:
                        extract_columns(value)
    
    extract_columns(predicate)
    return columns

def convert_predicate_to_cnf(predicate: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Convert predicate to Conjunctive Normal Form (CNF).
    
    Args:
        predicate: Predicate expression
        
    Returns:
        List of clauses (each clause is a list of literals)
    """
    # Simplified implementation
    # Full implementation would handle all logical operators
    
    if predicate.get('type') == 'AND':
        # Already in CNF if all operands are disjunctions or literals
        clauses = []
        for operand in predicate.get('operands', []):
            if operand.get('type') == 'OR':
                clauses.append(operand.get('operands', []))
            else:
                clauses.append([operand])
        return clauses
    
    elif predicate.get('type') == 'OR':
        # Single clause
        return [predicate.get('operands', [])]
    
    else:
        # Single literal
        return [[predicate]]

def predicate_implies(p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
    """
    Check if predicate p1 implies predicate p2.
    
    Args:
        p1: First predicate
        p2: Second predicate
        
    Returns:
        True if p1 implies p2
    """
    # Simplified implementation
    # Full implementation would use theorem proving techniques
    
    # If p2 is always true, then p1 implies p2
    if p2.get('type') == 'LITERAL' and p2.get('value') is True:
        return True
    
    # If p1 is always false, then p1 implies anything
    if p1.get('type') == 'LITERAL' and p1.get('value') is False:
        return True
    
    # If p1 and p2 are identical
    if p1 == p2:
        return True
    
    # Check simple cases
    if p1.get('type') == 'AND' and p2 in p1.get('operands', []):
        return True  # p1 is stronger than p2
    
    return False