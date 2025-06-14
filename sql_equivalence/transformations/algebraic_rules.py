# sql_equivalence/transformations/algebraic_rules.py
"""Algebraic transformation rules for SQL queries."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import copy
import logging

from ..representations.algebraic.operators import (
    AlgebraicOperator, ProjectOperator, SelectOperator, JoinOperator,
    UnionOperator, IntersectOperator, ExceptOperator, AggregateOperator,
    RelationOperator, GroupByOperator, OrderByOperator, OperatorType
)
from ..representations.algebraic.relational_algebra import AlgebraicExpression

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types of algebraic transformation rules."""
    COMMUTATIVITY = "commutativity"
    ASSOCIATIVITY = "associativity"
    DISTRIBUTIVITY = "distributivity"
    IDEMPOTENCE = "idempotence"
    SIMPLIFICATION = "simplification"
    PUSHDOWN = "pushdown"
    PULLUP = "pullup"
    ELIMINATION = "elimination"
    MERGING = "merging"

@dataclass
class RuleApplication:
    """Result of applying a transformation rule."""
    success: bool
    transformed_operator: Optional[AlgebraicOperator]
    description: str
    confidence: float = 1.0

class AlgebraicRule(ABC):
    """Abstract base class for algebraic transformation rules."""
    
    def __init__(self, name: str, rule_type: RuleType, description: str):
        self.name = name
        self.rule_type = rule_type
        self.description = description
        self.application_count = 0
    
    @abstractmethod
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        """Check if the rule can be applied to the given operator."""
        pass
    
    @abstractmethod
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        """Apply the transformation rule to the operator."""
        pass
    
    def apply_to_expression(self, expression: AlgebraicExpression) -> bool:
        """Apply the rule to an entire expression tree."""
        if not expression.root_operator:
            return False
        
        result = self._apply_recursive(expression.root_operator)
        if result.success:
            expression.root_operator = result.transformed_operator
            self.application_count += 1
            return True
        
        return False
    
    def _apply_recursive(self, operator: AlgebraicOperator) -> RuleApplication:
        """Recursively apply the rule to the operator tree."""
        # First try to apply to current operator
        if self.is_applicable(operator):
            result = self.apply(operator)
            if result.success:
                return result
        
        # Then try to apply to children
        modified = False
        new_operator = operator.clone()
        
        for i, child in enumerate(operator.children):
            child_result = self._apply_recursive(child)
            if child_result.success:
                new_operator.children[i] = child_result.transformed_operator
                modified = True
                break  # Apply one transformation at a time
        
        if modified:
            return RuleApplication(
                success=True,
                transformed_operator=new_operator,
                description=f"Applied {self.name} to child operator"
            )
        
        return RuleApplication(success=False, transformed_operator=None, description="Not applicable")

class SelectionPushdown(AlgebraicRule):
    """Push selection operators down the tree (σ-pushdown)."""
    
    def __init__(self):
        super().__init__(
            name="Selection Pushdown",
            rule_type=RuleType.PUSHDOWN,
            description="Push selection below join/projection when possible"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, SelectOperator):
            return False
        
        if not operator.children:
            return False
        
        child = operator.children[0]
        return isinstance(child, (JoinOperator, ProjectOperator))
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None, 
                                 description="Rule not applicable")
        
        select_op = operator
        child = select_op.children[0]
        
        if isinstance(child, ProjectOperator):
            # Push selection below projection if it doesn't reference projected-out columns
            referenced_cols = self._extract_columns_from_condition(select_op.condition)
            projected_cols = set()
            for col in child.columns:
                if isinstance(col, dict):
                    projected_cols.add(col.get('expr', col.get('alias', '')))
                else:
                    projected_cols.add(col)
            
            # Check if all referenced columns are in projected columns
            if referenced_cols.issubset(projected_cols) or '*' in projected_cols:
                # Create new tree: π(σ(child))
                new_select = select_op.clone()
                new_select.children = child.children
                
                new_project = child.clone()
                new_project.children = [new_select]
                
                return RuleApplication(
                    success=True,
                    transformed_operator=new_project,
                    description="Pushed selection below projection"
                )
        
        elif isinstance(child, JoinOperator):
            # Try to push selection to appropriate side of join
            left_tables = self._get_tables_from_operator(child.children[0])
            right_tables = self._get_tables_from_operator(child.children[1])
            
            condition_tables = self._extract_tables_from_condition(select_op.condition)
            
            # If condition only references left tables
            if condition_tables.issubset(left_tables):
                new_join = child.clone()
                new_select = select_op.clone()
                new_select.children = [child.children[0]]
                new_join.children[0] = new_select
                
                return RuleApplication(
                    success=True,
                    transformed_operator=new_join,
                    description="Pushed selection to left side of join"
                )
            
            # If condition only references right tables
            elif condition_tables.issubset(right_tables):
                new_join = child.clone()
                new_select = select_op.clone()
                new_select.children = [child.children[1]]
                new_join.children[1] = new_select
                
                return RuleApplication(
                    success=True,
                    transformed_operator=new_join,
                    description="Pushed selection to right side of join"
                )
        
        return RuleApplication(success=False, transformed_operator=None, 
                             description="Cannot push selection down")
    
    def _extract_columns_from_condition(self, condition: Any) -> Set[str]:
        """Extract column names from a condition."""
        columns = set()
        
        if isinstance(condition, dict):
            if condition.get('type') == 'column':
                columns.add(condition.get('name', ''))
            
            for key, value in condition.items():
                if isinstance(value, dict):
                    columns.update(self._extract_columns_from_condition(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            columns.update(self._extract_columns_from_condition(item))
        
        return columns
    
    def _extract_tables_from_condition(self, condition: Any) -> Set[str]:
        """Extract table names from a condition."""
        tables = set()
        
        if isinstance(condition, dict):
            if condition.get('type') == 'column' and condition.get('table'):
                tables.add(condition['table'])
            
            for key, value in condition.items():
                if isinstance(value, dict):
                    tables.update(self._extract_tables_from_condition(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            tables.update(self._extract_tables_from_condition(item))
        
        return tables
    
    def _get_tables_from_operator(self, operator: AlgebraicOperator) -> Set[str]:
        """Get all table names from an operator subtree."""
        tables = set()
        
        if isinstance(operator, RelationOperator):
            tables.add(operator.table_name)
        
        for child in operator.children:
            tables.update(self._get_tables_from_operator(child))
        
        return tables

class ProjectionPushdown(AlgebraicRule):
    """Push projection operators down the tree."""
    
    def __init__(self):
        super().__init__(
            name="Projection Pushdown",
            rule_type=RuleType.PUSHDOWN,
            description="Push projection below joins when beneficial"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, ProjectOperator):
            return False
        
        if not operator.children:
            return False
        
        child = operator.children[0]
        return isinstance(child, JoinOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        project_op = operator
        join_op = project_op.children[0]
        
        # Analyze which columns are needed from each side of the join
        projected_cols = self._get_column_names(project_op.columns)
        join_cols = self._extract_columns_from_join_condition(join_op.condition)
        
        # Determine columns needed from left and right
        left_cols_needed = set()
        right_cols_needed = set()
        
        # This is a simplified implementation
        # In practice, we'd need more sophisticated column tracking
        
        # For now, don't push down to avoid incorrect transformations
        return RuleApplication(success=False, transformed_operator=None,
                             description="Projection pushdown not implemented yet")
    
    def _get_column_names(self, columns: List[Union[str, Dict]]) -> Set[str]:
        """Extract column names from projection list."""
        names = set()
        for col in columns:
            if isinstance(col, str):
                names.add(col)
            elif isinstance(col, dict):
                names.add(col.get('expr', col.get('alias', '')))
        return names
    
    def _extract_columns_from_join_condition(self, condition: Any) -> Set[str]:
        """Extract columns from join condition."""
        # Reuse logic from SelectionPushdown
        pushdown = SelectionPushdown()
        return pushdown._extract_columns_from_condition(condition)

class JoinCommutativity(AlgebraicRule):
    """Apply join commutativity (R ⋈ S ≡ S ⋈ R)."""
    
    def __init__(self):
        super().__init__(
            name="Join Commutativity",
            rule_type=RuleType.COMMUTATIVITY,
            description="Swap join operands for inner joins"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, JoinOperator):
            return False
        
        # Only inner joins are commutative
        return operator.join_type.upper() == 'INNER'
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        new_join = operator.clone()
        
        # Swap children
        if len(new_join.children) >= 2:
            new_join.children[0], new_join.children[1] = new_join.children[1], new_join.children[0]
            
            # Update join condition if needed (swap table references)
            # This is a simplified implementation
            
            return RuleApplication(
                success=True,
                transformed_operator=new_join,
                description="Applied join commutativity"
            )
        
        return RuleApplication(success=False, transformed_operator=None,
                             description="Insufficient children for join")

class JoinAssociativity(AlgebraicRule):
    """Apply join associativity ((R ⋈ S) ⋈ T ≡ R ⋈ (S ⋈ T))."""
    
    def __init__(self):
        super().__init__(
            name="Join Associativity",
            rule_type=RuleType.ASSOCIATIVITY,
            description="Reorder join operations"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, JoinOperator):
            return False
        
        if not operator.children:
            return False
        
        # Check if left child is also a join
        left_child = operator.children[0]
        return isinstance(left_child, JoinOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        # Transform ((R ⋈ S) ⋈ T) to (R ⋈ (S ⋈ T))
        outer_join = operator
        inner_join = outer_join.children[0]
        
        if len(inner_join.children) < 2 or len(outer_join.children) < 2:
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Insufficient children")
        
        R = inner_join.children[0]
        S = inner_join.children[1]
        T = outer_join.children[1]
        
        # Create new structure: R ⋈ (S ⋈ T)
        new_inner_join = JoinOperator(join_type='INNER')
        new_inner_join.add_child(S.clone())
        new_inner_join.add_child(T.clone())
        
        new_outer_join = JoinOperator(join_type='INNER')
        new_outer_join.add_child(R.clone())
        new_outer_join.add_child(new_inner_join)
        
        return RuleApplication(
            success=True,
            transformed_operator=new_outer_join,
            description="Applied join associativity"
        )

class SelectionSplit(AlgebraicRule):
    """Split conjunctive selections (σ[p1 AND p2] ≡ σ[p1] ∘ σ[p2])."""
    
    def __init__(self):
        super().__init__(
            name="Selection Split",
            rule_type=RuleType.SIMPLIFICATION,
            description="Split AND conditions into separate selections"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, SelectOperator):
            return False
        
        # Check if condition contains AND
        condition = operator.condition
        if isinstance(condition, dict):
            return condition.get('type') == 'AND' and len(condition.get('operands', [])) > 1
        
        return False
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        select_op = operator
        condition = select_op.condition
        operands = condition.get('operands', [])
        
        if len(operands) < 2:
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Not enough operands to split")
        
        # Create chain of selections
        current = select_op.children[0].clone() if select_op.children else None
        
        for operand in reversed(operands):
            new_select = SelectOperator(condition=operand)
            if current:
                new_select.add_child(current)
            current = new_select
        
        return RuleApplication(
            success=True,
            transformed_operator=current,
            description=f"Split AND condition into {len(operands)} selections"
        )

class SelectionCombine(AlgebraicRule):
    """Combine consecutive selections (σ[p1] ∘ σ[p2] ≡ σ[p1 AND p2])."""
    
    def __init__(self):
        super().__init__(
            name="Selection Combine",
            rule_type=RuleType.MERGING,
            description="Combine consecutive selections into one"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, SelectOperator):
            return False
        
        if not operator.children:
            return False
        
        child = operator.children[0]
        return isinstance(child, SelectOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        outer_select = operator
        inner_select = outer_select.children[0]
        
        # Combine conditions with AND
        combined_condition = {
            'type': 'AND',
            'operands': [outer_select.condition, inner_select.condition]
        }
        
        new_select = SelectOperator(condition=combined_condition)
        if inner_select.children:
            new_select.add_child(inner_select.children[0].clone())
        
        return RuleApplication(
            success=True,
            transformed_operator=new_select,
            description="Combined consecutive selections"
        )

class ProjectionCascade(AlgebraicRule):
    """Eliminate redundant projections (π[L1] ∘ π[L2] ≡ π[L1] if L1 ⊆ L2)."""
    
    def __init__(self):
        super().__init__(
            name="Projection Cascade",
            rule_type=RuleType.ELIMINATION,
            description="Remove redundant projections"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, ProjectOperator):
            return False
        
        if not operator.children:
            return False
        
        child = operator.children[0]
        return isinstance(child, ProjectOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        outer_proj = operator
        inner_proj = outer_proj.children[0]
        
        # Get column sets
        outer_cols = self._get_column_set(outer_proj.columns)
        inner_cols = self._get_column_set(inner_proj.columns)
        
        # If outer columns are subset of inner columns, we can eliminate inner projection
        if outer_cols.issubset(inner_cols) or '*' in inner_cols:
            new_proj = outer_proj.clone()
            if inner_proj.children:
                new_proj.children = [inner_proj.children[0].clone()]
            
            return RuleApplication(
                success=True,
                transformed_operator=new_proj,
                description="Eliminated redundant inner projection"
            )
        
        return RuleApplication(success=False, transformed_operator=None,
                             description="Cannot eliminate projection")
    
    def _get_column_set(self, columns: List[Union[str, Dict]]) -> Set[str]:
        """Convert column list to set of column names."""
        col_set = set()
        for col in columns:
            if isinstance(col, str):
                col_set.add(col)
            elif isinstance(col, dict):
                col_set.add(col.get('expr', col.get('alias', '')))
        return col_set

class UnionCommutativity(AlgebraicRule):
    """Apply union commutativity (R ∪ S ≡ S ∪ R)."""
    
    def __init__(self):
        super().__init__(
            name="Union Commutativity",
            rule_type=RuleType.COMMUTATIVITY,
            description="Swap union operands"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        return isinstance(operator, UnionOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        new_union = operator.clone()
        
        # Swap children
        if len(new_union.children) >= 2:
            new_union.children[0], new_union.children[1] = new_union.children[1], new_union.children[0]
            
            return RuleApplication(
                success=True,
                transformed_operator=new_union,
                description="Applied union commutativity"
            )
        
        return RuleApplication(success=False, transformed_operator=None,
                             description="Insufficient children")

class UnionAssociativity(AlgebraicRule):
    """Apply union associativity ((R ∪ S) ∪ T ≡ R ∪ (S ∪ T))."""
    
    def __init__(self):
        super().__init__(
            name="Union Associativity",
            rule_type=RuleType.ASSOCIATIVITY,
            description="Reorder union operations"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, UnionOperator):
            return False
        
        if not operator.children:
            return False
        
        left_child = operator.children[0]
        return isinstance(left_child, UnionOperator)
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        # Similar to JoinAssociativity but for unions
        # Implementation details omitted for brevity
        return RuleApplication(success=False, transformed_operator=None,
                             description="Not implemented yet")

class PredicateSimplification(AlgebraicRule):
    """Simplify predicates (e.g., remove tautologies, contradictions)."""
    
    def __init__(self):
        super().__init__(
            name="Predicate Simplification",
            rule_type=RuleType.SIMPLIFICATION,
            description="Simplify logical predicates"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if isinstance(operator, SelectOperator):
            return self._can_simplify_condition(operator.condition)
        return False
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        if not self.is_applicable(operator):
            return RuleApplication(success=False, transformed_operator=None,
                                 description="Rule not applicable")
        
        select_op = operator
        simplified_condition = self._simplify_condition(select_op.condition)
        
        if simplified_condition != select_op.condition:
            new_select = select_op.clone()
            new_select.condition = simplified_condition
            
            return RuleApplication(
                success=True,
                transformed_operator=new_select,
                description="Simplified predicate"
            )
        
        return RuleApplication(success=False, transformed_operator=None,
                             description="No simplification possible")
    
    def _can_simplify_condition(self, condition: Any) -> bool:
        """Check if condition can be simplified."""
        if isinstance(condition, dict):
            cond_type = condition.get('type')
            
            # Check for tautologies/contradictions
            if cond_type == 'LITERAL':
                return False  # Already simplified
            
            if cond_type == 'AND':
                operands = condition.get('operands', [])
                # Check for TRUE or FALSE operands
                for op in operands:
                    if isinstance(op, dict) and op.get('type') == 'LITERAL':
                        return True
            
            if cond_type == 'OR':
                operands = condition.get('operands', [])
                # Check for TRUE or FALSE operands
                for op in operands:
                    if isinstance(op, dict) and op.get('type') == 'LITERAL':
                        return True
            
            if cond_type == 'NOT':
                operand = condition.get('operand')
                # Double negation
                if isinstance(operand, dict) and operand.get('type') == 'NOT':
                    return True
        
        return False
    
    def _simplify_condition(self, condition: Any) -> Any:
        """Simplify a condition."""
        if not isinstance(condition, dict):
            return condition
        
        cond_type = condition.get('type')
        
        if cond_type == 'AND':
            operands = condition.get('operands', [])
            # Remove TRUE operands, return FALSE if any FALSE
            new_operands = []
            for op in operands:
                if isinstance(op, dict) and op.get('type') == 'LITERAL':
                    if not op.get('value'):  # FALSE
                        return {'type': 'LITERAL', 'value': False}
                    # Skip TRUE
                else:
                    new_operands.append(op)
            
            if not new_operands:
                return {'type': 'LITERAL', 'value': True}
            if len(new_operands) == 1:
                return new_operands[0]
            
            return {'type': 'AND', 'operands': new_operands}
        
        if cond_type == 'OR':
            operands = condition.get('operands', [])
            # Remove FALSE operands, return TRUE if any TRUE
            new_operands = []
            for op in operands:
                if isinstance(op, dict) and op.get('type') == 'LITERAL':
                    if op.get('value'):  # TRUE
                        return {'type': 'LITERAL', 'value': True}
                    # Skip FALSE
                else:
                    new_operands.append(op)
            
            if not new_operands:
                return {'type': 'LITERAL', 'value': False}
            if len(new_operands) == 1:
                return new_operands[0]
            
            return {'type': 'OR', 'operands': new_operands}
        
        if cond_type == 'NOT':
            operand = condition.get('operand')
            # Double negation elimination
            if isinstance(operand, dict) and operand.get('type') == 'NOT':
                return operand.get('operand')
        
        return condition

class RedundantJoinElimination(AlgebraicRule):
    """Eliminate redundant self-joins."""
    
    def __init__(self):
        super().__init__(
            name="Redundant Join Elimination",
            rule_type=RuleType.ELIMINATION,
            description="Remove unnecessary self-joins"
        )
    
    def is_applicable(self, operator: AlgebraicOperator) -> bool:
        if not isinstance(operator, JoinOperator):
            return False
        
        if len(operator.children) < 2:
            return False
        
        # Check if joining same table with same conditions
        left = operator.children[0]
        right = operator.children[1]
        
        # Simple check: both are relations of same table
        if isinstance(left, RelationOperator) and isinstance(right, RelationOperator):
            return left.table_name == right.table_name
        
        return False
    
    def apply(self, operator: AlgebraicOperator) -> RuleApplication:
        # This is a simplified implementation
        # Real implementation would need more sophisticated analysis
        return RuleApplication(success=False, transformed_operator=None,
                             description="Not implemented yet")

def get_all_algebraic_rules() -> List[AlgebraicRule]:
    """Get all available algebraic transformation rules."""
    return [
        SelectionPushdown(),
        ProjectionPushdown(),
        JoinCommutativity(),
        JoinAssociativity(),
        SelectionSplit(),
        SelectionCombine(),
        ProjectionCascade(),
        UnionCommutativity(),
        UnionAssociativity(),
        PredicateSimplification(),
        RedundantJoinElimination(),
    ]

def apply_algebraic_rules(expression: AlgebraicExpression, 
                         rules: Optional[List[AlgebraicRule]] = None,
                         max_iterations: int = 100) -> Tuple[AlgebraicExpression, List[str]]:
    """
    Apply algebraic transformation rules to an expression.
    
    Args:
        expression: Algebraic expression to transform
        rules: List of rules to apply (None for all rules)
        max_iterations: Maximum number of rule applications
        
    Returns:
        Tuple of (transformed expression, list of applied rules)
    """
    if rules is None:
        rules = get_all_algebraic_rules()
    
    transformed = expression.clone()
    applied_rules = []
    iterations = 0
    
    # Keep applying rules until no more changes or max iterations reached
    changed = True
    while changed and iterations < max_iterations:
        changed = False
        
        for rule in rules:
            if rule.apply_to_expression(transformed):
                applied_rules.append(f"{rule.name}: {rule.description}")
                changed = True
                iterations += 1
                break  # Start over with the first rule
    
    return transformed, applied_rules