# sql_equivalence/equivalence/algebraic_equivalence.py
"""Algebraic equivalence checking for SQL queries."""

from typing import Any, Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

from .base import EquivalenceChecker, EquivalenceResult, EquivalenceType
from ..representations.algebraic.relational_algebra import AlgebraicExpression
from ..representations.algebraic.operators import (
    AlgebraicOperator, ProjectOperator, SelectOperator, JoinOperator,
    UnionOperator, IntersectOperator, ExceptOperator, AggregateOperator,
    RelationOperator, OperatorType
)



logger = logging.getLogger(__name__)

class AlgebraicRule(Enum):
    """Types of algebraic transformation rules."""
    # Basic algebraic laws
    COMMUTATIVITY = "commutativity"
    ASSOCIATIVITY = "associativity"
    DISTRIBUTIVITY = "distributivity"
    IDEMPOTENCE = "idempotence"
    
    # Selection rules
    SELECTION_PUSHDOWN = "selection_pushdown"
    SELECTION_SPLIT = "selection_split"
    SELECTION_COMBINE = "selection_combine"
    
    # Projection rules
    PROJECTION_PUSHDOWN = "projection_pushdown"
    PROJECTION_CASCADE = "projection_cascade"
    
    # Join rules
    JOIN_COMMUTATIVITY = "join_commutativity"
    JOIN_ASSOCIATIVITY = "join_associativity"
    SELECTION_JOIN_PUSHDOWN = "selection_join_pushdown"
    
    # Set operation rules
    UNION_ASSOCIATIVITY = "union_associativity"
    UNION_COMMUTATIVITY = "union_commutativity"
    DISTRIBUTIVE_UNION = "distributive_union"

@dataclass
class TransformationRule:
    """A transformation rule for algebraic expressions."""
    name: str
    rule_type: AlgebraicRule
    description: str
    apply_func: callable
    is_applicable_func: callable

class AlgebraicEquivalenceChecker(EquivalenceChecker):
    """Check equivalence using algebraic transformations."""
    
    def __init__(self):
        super().__init__()
        self.transformation_rules = self._initialize_rules()
        self.max_transformations = 100
        self.enable_normalization = True
        self.enable_proof_generation = True
    
    def set_make_hashable(self, x):
        def make_hashable(item):
            if isinstance(item, dict):
                return tuple(sorted(item.items()))
            return item
        return set( [make_hashable(item) for item in x] )
    
    def check_equivalence(self, expr1: AlgebraicExpression, 
                         expr2: AlgebraicExpression) -> EquivalenceResult:
        """Check if two algebraic expressions are equivalent."""
        start_time = time.time()
        result = EquivalenceResult(
            is_equivalent=False,
            equivalence_type=EquivalenceType.NOT_EQUIVALENT,
            confidence=0.0
        )
        
        try:
            # Quick check: exact match
            if self._are_identical(expr1, expr2):
                result.is_equivalent = True
                result.equivalence_type = EquivalenceType.EXACT
                result.confidence = 1.0
                result.add_proof_step("Expressions are structurally identical")
                return result
            
            # Normalize both expressions
            if self.enable_normalization:
                norm_expr1 = self._normalize(expr1)
                norm_expr2 = self._normalize(expr2)
                
                if self._are_identical(norm_expr1, norm_expr2):
                    result.is_equivalent = True
                    result.equivalence_type = EquivalenceType.SEMANTIC
                    result.confidence = 1.0
                    result.add_proof_step("Normalized expressions are identical")
                    return result
            
            # Try to transform expr1 into expr2
            if self._can_transform(expr1, expr2, result):
                result.is_equivalent = True
                result.equivalence_type = EquivalenceType.SEMANTIC
                result.confidence = 1.0
            else:
                # Compute structural similarity
                similarity = self.compute_similarity(expr1, expr2)
                result.confidence = similarity
                
                if similarity > 0.95:
                    result.is_equivalent = True
                    result.equivalence_type = EquivalenceType.APPROXIMATE
        
        except Exception as e:
            logger.error(f"Error checking equivalence: {e}")
            result.details['error'] = str(e)
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def compute_similarity(self, expr1: AlgebraicExpression, 
                          expr2: AlgebraicExpression) -> float:
        """Compute similarity between two algebraic expressions."""
        if not expr1.is_built() or not expr2.is_built():
            return 0.0
        
        # Get operator trees
        tree1 = expr1.expression_tree
        tree2 = expr2.expression_tree
        
        if not tree1 or not tree2:
            return 0.0
        
        # Compare various aspects
        scores = []
        
        # 1. Operator type distribution similarity
        op_dist1 = self._get_operator_distribution(tree1)
        op_dist2 = self._get_operator_distribution(tree2)
        scores.append(self._compute_distribution_similarity(op_dist1, op_dist2))
        
        # 2. Tree structure similarity
        scores.append(self._compute_tree_structure_similarity(tree1, tree2))
        
        # 3. Table and column overlap
        tables1 = self._extract_tables(expr1)
        tables2 = self._extract_tables(expr2)
        scores.append(self._compute_set_similarity(tables1, tables2))
        
        # 4. Predicate similarity
        predicates1 = self._extract_predicates(expr1)
        predicates2 = self._extract_predicates(expr2)
        scores.append(self._compute_predicate_similarity(predicates1, predicates2))
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        return sum(s * w for s, w in zip(scores, weights))
    
    def _initialize_rules(self) -> List[TransformationRule]:
        """Initialize transformation rules."""
        rules = []
        
        # Selection pushdown rule
        rules.append(TransformationRule(
            name="Selection Pushdown",
            rule_type=AlgebraicRule.SELECTION_PUSHDOWN,
            description="Push selection below join: σ[p](R ⋈ S) → σ[p1](R) ⋈ σ[p2](S)",
            apply_func=self._apply_selection_pushdown,
            is_applicable_func=self._can_apply_selection_pushdown
        ))
        
        # Join commutativity
        rules.append(TransformationRule(
            name="Join Commutativity",
            rule_type=AlgebraicRule.JOIN_COMMUTATIVITY,
            description="R ⋈ S ≡ S ⋈ R",
            apply_func=self._apply_join_commutativity,
            is_applicable_func=self._can_apply_join_commutativity
        ))
        
        # Selection split
        rules.append(TransformationRule(
            name="Selection Split",
            rule_type=AlgebraicRule.SELECTION_SPLIT,
            description="σ[p1 AND p2](R) ≡ σ[p1](σ[p2](R))",
            apply_func=self._apply_selection_split,
            is_applicable_func=self._can_apply_selection_split
        ))
        
        # Projection cascade
        rules.append(TransformationRule(
            name="Projection Cascade",
            rule_type=AlgebraicRule.PROJECTION_CASCADE,
            description="π[L1](π[L2](R)) ≡ π[L1](R) if L1 ⊆ L2",
            apply_func=self._apply_projection_cascade,
            is_applicable_func=self._can_apply_projection_cascade
        ))
        
        # Union commutativity
        rules.append(TransformationRule(
            name="Union Commutativity",
            rule_type=AlgebraicRule.UNION_COMMUTATIVITY,
            description="R ∪ S ≡ S ∪ R",
            apply_func=self._apply_union_commutativity,
            is_applicable_func=self._can_apply_union_commutativity
        ))
        
        return rules
    
    def _are_identical(self, expr1: AlgebraicExpression, 
                      expr2: AlgebraicExpression) -> bool:
        """Check if two expressions are structurally identical."""
        if not expr1.is_built() or not expr2.is_built():
            return False
        
        return self._operators_equal(expr1.root_operator, expr2.root_operator)
    
    def _operators_equal(self, op1: AlgebraicOperator, 
                        op2: AlgebraicOperator) -> bool:
        """Check if two operators are equal."""
        if type(op1) != type(op2):
            logger.info(f"Disagree: Op")
            return False
        
        if op1.operator_type != op2.operator_type:
            logger.info(f"Disagree: Op-type")
            return False
        
        # Check operator-specific attributes
        if isinstance(op1, ProjectOperator):
            if self.set_make_hashable(op1.columns) != self.set_make_hashable(op2.columns):
                logger.info(f"Disagree:\n{self.set_make_hashable(op1.columns)}\n{self.set_make_hashable(op2.columns)}")
                return False
        
        elif isinstance(op1, SelectOperator):
            if op1.condition != op2.condition:
                logger.info(f"Disagree: SelectOperator")
                return False
        
        elif isinstance(op1, JoinOperator):
            if op1.join_type != op2.join_type:
                logger.info(f"Disagree: JoinOperator.type")
                return False
            if op1.condition != op2.condition:
                logger.info(f"Disagree: JoinOperator.condition")
                return False
        
        elif isinstance(op1, RelationOperator):
            if op1.table_name != op2.table_name:
                logger.info(f"Disagree: RelationOperator.table")
                return False
        
        # Check children
        if len(op1.children) != len(op2.children):
            logger.info(f"Disagree: children")
            return False
        
        for c1, c2 in zip(op1.children, op2.children):
            if not self._operators_equal(c1, c2):
                logger.info(f"Disagree: children-structure")
                return False
        
        return True
    
    def _normalize(self, expr: AlgebraicExpression) -> AlgebraicExpression:
        """Normalize an algebraic expression."""
        normalized = expr.clone()
        
        # Apply normalization rules
        # 1. Push selections down
        self._push_selections_down(normalized.root_operator)
        
        # 2. Combine consecutive selections
        self._combine_selections(normalized.root_operator)
        
        # 3. Remove redundant projections
        self._remove_redundant_projections(normalized.root_operator)
        
        # 4. Sort commutative operations
        self._sort_commutative_operations(normalized.root_operator)
        
        return normalized
    
    def _push_selections_down(self, operator: AlgebraicOperator) -> None:
        """Push selection operators down the tree."""
        # Recursively process children first
        for child in operator.children:
            self._push_selections_down(child)
        
        # Apply selection pushdown if applicable
        if isinstance(operator, SelectOperator) and operator.children:
            child = operator.children[0]
            
            # Push through projection
            if isinstance(child, ProjectOperator):
                # Check if selection references only projected columns
                # If yes, swap selection and projection
                pass
            
            # Push through join
            elif isinstance(child, JoinOperator):
                # Split selection predicate based on tables referenced
                # Push relevant parts to each side of join
                pass
    
    def _combine_selections(self, operator: AlgebraicOperator) -> None:
        """Combine consecutive selection operators."""
        # Implementation would combine σ[p1](σ[p2](R)) into σ[p1 AND p2](R)
        pass
    
    def _remove_redundant_projections(self, operator: AlgebraicOperator) -> None:
        """Remove redundant projection operators."""
        # Implementation would remove projections that don't reduce columns
        pass
    
    def _sort_commutative_operations(self, operator: AlgebraicOperator) -> None:
        """Sort children of commutative operations for consistent ordering."""
        if operator.attributes.get('is_commutative', False):
            # Sort children by some canonical ordering (e.g., hash)
            operator.children.sort(key=lambda op: hash(op))
            ### operator.children.sort(key=lambda op: hash(self._operator_to_string(op)))
        
        # Recursively process children
        for child in operator.children:
            self._sort_commutative_operations(child)
    
    def _can_transform(self, expr1: AlgebraicExpression, 
                    expr2: AlgebraicExpression,
                    result: EquivalenceResult) -> bool:
        """Try to transform expr1 into expr2 using transformation rules."""
        # Keep track of visited states to avoid cycles
        visited = set()
        queue = [(expr1.root_operator.clone(), [])]
        
        # Create a stable string representation for hashing
        def get_operator_hash(op):
            """Get a stable hash for an operator."""
            return self._operator_to_string(op)
        
        visited.add(get_operator_hash(expr1.root_operator))
        
        iterations = 0
        
        while queue and iterations < self.max_transformations:
            current_op, proof_steps = queue.pop(0)
            iterations += 1
            
            # Check if we've reached the target
            if self._operators_equal(current_op, expr2.root_operator):
                if self.enable_proof_generation:
                    for step in proof_steps:
                        result.add_proof_step(step)
                return True
            
            # Try applying each transformation rule
            for rule in self.transformation_rules:
                if rule.is_applicable_func(current_op):
                    transformed = rule.apply_func(current_op.clone())
                    
                    if transformed:
                        transformed_hash = get_operator_hash(transformed)
                        if transformed_hash not in visited:
                            visited.add(transformed_hash)
                            new_steps = proof_steps + [f"Applied {rule.name}: {rule.description}"]
                            queue.append((transformed, new_steps))
        
        return False
    
    def _operator_to_string(self, op: AlgebraicOperator) -> str:
        """Convert operator to a stable string representation for hashing."""
        if not op:
            return ""
        
        # Create a string that uniquely identifies the operator
        parts = [
            op.operator_type.value,
            str(type(op).__name__)
        ]
        
        # Add operator-specific attributes
        if isinstance(op, SelectOperator):
            #### condition_str = json.dumps(
            ####     op.condition, sort_keys=True) if isinstance(op.condition, dict) else str(op.condition)
            #### parts.append(condition_str)
            
            if isinstance(op.condition, dict):
                parts.append(json.dumps(op.condition, sort_keys=True))
            else:
                parts.append(str(op.condition))
        elif isinstance(op, ProjectOperator):
            parts.append(str(sorted(str(c) for c in op.columns)))
        elif isinstance(op, JoinOperator):
            parts.append(op.join_type)
            if isinstance(op.condition, dict):
                parts.append(json.dumps(op.condition, sort_keys=True))
            else:
                parts.append(str(op.condition))
        elif isinstance(op, RelationOperator):
            parts.append(op.table_name)
        elif isinstance(op, AggregateOperator):
            parts.append(str(sorted(str(a) for a in op.aggregations)))
            parts.append(str(sorted(op.group_by)))
        
        # Add children hashes
        child_hashes = []
        for child in op.children:
            child_hashes.append(self._operator_to_string(child))
        parts.append(str(child_hashes))
        
        return "|".join(parts)
    
    # Transformation rule implementations

    def _can_apply_selection_pushdown(self, operator: AlgebraicOperator) -> bool:
        """Check if selection pushdown can be applied."""
        if not isinstance(operator, SelectOperator):
            return False
        
        if not operator.children:
            return False
        
        child = operator.children[0]
        return isinstance(child, (JoinOperator, ProjectOperator))
    
    def _apply_selection_pushdown(self, operator: AlgebraicOperator) -> AlgebraicOperator:
        """Apply selection pushdown transformation."""
        if not self._can_apply_selection_pushdown(operator):
            return operator
        
        select_op = operator
        child = select_op.children[0]
        
        if isinstance(child, JoinOperator):
            # Push selection through join
            # This is a simplified implementation
            # Full implementation would analyze predicates
            return operator
        
        return operator
    
    def _can_apply_join_commutativity(self, operator: AlgebraicOperator) -> bool:
        """Check if join commutativity can be applied."""
        return isinstance(operator, JoinOperator) and operator.join_type == 'INNER'
    
    def _apply_join_commutativity(self, operator: AlgebraicOperator) -> AlgebraicOperator:
        """Apply join commutativity transformation."""
        if not self._can_apply_join_commutativity(operator):
            return operator
        
        join_op = operator.clone()
        # Swap children
        if len(join_op.children) >= 2:
            join_op.children[0], join_op.children[1] = join_op.children[1], join_op.children[0]
        
        return join_op
    
    def _can_apply_selection_split(self, operator: AlgebraicOperator) -> bool:
        """Check if selection split can be applied."""
        if not isinstance(operator, SelectOperator):
            return False
        
        # Check if condition contains AND
        # This is simplified - real implementation would parse condition
        return isinstance(operator.condition, dict) and operator.condition.get('type') == 'AND'
    
    def _apply_selection_split(self, operator: AlgebraicOperator) -> AlgebraicOperator:
        """Apply selection split transformation."""
        # Simplified implementation
        return operator
    
    def _can_apply_projection_cascade(self, operator: AlgebraicOperator) -> bool:
        """Check if projection cascade can be applied."""
        if not isinstance(operator, ProjectOperator):
            return False
        
        if not operator.children:
            return False
        
        return isinstance(operator.children[0], ProjectOperator)
    
    def _apply_projection_cascade(self, operator: AlgebraicOperator) -> AlgebraicOperator:
        """Apply projection cascade transformation."""
        if not self._can_apply_projection_cascade(operator):
            return operator
        
        outer_proj = operator
        inner_proj = outer_proj.children[0]
        
        # Check if outer columns are subset of inner columns
        outer_cols = set(col['expr'] if isinstance(col, dict) else col 
                        for col in outer_proj.columns)
        inner_cols = set(col['expr'] if isinstance(col, dict) else col 
                        for col in inner_proj.columns)
        
        if outer_cols.issubset(inner_cols):
            # Can eliminate inner projection
            new_proj = outer_proj.clone()
            new_proj.children = inner_proj.children
            return new_proj
        
        return operator
    
    def _can_apply_union_commutativity(self, operator: AlgebraicOperator) -> bool:
        """Check if union commutativity can be applied."""
        return isinstance(operator, UnionOperator)
    
    def _apply_union_commutativity(self, operator: AlgebraicOperator) -> AlgebraicOperator:
        """Apply union commutativity transformation."""
        if not self._can_apply_union_commutativity(operator):
            return operator
        
        union_op = operator.clone()
        # Swap children
        if len(union_op.children) >= 2:
            union_op.children[0], union_op.children[1] = union_op.children[1], union_op.children[0]
        
        return union_op
    
    # Helper methods for similarity computation
    
    def _get_operator_distribution(self, tree) -> Dict[str, int]:
        """Get distribution of operator types in tree."""
        distribution = {}
        
        for node in tree.get_nodes():
            op_type = node.operator.operator_type.name
            distribution[op_type] = distribution.get(op_type, 0) + 1
        
        return distribution
    
    def _compute_distribution_similarity(self, dist1: Dict[str, int], 
                                       dist2: Dict[str, int]) -> float:
        """Compute similarity between two distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 0.0
        
        total_diff = 0
        total_count = 0
        
        for key in all_keys:
            count1 = dist1.get(key, 0)
            count2 = dist2.get(key, 0)
            total_diff += abs(count1 - count2)
            total_count += max(count1, count2)
        
        if total_count == 0:
            return 0.0
        
        return 1.0 - (total_diff / total_count)
    
    def _compute_tree_structure_similarity(self, tree1, tree2) -> float:
        """Compute structural similarity between trees."""
        # Simple metrics
        height_sim = 1.0 - abs(tree1.get_height() - tree2.get_height()) / max(tree1.get_height(), tree2.get_height(), 1)
        
        node_count1 = len(tree1.get_nodes())
        node_count2 = len(tree2.get_nodes())
        node_sim = 1.0 - abs(node_count1 - node_count2) / max(node_count1, node_count2, 1)
        
        return (height_sim + node_sim) / 2
    
    def _extract_tables(self, expr: AlgebraicExpression) -> Set[str]:
        """Extract all table names from expression."""
        tables = set()
        
        def visit(op: AlgebraicOperator):
            if isinstance(op, RelationOperator):
                tables.add(op.table_name)
            for child in op.children:
                visit(child)
        
        if expr.root_operator:
            visit(expr.root_operator)
        
        return tables
    
    def _extract_predicates(self, expr: AlgebraicExpression) -> List[Any]:
        """Extract all predicates from expression."""
        predicates = []
        
        def visit(op: AlgebraicOperator):
            if isinstance(op, SelectOperator):
                predicates.append(op.condition)
            elif isinstance(op, JoinOperator) and op.condition:
                predicates.append(op.condition)
            for child in op.children:
                visit(child)
        
        if expr.root_operator:
            visit(expr.root_operator)
        
        return predicates
    
    def _compute_set_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_predicate_similarity(self, preds1: List[Any], 
                                    preds2: List[Any]) -> float:
        """Compute similarity between predicate lists."""
        if not preds1 and not preds2:
            return 1.0
        if not preds1 or not preds2:
            return 0.0
        
        # Simplified - just check for exact matches
        matches = 0
        for p1 in preds1:
            for p2 in preds2:
                if p1 == p2:
                    matches += 1
                    break
        
        return matches / max(len(preds1), len(preds2))

# Additional imports needed
import time