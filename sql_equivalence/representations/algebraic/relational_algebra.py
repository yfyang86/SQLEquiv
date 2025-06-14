# sql_equivalence/representations/algebraic/relational_algebra.py (comprehensive fix)
"""Relational algebra representation of SQL queries."""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
import copy
import logging

from ...parser.sql_parser import ParsedQuery
from ...parser.ast_builder import NodeType, ASTNode  # Import both NodeType and ASTNode
from .operators import (
    AlgebraicOperator, ProjectOperator, SelectOperator, JoinOperator,
    UnionOperator, IntersectOperator, ExceptOperator, AggregateOperator,
    RelationOperator, GroupByOperator, OrderByOperator, DistinctOperator,
    OperatorType
)
from .expression_tree import ExpressionTree, ExpressionNode

logger = logging.getLogger(__name__)

class AlgebraicExpression:
    """Represents a SQL query as a relational algebra expression."""
    
    def __init__(self, parsed_query: ParsedQuery):
        """
        Initialize algebraic expression from parsed query.
        
        Args:
            parsed_query: ParsedQuery object containing AST and metadata
        """
        self.parsed_query = parsed_query
        self.root_operator: Optional[AlgebraicOperator] = None
        self.expression_tree: Optional[ExpressionTree] = None
        self._is_built = False

    def visualize(self, output_path: Optional[str] = None) -> Any:
        """Visualize the algebraic expression tree."""
        if not self.expression_tree:
            raise ValueError("Expression not built yet")
        
        return self.expression_tree.visualize(output_path)


    def build(self) -> None:
        """Build the algebraic expression from the parsed query."""
        if self._is_built:
            return
            
        try:
            logger.debug(f"Building algebraic expression from AST: {self.parsed_query.ast}")
            
            # Build operator tree from AST
            self.root_operator = self._build_from_ast(self.parsed_query.ast)
            
            if not self.root_operator:
                # If AST building fails, try building from sqlglot AST
                logger.warning("Failed to build from custom AST, trying sqlglot AST")
                if hasattr(self.parsed_query, 'sqlglot_ast') and self.parsed_query.sqlglot_ast:
                    self.root_operator = self._build_from_sqlglot_ast(self.parsed_query.sqlglot_ast)
                else:
                    # Fallback: build a simple structure from metadata
                    self.root_operator = self._build_fallback()
            
            # Create expression tree
            if self.root_operator:
                self.expression_tree = self._create_expression_tree(self.root_operator)
            
            self._is_built = True
            logger.debug(f"Successfully built algebraic expression: {self.root_operator}")
            
        except Exception as e:
            logger.error(f"Failed to build algebraic expression: {e}")
            # Create a minimal fallback structure
            self.root_operator = self._build_fallback()
            self._is_built = True
    
    def _get_node_type_string(self, node_type) -> str:
        """Convert node type to string, handling enums and strings."""
        if hasattr(node_type, 'value'):
            return str(node_type.value).upper()
        elif hasattr(node_type, 'name'):
            return str(node_type.name).upper()
        else:
            return str(node_type).upper()
    
    def _build_fallback(self) -> Optional[AlgebraicOperator]:
        """Build a fallback algebraic expression from metadata."""
        logger.debug("Building fallback algebraic expression from metadata")
        
        # Get tables from metadata
        tables = list(self.parsed_query.metadata.tables)
        if not tables:
            # Default table if none found
            tables = ['unknown_table']
        
        # Start with first table
        current_operator = RelationOperator(tables[0])
        
        # Add simple projection for all columns
        project_op = ProjectOperator(columns=['*'])
        project_op.add_child(current_operator)
        
        return project_op
    
    def _build_from_sqlglot_ast(self, sqlglot_ast) -> Optional[AlgebraicOperator]:
        """Build from sqlglot AST as fallback."""
        import sqlglot
        from sqlglot import expressions as exp
        
        logger.debug(f"Building from sqlglot AST: {type(sqlglot_ast)}")
        
        if isinstance(sqlglot_ast, exp.Select):
            # Build FROM clause
            from_expr = sqlglot_ast.find(exp.From)
            if from_expr and from_expr.this:
                table_name = from_expr.this.name if hasattr(from_expr.this, 'name') else 'table'
                current_operator = RelationOperator(table_name)
            else:
                current_operator = RelationOperator('unknown_table')
            
            # Add WHERE clause
            where_expr = sqlglot_ast.find(exp.Where)
            if where_expr:
                condition_str = where_expr.this.sql() if where_expr.this else ""
                select_op = SelectOperator(condition=condition_str)
                select_op.add_child(current_operator)
                current_operator = select_op
            
            # Add projection
            project_cols = []
            for expr in sqlglot_ast.expressions:
                if hasattr(expr, 'alias'):
                    project_cols.append(expr.alias if expr.alias else str(expr))
                else:
                    project_cols.append(str(expr))
            
            if not project_cols:
                project_cols = ['*']
            
            project_op = ProjectOperator(columns=project_cols)
            project_op.add_child(current_operator)
            current_operator = project_op
            
            return current_operator
        
        return None
    
    def clone(self) -> 'AlgebraicExpression':
        """Create a deep copy of this algebraic expression."""
        # Create a new instance with the same parsed query
        cloned = AlgebraicExpression(self.parsed_query)
        
        # Clone the operator tree
        if self.root_operator:
            cloned.root_operator = self.root_operator.clone()
        
        # Clone the expression tree
        if self.expression_tree:
            cloned.expression_tree = ExpressionTree()
            if self.expression_tree.root:
                cloned.expression_tree.root = self._clone_expression_node(self.expression_tree.root)
        
        cloned._is_built = self._is_built
        
        return cloned
    
    def _clone_expression_node(self, node: ExpressionNode) -> ExpressionNode:
        """Recursively clone an expression node."""
        cloned_node = ExpressionNode(
            node_id=node.node_id,
            operator=node.operator.clone() if node.operator else None
        )
        
        # Clone children
        for child in node.children:
            cloned_child = self._clone_expression_node(child)
            cloned_node.add_child(cloned_child)
        
        return cloned_node
    
    def _build_from_ast(self, ast_node: ASTNode) -> Optional[AlgebraicOperator]:
        """Build algebraic operators from AST node."""
        if not ast_node:
            logger.debug("AST node is None")
            return None
        
        # Get node type as uppercase string
        node_type = self._get_node_type_string(ast_node.node_type)
        
        logger.debug(f"Building from AST node type: {node_type}")
        
        if node_type == 'SELECT':
            return self._build_select_query(ast_node)
        elif node_type == 'UNION':
            return self._build_union(ast_node)
        elif node_type == 'INTERSECT':
            return self._build_intersect(ast_node)
        elif node_type == 'EXCEPT':
            return self._build_except(ast_node)
        else:
            logger.warning(f"Unsupported node type: {node_type}")
            # Try to handle as SELECT if it has children
            if ast_node.children:
                return self._build_select_query(ast_node)
            return None
    
    def _build_select_query(self, ast_node: ASTNode) -> Optional[AlgebraicOperator]:
        """Build operators for a SELECT query."""
        logger.debug(f"Building SELECT query from node with {len(ast_node.children)} children")
        
        # Start with FROM clause (base relations)
        from_operator = self._build_from_clause(ast_node)
        if not from_operator:
            # If no FROM clause found, create a default one
            logger.debug("No FROM clause found, using metadata")
            tables = list(self.parsed_query.metadata.tables)
            if tables:
                from_operator = RelationOperator(tables[0])
            else:
                from_operator = RelationOperator('unknown_table')
        
        current_operator = from_operator
        
        # Add WHERE clause (selection)
        where_clause = self._find_child_by_type(ast_node, 'WHERE')
        if where_clause:
            logger.debug("Found WHERE clause")
            select_op = self._build_where_clause(where_clause)
            if select_op:
                select_op.add_child(current_operator)
                current_operator = select_op
        
        # Add GROUP BY clause
        group_by_clause = self._find_child_by_type(ast_node, 'GROUP_BY')
        if group_by_clause:
            logger.debug("Found GROUP BY clause")
            group_op = self._build_group_by_clause(group_by_clause)
            if group_op:
                group_op.add_child(current_operator)
                current_operator = group_op
        
        # Add HAVING clause
        having_clause = self._find_child_by_type(ast_node, 'HAVING')
        if having_clause:
            logger.debug("Found HAVING clause")
            having_op = self._build_having_clause(having_clause)
            if having_op:
                having_op.add_child(current_operator)
                current_operator = having_op
        
        # Add SELECT clause (projection)
        select_list = self._find_child_by_type(ast_node, 'SELECT_LIST')
        if select_list:
            logger.debug("Found SELECT list")
            project_op = self._build_projection(select_list)
            if project_op:
                project_op.add_child(current_operator)
                current_operator = project_op
        else:
            # Default projection
            logger.debug("No SELECT list found, adding default projection")
            project_op = ProjectOperator(columns=['*'])
            project_op.add_child(current_operator)
            current_operator = project_op
        
        # Add DISTINCT if present
        if self._has_distinct(ast_node):
            logger.debug("Adding DISTINCT")
            distinct_op = DistinctOperator()
            distinct_op.add_child(current_operator)
            current_operator = distinct_op
        
        # Add ORDER BY clause
        order_by_clause = self._find_child_by_type(ast_node, 'ORDER_BY')
        if order_by_clause:
            logger.debug("Found ORDER BY clause")
            order_op = self._build_order_by_clause(order_by_clause)
            if order_op:
                order_op.add_child(current_operator)
                current_operator = order_op
        
        return current_operator
    
    def _build_from_clause(self, ast_node: ASTNode) -> Optional[AlgebraicOperator]:
        """Build operators for FROM clause."""
        from_clause = self._find_child_by_type(ast_node, 'FROM')
        if not from_clause:
            logger.debug("No FROM clause found in AST")
            return None
        
        tables = []
        joins = []
        
        # Extract tables and joins
        for child in from_clause.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'TABLE':
                table_name = child.attributes.get('name', 'unknown')
                alias = child.attributes.get('alias')
                relation_op = RelationOperator(table_name, alias)
                tables.append(relation_op)
                logger.debug(f"Found table: {table_name}")
            elif child_type == 'JOIN':
                joins.append(child)
                logger.debug(f"Found join")
        
        if not tables:
            logger.debug("No tables found in FROM clause")
            return None
        
        # Start with first table
        current_operator = tables[0]
        
        # Process joins
        for i, join_node in enumerate(joins):
            if i + 1 < len(tables):
                right_table = tables[i + 1]
                join_op = self._build_join(join_node, current_operator, right_table)
                current_operator = join_op
        
        return current_operator
    
    def _build_join(self, join_node: ASTNode, left_op: AlgebraicOperator, 
                    right_op: AlgebraicOperator) -> JoinOperator:
        """Build join operator."""
        join_type = join_node.attributes.get('type', 'INNER')
        join_op = JoinOperator(join_type=join_type)
        
        join_op.add_child(left_op)
        join_op.add_child(right_op)
        
        # Add join condition
        on_clause = self._find_child_by_type(join_node, 'ON')
        if on_clause and on_clause.children:
            condition = self._build_condition(on_clause.children[0])
            join_op.set_condition(condition)
        
        return join_op
    
    def _build_where_clause(self, where_node: ASTNode) -> Optional[SelectOperator]:
        """Build selection operator from WHERE clause."""
        if not where_node.children:
            return None
        
        condition = self._build_condition(where_node.children[0])
        return SelectOperator(condition=condition)
    
    def _build_condition(self, condition_node: ASTNode) -> Union[Dict[str, Any], str]:
        """Build condition expression from AST node."""
        if not condition_node:
            return ""
        
        node_type = self._get_node_type_string(condition_node.node_type)
        
        if node_type == 'COMPARISON':
            left = self._build_condition(condition_node.children[0]) if condition_node.children else None
            right = self._build_condition(condition_node.children[1]) if len(condition_node.children) > 1 else None
            operator = condition_node.attributes.get('operator', '=')
            
            return {
                'type': 'COMPARISON',
                'operator': operator,
                'left': left,
                'right': right
            }
        
        elif node_type == 'AND':
            operands = [self._build_condition(child) for child in condition_node.children]
            return {
                'type': 'AND',
                'operands': operands
            }
        
        elif node_type == 'OR':
            operands = [self._build_condition(child) for child in condition_node.children]
            return {
                'type': 'OR',
                'operands': operands
            }
        
        elif node_type == 'NOT':
            operand = self._build_condition(condition_node.children[0]) if condition_node.children else None
            return {
                'type': 'NOT',
                'operand': operand
            }
        
        elif node_type == 'COLUMN':
            return {
                'type': 'COLUMN',
                'name': condition_node.attributes.get('name', ''),
                'table': condition_node.attributes.get('table')
            }
        
        elif node_type == 'LITERAL':
            return {
                'type': 'LITERAL',
                'value': condition_node.attributes.get('value')
            }
        
        else:
            # Return as string for unsupported types
            return str(condition_node.attributes)
    
    def _build_projection(self, select_list_node: ASTNode) -> Optional[ProjectOperator]:
        """Build projection operator from SELECT list."""
        columns = []
        
        for child in select_list_node.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'COLUMN':
                col_name = child.attributes.get('name', '*')
                alias = child.attributes.get('alias')
                
                if alias:
                    columns.append({'expr': col_name, 'alias': alias})
                else:
                    columns.append(col_name)
            
            elif child_type == 'FUNCTION':
                func_name = child.attributes.get('name', 'UNKNOWN')
                alias = child.attributes.get('alias')
                
                # Build function expression
                func_expr = self._build_function_expr(child)
                
                if alias:
                    columns.append({'expr': func_expr, 'alias': alias})
                else:
                    columns.append(func_expr)
            
            elif child_type == 'WILDCARD':
                columns.append('*')
        
        if columns:
            return ProjectOperator(columns=columns)
        
        return None
    
    def _build_function_expr(self, func_node: ASTNode) -> str:
        """Build function expression string."""
        func_name = func_node.attributes.get('name', 'UNKNOWN')
        args = []
        
        for child in func_node.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'COLUMN':
                args.append(child.attributes.get('name', ''))
            else:
                args.append(str(child.attributes))
        
        return f"{func_name}({', '.join(args)})"
    
    def _build_group_by_clause(self, group_by_node: ASTNode) -> Optional[Union[GroupByOperator, AggregateOperator]]:
        """Build GROUP BY operator."""
        columns = []
        
        for child in group_by_node.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'COLUMN':
                columns.append(child.attributes.get('name', ''))
        
        if columns:
            # Check for aggregations in SELECT
            aggregations = self._extract_aggregations(group_by_node.parent if hasattr(group_by_node, 'parent') else None)
            
            if aggregations:
                return AggregateOperator(
                    aggregations=aggregations,
                    group_by=columns
                )
            else:
                return GroupByOperator(group_by_columns=columns)
        
        return None
    
    def _extract_aggregations(self, select_node: Optional[ASTNode]) -> List[Dict[str, Any]]:
        """Extract aggregation functions from SELECT."""
        aggregations = []
        
        if not select_node:
            return aggregations
        
        select_list = self._find_child_by_type(select_node, 'SELECT_LIST')
        if not select_list:
            return aggregations
        
        for child in select_list.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'FUNCTION':
                func_name = child.attributes.get('name', '').upper()
                if func_name in ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']:
                    agg = {
                        'function': func_name,
                        'arguments': [c.attributes.get('name', '') for c in child.children],
                        'alias': child.attributes.get('alias')
                    }
                    aggregations.append(agg)
        
        return aggregations
    
    def _build_having_clause(self, having_node: ASTNode) -> Optional[SelectOperator]:
        """Build selection operator for HAVING clause."""
        if not having_node.children:
            return None
        
        condition = self._build_condition(having_node.children[0])
        # HAVING is just a selection after aggregation
        return SelectOperator(condition=condition)
    
    def _build_order_by_clause(self, order_by_node: ASTNode) -> Optional[OrderByOperator]:
        """Build ORDER BY operator."""
        columns = []
        
        for child in order_by_node.children:
            child_type = self._get_node_type_string(child.node_type)
            if child_type == 'COLUMN':
                col_spec = {
                    'column': child.attributes.get('name', ''),
                    'direction': child.attributes.get('direction', 'ASC')
                }
                columns.append(col_spec)
        
        if columns:
            return OrderByOperator(order_by_columns=columns)
        
        return None
    
    def _build_union(self, union_node: ASTNode) -> Optional[UnionOperator]:
        """Build UNION operator."""
        if len(union_node.children) < 2:
            return None
        
        left = self._build_from_ast(union_node.children[0])
        right = self._build_from_ast(union_node.children[1])
        
        if left and right:
            distinct = union_node.attributes.get('distinct', True)
            union_op = UnionOperator(distinct=distinct)
            union_op.add_child(left)
            union_op.add_child(right)
            return union_op
        
        return None
    
    def _build_intersect(self, intersect_node: ASTNode) -> Optional[IntersectOperator]:
        """Build INTERSECT operator."""
        if len(intersect_node.children) < 2:
            return None
        
        left = self._build_from_ast(intersect_node.children[0])
        right = self._build_from_ast(intersect_node.children[1])
        
        if left and right:
            distinct = intersect_node.attributes.get('distinct', True)
            intersect_op = IntersectOperator(distinct=distinct)
            intersect_op.add_child(left)
            intersect_op.add_child(right)
            return intersect_op
        
        return None
    
    def _build_except(self, except_node: ASTNode) -> Optional[ExceptOperator]:
        """Build EXCEPT operator."""
        if len(except_node.children) < 2:
            return None
        
        left = self._build_from_ast(except_node.children[0])
        right = self._build_from_ast(except_node.children[1])
        
        if left and right:
            distinct = except_node.attributes.get('distinct', True)
            except_op = ExceptOperator(distinct=distinct)
            except_op.add_child(left)
            except_op.add_child(right)
            return except_op
        
        return None
    
    def _find_child_by_type(self, node: ASTNode, child_type: Union[str, 'NodeType']) -> Optional[ASTNode]:
        """Find child node by type."""
        if not node or not hasattr(node, 'children'):
            return None
        
        # Convert to uppercase string for comparison
        if isinstance(child_type, str):
            target_type = child_type.upper()
        else:
            target_type = self._get_node_type_string(child_type)
        
        for child in node.children:
            child_type_str = self._get_node_type_string(child.node_type)
            if child_type_str == target_type:
                return child
        
        return None
    
    def _has_distinct(self, select_node: ASTNode) -> bool:
        """Check if SELECT has DISTINCT."""
        return select_node.attributes.get('distinct', False)
    
    def _create_expression_tree(self, operator: AlgebraicOperator) -> ExpressionTree:
        """Create expression tree from operator tree."""
        tree = ExpressionTree()
        tree.root = self._operator_to_node(operator, 0)
        return tree
    
    def _operator_to_node(self, operator: AlgebraicOperator, 
                          node_id: int) -> ExpressionNode:
        """Convert operator to expression node."""
        node = ExpressionNode(node_id=node_id, operator=operator)
        
        # Recursively add children
        child_id = node_id + 1
        for child_op in operator.children:
            child_node = self._operator_to_node(child_op, child_id)
            node.add_child(child_node)
            child_id = self._get_max_node_id(child_node) + 1
        
        return node
    
    def _get_max_node_id(self, node: ExpressionNode) -> int:
        """Get maximum node ID in subtree."""
        max_id = node.node_id
        for child in node.children:
            child_max = self._get_max_node_id(child)
            if child_max > max_id:
                max_id = child_max
        return max_id
    
    def to_string(self) -> str:
        """Convert algebraic expression to string notation."""
        if not self._is_built or not self.root_operator:
            return "Empty expression"
        
        return self.root_operator.to_string()
    
    def to_canonical_form(self) -> str:
        """Convert to canonical form for comparison."""
        if not self._is_built or not self.root_operator:
            return ""
        
        # Apply normalization rules
        from ...transformations.algebraic_rules import (
            get_all_algebraic_rules, apply_algebraic_rules
        )
        
        # Apply only deterministic rules for canonical form
        canonical_rules = [
            rule for rule in get_all_algebraic_rules()
            if rule.name in ['Selection Split', 'Projection Cascade', 
                           'Predicate Simplification']
        ]
        
        normalized, _ = apply_algebraic_rules(self, canonical_rules, max_iterations=50)
        
        return normalized.to_string()
    
    def optimize(self) -> 'AlgebraicExpression':
        """Apply optimization rules to the expression."""
        from ...transformations.algebraic_rules import apply_algebraic_rules
        
        optimized, applied_rules = apply_algebraic_rules(self)
        logger.info(f"Applied {len(applied_rules)} optimization rules")
        
        return optimized
    
    def is_built(self) -> bool:
        """Check if expression has been built."""
        return self._is_built
    
    def get_tables(self) -> Set[str]:
        """Get all table names in the expression."""
        tables = set()
        
        def collect_tables(op: AlgebraicOperator):
            if isinstance(op, RelationOperator):
                tables.add(op.table_name)
            for child in op.children:
                collect_tables(child)
        
        if self.root_operator:
            collect_tables(self.root_operator)
        
        return tables
    
    def get_columns(self) -> Set[str]:
        """Get all column names in the expression."""
        columns = set()
        
        def collect_columns(op: AlgebraicOperator):
            if isinstance(op, ProjectOperator):
                for col in op.columns:
                    if isinstance(col, str) and col != '*':
                        columns.add(col)
                    elif isinstance(col, dict):
                        expr = col.get('expr', '')
                        if expr and expr != '*':
                            columns.add(expr)
            
            for child in op.children:
                collect_columns(child)
        
        if self.root_operator:
            collect_columns(self.root_operator)
        
        return columns
    
    def __repr__(self) -> str:
        return f"AlgebraicExpression(built={self._is_built}, tables={self.get_tables()})"