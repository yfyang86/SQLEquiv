# sql_equivalence/parser/ast_builder.py
"""AST builder for creating custom abstract syntax trees from SQL queries."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlglot
from sqlglot import expressions as exp

class NodeType(Enum):
    """Types of AST nodes."""
    # Query structure
    QUERY = "query"
    SUBQUERY = "subquery"
    CTE = "cte"
    
    # Clauses
    SELECT = "select"
    FROM = "from"
    WHERE = "where"
    GROUP_BY = "group_by"
    HAVING = "having"
    ORDER_BY = "order_by"
    LIMIT = "limit"
    
    # Joins
    JOIN = "join"
    
    # Set operations
    UNION = "union"
    INTERSECT = "intersect"
    EXCEPT = "except"
    
    # Expressions
    COLUMN = "column"
    LITERAL = "literal"
    STAR = "star"
    ALIAS = "alias"
    
    # Functions
    FUNCTION = "function"
    AGGREGATE = "aggregate"
    WINDOW = "window"
    
    # Operators
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    COMPARISON = "comparison"
    LOGICAL = "logical"
    
    # Others
    TABLE = "table"
    CASE = "case"
    IN = "in"
    EXISTS = "exists"
    BETWEEN = "between"

@dataclass
class ASTNode:
    """Represents a node in the abstract syntax tree."""
    node_type: NodeType
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'ASTNode') -> None:
        """Add a child node."""
        self.children.append(child)
    
    def get_children_by_type(self, node_type: NodeType) -> List['ASTNode']:
        """Get all children of a specific type."""
        return [child for child in self.children if child.node_type == node_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': self.node_type.value,
            'value': self.value,
            'attributes': self.attributes,
            'children': [child.to_dict() for child in self.children]
        }
    
    def __repr__(self) -> str:
        return f"ASTNode({self.node_type.value}, value={self.value})"

class ASTBuilder:
    """Builds custom AST from sqlglot expressions."""
    
    def __init__(self):
        self.node_mapping = {
            exp.Select: self._build_select,
            exp.From: self._build_from,
            exp.Where: self._build_where,
            exp.Group: self._build_group_by,
            exp.Having: self._build_having,
            exp.Order: self._build_order_by,
            exp.Limit: self._build_limit,
            exp.Join: self._build_join,
            exp.Union: self._build_union,
            exp.Intersect: self._build_intersect,
            exp.Except: self._build_except,
            exp.Table: self._build_table,
            exp.Column: self._build_column,
            exp.Literal: self._build_literal,
            exp.Star: self._build_star,
            exp.Alias: self._build_alias,
            exp.Func: self._build_function,
            exp.Window: self._build_window,
            exp.Case: self._build_case,
            exp.In: self._build_in,
            exp.Exists: self._build_exists,
            exp.Between: self._build_between,
            exp.Subquery: self._build_subquery,
            exp.CTE: self._build_cte,
        }
    
    def build(self, sqlglot_ast: exp.Expression) -> ASTNode:
        """Build custom AST from sqlglot expression."""
        return self._build_node(sqlglot_ast)
    
    def _build_node(self, expr: exp.Expression) -> Optional[ASTNode]:
        """Build a node from a sqlglot expression."""
        if expr is None:
            return None
        
        # Check if we have a specific builder for this expression type
        for expr_type, builder_func in self.node_mapping.items():
            if isinstance(expr, expr_type):
                return builder_func(expr)
        
        # Handle binary operations
        if isinstance(expr, exp.Binary):
            return self._build_binary_op(expr)
        
        # Handle unary operations
        if isinstance(expr, exp.Unary):
            return self._build_unary_op(expr)
        
        # Handle comparison operations
        if isinstance(expr, exp.Condition):
            return self._build_comparison(expr)
        
        # Default: create a generic query node
        if hasattr(expr, 'expressions'):
            node = ASTNode(NodeType.QUERY)
            for child_expr in expr.expressions:
                child_node = self._build_node(child_expr)
                if child_node:
                    node.add_child(child_node)
            return node
        
        # If we can't handle it, return None
        return None
    
    def _build_select(self, expr: exp.Select) -> ASTNode:
        """Build SELECT node."""
        node = ASTNode(NodeType.SELECT)
        
        # Process SELECT expressions
        for select_expr in expr.expressions:
            child_node = self._build_node(select_expr)
            if child_node:
                node.add_child(child_node)
        
        # Add attributes
        if expr.distinct:
            node.attributes['distinct'] = True
        
        # Process other clauses
        if expr.args.get('from'):
            from_node = self._build_from(expr.args['from'])
            if from_node:
                node.add_child(from_node)
        
        if expr.args.get('where'):
            where_node = self._build_where(expr.args['where'])
            if where_node:
                node.add_child(where_node)
        
        if expr.args.get('group'):
            group_node = self._build_group_by(expr.args['group'])
            if group_node:
                node.add_child(group_node)
        
        if expr.args.get('having'):
            having_node = self._build_having(expr.args['having'])
            if having_node:
                node.add_child(having_node)
        
        if expr.args.get('order'):
            order_node = self._build_order_by(expr.args['order'])
            if order_node:
                node.add_child(order_node)
        
        if expr.args.get('limit'):
            limit_node = self._build_limit(expr.args['limit'])
            if limit_node:
                node.add_child(limit_node)
        
        return node
    
    def _build_from(self, expr: exp.From) -> ASTNode:
        """Build FROM node."""
        node = ASTNode(NodeType.FROM)
        
        # Process table references
        if expr.this:
            table_node = self._build_node(expr.this)
            if table_node:
                node.add_child(table_node)
        
        # Process joins
        if hasattr(expr, 'joins'):
            for join in expr.joins:
                join_node = self._build_node(join)
                if join_node:
                    node.add_child(join_node)
        
        return node
    
    def _build_where(self, expr: exp.Where) -> ASTNode:
        """Build WHERE node."""
        node = ASTNode(NodeType.WHERE)
        
        if expr.this:
            condition_node = self._build_node(expr.this)
            if condition_node:
                node.add_child(condition_node)
        
        return node
    
    def _build_group_by(self, expr: exp.Group) -> ASTNode:
        """Build GROUP BY node."""
        node = ASTNode(NodeType.GROUP_BY)
        
        for group_expr in expr.expressions:
            child_node = self._build_node(group_expr)
            if child_node:
                node.add_child(child_node)
        
        return node
    
    def _build_having(self, expr: exp.Having) -> ASTNode:
        """Build HAVING node."""
        node = ASTNode(NodeType.HAVING)
        
        if expr.this:
            condition_node = self._build_node(expr.this)
            if condition_node:
                node.add_child(condition_node)
        
        return node
    
    def _build_order_by(self, expr: exp.Order) -> ASTNode:
        """Build ORDER BY node."""
        node = ASTNode(NodeType.ORDER_BY)
        
        for order_expr in expr.expressions:
            child_node = self._build_node(order_expr)
            if child_node:
                node.add_child(child_node)
                # Add sort direction
                if hasattr(order_expr, 'desc') and order_expr.desc:
                    child_node.attributes['direction'] = 'DESC'
                else:
                    child_node.attributes['direction'] = 'ASC'
        
        return node
    
    def _build_limit(self, expr: exp.Limit) -> ASTNode:
        """Build LIMIT node."""
        node = ASTNode(NodeType.LIMIT)
        
        if expr.expression:
            node.value = self._get_literal_value(expr.expression)
        
        if expr.offset:
            node.attributes['offset'] = self._get_literal_value(expr.offset)
        
        return node
    
    def _build_join(self, expr: exp.Join) -> ASTNode:
        """Build JOIN node."""
        node = ASTNode(NodeType.JOIN)
        
        # Set join type
        join_type = 'INNER'  # default
        if expr.kind:
            join_type = expr.kind.upper()
        node.attributes['join_type'] = join_type
        
        # Add table
        if expr.this:
            table_node = self._build_node(expr.this)
            if table_node:
                node.add_child(table_node)
        
        # Add ON condition
        if expr.on:
            on_node = self._build_node(expr.on)
            if on_node:
                on_node.attributes['is_join_condition'] = True
                node.add_child(on_node)
        
        return node
    
    def _build_union(self, expr: exp.Union) -> ASTNode:
        """Build UNION node."""
        node = ASTNode(NodeType.UNION)
        
        if expr.distinct:
            node.attributes['distinct'] = True
        else:
            node.attributes['all'] = True
        
        # Add both sides of union
        if expr.this:
            left_node = self._build_node(expr.this)
            if left_node:
                node.add_child(left_node)
        
        if expr.expression:
            right_node = self._build_node(expr.expression)
            if right_node:
                node.add_child(right_node)
        
        return node
    
    def _build_intersect(self, expr: exp.Intersect) -> ASTNode:
        """Build INTERSECT node."""
        node = ASTNode(NodeType.INTERSECT)
        
        if expr.distinct:
            node.attributes['distinct'] = True
        
        # Add both sides
        if expr.this:
            left_node = self._build_node(expr.this)
            if left_node:
                node.add_child(left_node)
        
        if expr.expression:
            right_node = self._build_node(expr.expression)
            if right_node:
                node.add_child(right_node)
        
        return node
    
    def _build_except(self, expr: exp.Except) -> ASTNode:
        """Build EXCEPT node."""
        node = ASTNode(NodeType.EXCEPT)
        
        if expr.distinct:
            node.attributes['distinct'] = True
        
        # Add both sides
        if expr.this:
            left_node = self._build_node(expr.this)
            if left_node:
                node.add_child(left_node)
        
        if expr.expression:
            right_node = self._build_node(expr.expression)
            if right_node:
                node.add_child(right_node)
        
        return node
    
    def _build_table(self, expr: exp.Table) -> ASTNode:
        """Build TABLE node."""
        node = ASTNode(NodeType.TABLE)
        node.value = expr.name
        
        if expr.db:
            node.attributes['schema'] = expr.db
        
        if expr.alias:
            node.attributes['alias'] = expr.alias
        
        return node
    
    def _build_column(self, expr: exp.Column) -> ASTNode:
        """Build COLUMN node."""
        node = ASTNode(NodeType.COLUMN)
        node.value = expr.name
        
        if expr.table:
            node.attributes['table'] = expr.table
        
        return node
    
    def _build_literal(self, expr: exp.Literal) -> ASTNode:
        """Build LITERAL node."""
        node = ASTNode(NodeType.LITERAL)
        node.value = expr.this
        
        # Infer type
        if expr.is_int:
            node.attributes['type'] = 'INTEGER'
        elif expr.is_number:
            node.attributes['type'] = 'NUMERIC'
        elif expr.is_string:
            node.attributes['type'] = 'STRING'
        else:
            node.attributes['type'] = 'UNKNOWN'
        
        return node
    
    def _build_star(self, expr: exp.Star) -> ASTNode:
        """Build STAR (*) node."""
        node = ASTNode(NodeType.STAR)
        node.value = '*'
        
        if hasattr(expr, 'table') and expr.table:
            node.attributes['table'] = expr.table
        
        return node
    
    def _build_alias(self, expr: exp.Alias) -> ASTNode:
        """Build ALIAS node."""
        node = ASTNode(NodeType.ALIAS)
        node.value = expr.alias
        
        # Add the aliased expression as child
        if expr.this:
            child_node = self._build_node(expr.this)
            if child_node:
                node.add_child(child_node)
        
        return node
    
    def _build_function(self, expr: exp.Func) -> ASTNode:
        """Build FUNCTION node."""
        func_name = expr.name.upper()
        
        # Determine function type
        if func_name in {'SUM', 'COUNT', 'AVG', 'MIN', 'MAX'}:
            node = ASTNode(NodeType.AGGREGATE)
        elif func_name in {'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LEAD', 'LAG'}:
            node = ASTNode(NodeType.WINDOW)
        else:
            node = ASTNode(NodeType.FUNCTION)
        
        node.value = func_name
        
        # Add function arguments
        for arg in expr.args.get('expressions', []):
            arg_node = self._build_node(arg)
            if arg_node:
                node.add_child(arg_node)
        
        # Handle DISTINCT in aggregate functions
        if hasattr(expr, 'distinct') and expr.distinct:
            node.attributes['distinct'] = True
        
        return node
    
    def _build_window(self, expr: exp.Window) -> ASTNode:
        """Build WINDOW node."""
        node = ASTNode(NodeType.WINDOW)
        
        # Add the window function
        if expr.this:
            func_node = self._build_node(expr.this)
            if func_node:
                node.add_child(func_node)
        
        # Add PARTITION BY
        if expr.partition_by:
            for partition_expr in expr.partition_by:
                partition_node = self._build_node(partition_expr)
                if partition_node:
                    partition_node.attributes['is_partition'] = True
                    node.add_child(partition_node)
        
        # Add ORDER BY
        if expr.order:
            order_node = self._build_node(expr.order)
            if order_node:
                order_node.attributes['is_window_order'] = True
                node.add_child(order_node)
        
        return node
    
    def _build_case(self, expr: exp.Case) -> ASTNode:
        """Build CASE node."""
        node = ASTNode(NodeType.CASE)
        
        # Add CASE expression if exists
        if expr.this:
            case_expr_node = self._build_node(expr.this)
            if case_expr_node:
                case_expr_node.attributes['is_case_expr'] = True
                node.add_child(case_expr_node)
        
        # Add WHEN clauses
        for when in expr.args.get('ifs', []):
            when_node = self._build_node(when)
            if when_node:
                when_node.attributes['is_when'] = True
                node.add_child(when_node)
        
        # Add ELSE clause
        if expr.default:
            else_node = self._build_node(expr.default)
            if else_node:
                else_node.attributes['is_else'] = True
                node.add_child(else_node)
        
        return node
    
    def _build_in(self, expr: exp.In) -> ASTNode:
        """Build IN node."""
        node = ASTNode(NodeType.IN)
        
        # Add expression
        if expr.this:
            expr_node = self._build_node(expr.this)
            if expr_node:
                node.add_child(expr_node)
        
        # Add values or subquery
        for value in expr.expressions:
            value_node = self._build_node(value)
            if value_node:
                node.add_child(value_node)
        
        return node
    
    def _build_exists(self, expr: exp.Exists) -> ASTNode:
        """Build EXISTS node."""
        node = ASTNode(NodeType.EXISTS)
        
        # Add subquery
        if expr.this:
            subquery_node = self._build_node(expr.this)
            if subquery_node:
                node.add_child(subquery_node)
        
        return node
    
    def _build_between(self, expr: exp.Between) -> ASTNode:
        """Build BETWEEN node."""
        node = ASTNode(NodeType.BETWEEN)
        
        # Add expression
        if expr.this:
            expr_node = self._build_node(expr.this)
            if expr_node:
                node.add_child(expr_node)
        
        # Add low bound
        if expr.low:
            low_node = self._build_node(expr.low)
            if low_node:
                low_node.attributes['is_low_bound'] = True
                node.add_child(low_node)
        
        # Add high bound
        if expr.high:
            high_node = self._build_node(expr.high)
            if high_node:
                high_node.attributes['is_high_bound'] = True
                node.add_child(high_node)
        
        return node
    
    def _build_subquery(self, expr: exp.Subquery) -> ASTNode:
        """Build SUBQUERY node."""
        node = ASTNode(NodeType.SUBQUERY)
        
        # Add the query
        if expr.this:
            query_node = self._build_node(expr.this)
            if query_node:
                node.add_child(query_node)
        
        # Add alias if exists
        if expr.alias:
            node.attributes['alias'] = expr.alias
        
        return node
    
    def _build_cte(self, expr: exp.CTE) -> ASTNode:
        """Build CTE node."""
        node = ASTNode(NodeType.CTE)
        
        if expr.alias:
            node.value = expr.alias
        
        # Add the query
        if expr.this:
            query_node = self._build_node(expr.this)
            if query_node:
                node.add_child(query_node)
        
        return node
    
    def _build_binary_op(self, expr: exp.Binary) -> ASTNode:
        """Build binary operation node."""
        node = ASTNode(NodeType.BINARY_OP)
        node.value = expr.key.upper()
        
        # Add left operand
        if expr.this:
            left_node = self._build_node(expr.this)
            if left_node:
                node.add_child(left_node)
        
        # Add right operand
        if expr.expression:
            right_node = self._build_node(expr.expression)
            if right_node:
                node.add_child(right_node)
        
        return node
    
    def _build_unary_op(self, expr: exp.Unary) -> ASTNode:
        """Build unary operation node."""
        node = ASTNode(NodeType.UNARY_OP)
        node.value = expr.key.upper()
        
        # Add operand
        if expr.this:
            operand_node = self._build_node(expr.this)
            if operand_node:
                node.add_child(operand_node)
        
        return node
    
    def _build_comparison(self, expr: exp.Condition) -> ASTNode:
        """Build comparison node."""
        node = ASTNode(NodeType.COMPARISON)
        node.value = expr.key.upper()
        
        # Add operands
        if hasattr(expr, 'this') and expr.this:
            left_node = self._build_node(expr.this)
            if left_node:
                node.add_child(left_node)
        
        if hasattr(expr, 'expression') and expr.expression:
            right_node = self._build_node(expr.expression)
            if right_node:
                node.add_child(right_node)
        
        return node
    
    def _get_literal_value(self, expr: exp.Expression) -> Any:
        """Extract literal value from expression."""
        if isinstance(expr, exp.Literal):
            return expr.this
        elif isinstance(expr, exp.Number):
            return float(expr.this)
        elif isinstance(expr, exp.DataType.Type):
            return str(expr)
        else:
            return str(expr)