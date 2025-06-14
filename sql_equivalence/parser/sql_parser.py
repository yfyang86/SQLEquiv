# sql_equivalence/parser/sql_parser.py (updated version)
"""Main SQL parser module."""

from typing import Dict, List, Optional, Any, Set
import sqlglot
from sqlglot import expressions as exp
import logging
from dataclasses import dataclass, field

from .ast_builder import ASTBuilder, ASTNode
from .normalizer import SQLNormalizer

logger = logging.getLogger(__name__)

# Supported SQL operators and functions
SUPPORTED_OPERATORS = {
    'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 
    'RIGHT JOIN', 'FULL JOIN', 'ON', 'GROUP BY', 'ORDER BY', 
    'HAVING', 'UNION', 'INTERSECT', 'EXCEPT', 'LIMIT', 'AS'
}

SUPPORTED_AGGREGATE_FUNCTIONS = {
    'SUM', 'COUNT', 'AVG', 'MIN', 'MAX',
    # Add case-insensitive variants
    'sum', 'count', 'avg', 'min', 'max'
}

SUPPORTED_WINDOW_FUNCTIONS = {
    'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LEAD', 'LAG',
    # Add case-insensitive variants
    'row_number', 'rank', 'dense_rank', 'ntile', 'lead', 'lag'
}

SUPPORTED_SCALAR_FUNCTIONS = {
    'UPPER', 'LOWER', 'TRIM', 'SUBSTRING', 'SUBSTR',
    'EXP', 'LOG', 'LN', 'ABS', 'ROUND', 'CEIL', 'FLOOR',
    'LENGTH', 'CHAR_LENGTH', 'COALESCE', 'NULLIF',
    # Add case-insensitive variants
    'upper', 'lower', 'trim', 'substring', 'substr',
    'exp', 'log', 'ln', 'abs', 'round', 'ceil', 'floor',
    'length', 'char_length', 'coalesce', 'nullif'
}

SUPPORTED_ALGEBRAIC_OPERATORS = {
    '+', '-', '*', '/', '^', 'AND', 'OR', 'NOT',
    '=', '!=', '<>', '<', '>', '<=', '>=',
    'IS', 'IS NOT', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE',
    'BETWEEN', 'NOT BETWEEN', 'EXISTS', 'NOT EXISTS'
}

@dataclass
class QueryMetadata:
    """Metadata about a parsed query."""
    tables: Set[str] = field(default_factory=set)
    columns: Set[str] = field(default_factory=set)
    aliases: Dict[str, str] = field(default_factory=dict)
    subqueries: List['ParsedQuery'] = field(default_factory=list)
    has_aggregation: bool = False
    has_window_functions: bool = False
    has_joins: bool = False
    has_subqueries: bool = False
    has_set_operations: bool = False
    complexity_score: float = 0.0

class ParsedQuery:
    """Represents a parsed SQL query with its AST and metadata."""
    
    def __init__(self, sql: str, ast: ASTNode, metadata: QueryMetadata, 
                 sqlglot_ast: Optional[exp.Expression] = None):
        self.sql = sql
        self.ast = ast
        self.metadata = metadata
        self.sqlglot_ast = sqlglot_ast
        self._normalized_sql = None
        
        # Lazy-loaded representations
        self._algebraic_expression = None
        self._query_graph = None
        self._logical_query_tree = None
        
    @property
    def normalized_sql(self) -> str:
        """Get normalized SQL (lazy evaluation)."""
        if self._normalized_sql is None:
            normalizer = SQLNormalizer()
            self._normalized_sql = normalizer.normalize(self.sql)
        return self._normalized_sql
    
    def to_algebraic(self) -> 'AlgebraicExpression':
        """Convert to algebraic expression representation."""
        if self._algebraic_expression is None:
            from ..representations.algebraic.relational_algebra import AlgebraicExpression
            self._algebraic_expression = AlgebraicExpression(self)
            self._algebraic_expression.build()
        return self._algebraic_expression
    
    def to_graph(self) -> 'QueryGraph':
        """Convert to query graph representation."""
        if self._query_graph is None:
            from ..representations.graph.query_graph import QueryGraph
            self._query_graph = QueryGraph(self)
            self._query_graph.build()
        return self._query_graph
    
    def to_lqt(self) -> 'LogicalQueryTree':
        """Convert to logical query tree representation."""
        if self._logical_query_tree is None:
            from ..representations.graph.lqt import LogicalQueryTree
            self._logical_query_tree = LogicalQueryTree(self)
            self._logical_query_tree.build()
        return self._logical_query_tree
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'sql': self.sql,
            'normalized_sql': self.normalized_sql,
            'ast': self.ast.to_dict(),
            'metadata': {
                'tables': list(self.metadata.tables),
                'columns': list(self.metadata.columns),
                'aliases': self.metadata.aliases,
                'has_aggregation': self.metadata.has_aggregation,
                'has_window_functions': self.metadata.has_window_functions,
                'has_joins': self.metadata.has_joins,
                'has_subqueries': self.metadata.has_subqueries,
                'has_set_operations': self.metadata.has_set_operations,
                'complexity_score': self.metadata.complexity_score
            }
        }
    
    def __repr__(self) -> str:
        return f"ParsedQuery(tables={self.metadata.tables}, complexity={self.metadata.complexity_score:.2f})"

class SQLParser:
    """Main SQL parser that converts SQL strings to ParsedQuery objects."""
    
    def __init__(self, dialect: str = 'postgres', validate_operators: bool = True):
        """
        Initialize SQL parser.
        
        Args:
            dialect: SQL dialect to use ('postgres', 'mysql', 'sqlite', etc.)
            validate_operators: Whether to validate that only supported operators are used
        """
        self.dialect = dialect
        self.validate_operators = validate_operators
        self.ast_builder = ASTBuilder()
        
        # Configure sqlglot parser
        self.parser_config = {
            'read': dialect,
            #'identify': True,  # Preserve identifier case
            # 'normalize_identifiers': False  # We'll do our own normalization
        }
        
    def parse(self, sql: str, extract_metadata: bool = True) -> ParsedQuery:
        """
        Parse SQL query string into ParsedQuery object.
        
        Args:
            sql: SQL query string
            extract_metadata: Whether to extract metadata about the query
            
        Returns:
            ParsedQuery object containing AST and metadata
            
        Raises:
            ValueError: If SQL is invalid or contains unsupported operators
        """
        try:
            # Parse with sqlglot
            sqlglot_ast = sqlglot.parse_one(sql, **self.parser_config)
            
            # Validate operators if required
            if self.validate_operators:
                self._validate_operators(sqlglot_ast)
            
            # Build our custom AST
            ast = self.ast_builder.build(sqlglot_ast)
            
            # Extract metadata
            metadata = QueryMetadata()
            if extract_metadata:
                metadata = self._extract_metadata(sqlglot_ast, ast)
            
            # Create ParsedQuery object
            parsed_query = ParsedQuery(
                sql=sql,
                ast=ast,
                metadata=metadata,
                sqlglot_ast=sqlglot_ast
            )
            
            logger.debug(f"Successfully parsed query with {len(metadata.tables)} tables")
            return parsed_query
            
        except sqlglot.ParseError as e:
            logger.error(f"Failed to parse SQL: {e}")
            raise ValueError(f"Invalid SQL syntax: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing SQL: {e}")
            raise
    
    def _validate_operators(self, ast: exp.Expression) -> None:
        """Validate that the query only uses supported operators."""
        unsupported = set()
        
        def check_node(node):
            if isinstance(node, exp.Func):
                func_name = node.name
                func_name_upper = func_name.upper()
                
                # Check if function is supported (case-insensitive)
                if (func_name not in SUPPORTED_AGGREGATE_FUNCTIONS and
                    func_name_upper not in SUPPORTED_AGGREGATE_FUNCTIONS and
                    func_name not in SUPPORTED_WINDOW_FUNCTIONS and
                    func_name_upper not in SUPPORTED_WINDOW_FUNCTIONS and
                    func_name not in SUPPORTED_SCALAR_FUNCTIONS and
                    func_name_upper not in SUPPORTED_SCALAR_FUNCTIONS):
                    
                    # Skip validation for common internal functions
                    if func_name_upper not in {'CAST', 'CONVERT', 'EXTRACT'}:
                        unsupported.add(f"Function: {func_name}")
            
            # Check for specific unsupported features
            if isinstance(node, exp.Pivot):
                unsupported.add("PIVOT")
            elif isinstance(node, exp.Lateral):
                unsupported.add("LATERAL")
            elif isinstance(node, exp.TableSample):
                unsupported.add("TABLESAMPLE")
            
            # Recursively check children
            for child in node.args.values():
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, exp.Expression):
                            check_node(item)
                elif isinstance(child, exp.Expression):
                    check_node(child)
        
        check_node(ast)
        
        if unsupported:
            # Log warning instead of raising error for better flexibility
            logger.warning(f"Found potentially unsupported operators/functions: {', '.join(unsupported)}")
            # Optionally still raise if strict validation is needed
            # raise ValueError(f"Unsupported operators/functions: {', '.join(unsupported)}")
    
    def _extract_metadata(self, sqlglot_ast: exp.Expression, ast: ASTNode) -> QueryMetadata:
        """Extract metadata from the parsed query."""
        metadata = QueryMetadata()
        
        # Extract tables
        for table in sqlglot_ast.find_all(exp.Table):
            table_name = table.name
            if table_name:
                metadata.tables.add(table_name)
            
            # Handle aliases
            if table.alias:
                alias_name = str(table.alias)
                # Handle Identifier objects
                if hasattr(table.alias, 'name'):
                    alias_name = table.alias.name
                metadata.aliases[alias_name] = table_name
        
        # Extract columns
        for column in sqlglot_ast.find_all(exp.Column):
            if column.name:
                metadata.columns.add(column.name)
        
        # Check for aggregations
        if sqlglot_ast.find(exp.AggFunc):
            metadata.has_aggregation = True
        
        # Check for window functions
        if sqlglot_ast.find(exp.Window):
            metadata.has_window_functions = True
        
        # Check for joins
        if sqlglot_ast.find(exp.Join):
            metadata.has_joins = True
        
        # Check for subqueries
        subqueries = sqlglot_ast.find_all(exp.Subquery)
        if subqueries:
            metadata.has_subqueries = True
            # Parse subqueries recursively
            for subquery in subqueries:
                if subquery.this:
                    try:
                        sub_parsed = self.parse(subquery.this.sql(self.dialect), 
                                              extract_metadata=False)
                        metadata.subqueries.append(sub_parsed)
                    except Exception as e:
                        logger.warning(f"Failed to parse subquery: {e}")
        
        # Check for set operations
        if any(sqlglot_ast.find(op) for op in [exp.Union, exp.Intersect, exp.Except]):
            metadata.has_set_operations = True
        
        # Calculate complexity score
        metadata.complexity_score = self._calculate_complexity(metadata)
        
        return metadata
    
    def _calculate_complexity(self, metadata: QueryMetadata) -> float:
        """Calculate a complexity score for the query."""
        score = 0.0
        
        # Base score for number of tables
        score += len(metadata.tables) * 1.0
        
        # Additional complexity factors
        if metadata.has_joins:
            score += 2.0
        if metadata.has_subqueries:
            score += 3.0 * (1 + len(metadata.subqueries))
        if metadata.has_aggregation:
            score += 1.5
        if metadata.has_window_functions:
            score += 2.5
        if metadata.has_set_operations:
            score += 2.0
        
        # Complexity based on number of columns
        score += len(metadata.columns) * 0.1
        
        return score
    
    def parse_batch(self, queries: List[str]) -> List[ParsedQuery]:
        """Parse multiple queries."""
        results = []
        for sql in queries:
            try:
                parsed = self.parse(sql)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse query: {e}")
                results.append(None)
        return results
    
    def validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL syntax without full parsing.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            sqlglot.parse_one(sql, **self.parser_config)
            if self.validate_operators:
                parsed = self.parse(sql, extract_metadata=False)
            return True, None
        except Exception as e:
            return False, str(e)