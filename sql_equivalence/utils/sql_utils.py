# sql_equivalence/utils/sql_utils.py
"""SQL utility functions."""

import re
from typing import List, Set, Tuple, Optional, Dict, Any
import sqlglot
from sqlglot import expressions as exp
import logging

logger = logging.getLogger(__name__)

def format_sql(sql: str, dialect: str = 'postgres', pretty: bool = True) -> str:
    """
    Format SQL query string.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        pretty: Whether to use pretty formatting
        
    Returns:
        Formatted SQL string
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        return parsed.sql(dialect=dialect, pretty=pretty)
    except Exception as e:
        logger.warning(f"Failed to format SQL: {e}")
        return sql

def validate_sql(sql: str, dialect: str = 'postgres') -> Tuple[bool, Optional[str]]:
    """
    Validate SQL syntax.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True, None
    except Exception as e:
        return False, str(e)

def extract_tables_from_sql(sql: str, dialect: str = 'postgres') -> Set[str]:
    """
    Extract all table names from SQL query.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Set of table names
    """
    tables = set()
    
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        
        # Find all table references
        for table in parsed.find_all(exp.Table):
            if table.name:
                tables.add(table.name)
        
        # Also check subqueries
        for subquery in parsed.find_all(exp.Subquery):
            if subquery.this:
                sub_tables = extract_tables_from_sql(
                    subquery.this.sql(dialect=dialect), 
                    dialect=dialect
                )
                tables.update(sub_tables)
    
    except Exception as e:
        logger.warning(f"Failed to extract tables: {e}")
    
    return tables

def extract_columns_from_sql(sql: str, dialect: str = 'postgres') -> Set[str]:
    """
    Extract all column references from SQL query.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Set of column names
    """
    columns = set()
    
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        
        # Find all column references
        for column in parsed.find_all(exp.Column):
            if column.name:
                columns.add(column.name)
    
    except Exception as e:
        logger.warning(f"Failed to extract columns: {e}")
    
    return columns

def sql_to_lowercase_normalized(sql: str) -> str:
    """
    Convert SQL to lowercase while preserving string literals.
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL string
    """
    # Pattern to match string literals
    string_pattern = r"'[^']*'"
    
    # Find all string literals
    strings = []
    for match in re.finditer(string_pattern, sql):
        strings.append((match.start(), match.end(), match.group()))
    
    # Convert to lowercase
    result = sql.lower()
    
    # Restore string literals
    offset = 0
    for start, end, original in strings:
        result = result[:start] + original + result[end:]
    
    return result

def split_sql_statements(sql: str) -> List[str]:
    """
    Split multiple SQL statements.
    
    Args:
        sql: SQL string potentially containing multiple statements
        
    Returns:
        List of individual SQL statements
    """
    # Simple implementation - split by semicolon
    # More sophisticated implementation would handle semicolons in strings
    statements = []
    
    # Remove comments first
    sql_no_comments = remove_sql_comments(sql)
    
    # Split by semicolon
    parts = sql_no_comments.split(';')
    
    for part in parts:
        part = part.strip()
        if part:
            statements.append(part)
    
    return statements

def is_select_query(sql: str, dialect: str = 'postgres') -> bool:
    """
    Check if SQL is a SELECT query.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        True if SELECT query
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        return isinstance(parsed, exp.Select)
    except:
        # Fallback to simple string check
        sql_lower = sql.strip().lower()
        return sql_lower.startswith('select')

def get_sql_type(sql: str, dialect: str = 'postgres') -> str:
    """
    Get the type of SQL statement.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Statement type (SELECT, INSERT, UPDATE, DELETE, etc.)
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        return type(parsed).__name__.upper()
    except:
        # Fallback to simple string check
        sql_lower = sql.strip().lower()
        for keyword in ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']:
            if sql_lower.startswith(keyword):
                return keyword.upper()
        return 'UNKNOWN'

def standardize_sql(sql: str, dialect: str = 'postgres') -> str:
    """
    Standardize SQL query format.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Standardized SQL string
    """
    try:
        # Parse and regenerate
        parsed = sqlglot.parse_one(sql, read=dialect)
        
        # Apply some standardization transformations
        # 1. Remove unnecessary parentheses
        # 2. Standardize operator spacing
        # 3. Consistent keyword casing
        
        return parsed.sql(dialect=dialect, normalize=True, pretty=False)
    
    except Exception as e:
        logger.warning(f"Failed to standardize SQL: {e}")
        return sql

def remove_sql_comments(sql: str) -> str:
    """
    Remove comments from SQL string.
    
    Args:
        sql: SQL query string
        
    Returns:
        SQL string without comments
    """
    # Remove single-line comments
    sql = re.sub(r'--[^\n]*', '', sql)
    
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    return sql.strip()

def get_query_complexity_score(sql: str, dialect: str = 'postgres') -> float:
    """
    Calculate complexity score for SQL query.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Complexity score
    """
    score = 0.0
    
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        
        # Count different elements
        tables = len(list(parsed.find_all(exp.Table)))
        joins = len(list(parsed.find_all(exp.Join)))
        subqueries = len(list(parsed.find_all(exp.Subquery)))
        aggregates = len(list(parsed.find_all(exp.AggFunc)))
        windows = len(list(parsed.find_all(exp.Window)))
        
        # Calculate score
        score += tables * 1.0
        score += joins * 2.0
        score += subqueries * 3.0
        score += aggregates * 1.5
        score += windows * 2.5
        
        # Check for complex features
        if parsed.find(exp.Union):
            score += 2.0
        if parsed.find(exp.CTE):
            score += 2.5
        if parsed.find(exp.Case):
            score += 1.5
    
    except Exception as e:
        logger.warning(f"Failed to calculate complexity: {e}")
        score = 1.0
    
    return score

def extract_query_metadata(sql: str, dialect: str = 'postgres') -> Dict[str, Any]:
    """
    Extract comprehensive metadata from SQL query.
    
    Args:
        sql: SQL query string
        dialect: SQL dialect
        
    Returns:
        Dictionary with query metadata
    """
    metadata = {
        'tables': set(),
        'columns': set(),
        'joins': [],
        'aggregations': [],
        'filters': [],
        'ordering': [],
        'grouping': [],
        'has_subquery': False,
        'has_cte': False,
        'has_window_functions': False,
        'complexity_score': 0.0
    }
    
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        
        # Extract tables
        metadata['tables'] = extract_tables_from_sql(sql, dialect)
        
        # Extract columns
        metadata['columns'] = extract_columns_from_sql(sql, dialect)
        
        # Extract joins
        for join in parsed.find_all(exp.Join):
            join_info = {
                'type': join.kind or 'INNER',
                'table': join.this.name if join.this else None,
                'condition': join.on.sql() if join.on else None
            }
            metadata['joins'].append(join_info)
        
        # Extract aggregations
        for agg in parsed.find_all(exp.AggFunc):
            agg_info = {
                'function': agg.name,
                'arguments': [arg.sql() for arg in agg.args.get('expressions', [])]
            }
            metadata['aggregations'].append(agg_info)
        
        # Extract filters (WHERE conditions)
        where = parsed.find(exp.Where)
        if where and where.this:
            metadata['filters'].append(where.this.sql())
        
        # Extract ordering
        order = parsed.find(exp.Order)
        if order:
            for expr in order.expressions:
                order_info = {
                    'column': expr.this.sql() if expr.this else None,
                    'direction': 'DESC' if expr.desc else 'ASC'
                }
                metadata['ordering'].append(order_info)
        
        # Extract grouping
        group = parsed.find(exp.Group)
        if group:
            metadata['grouping'] = [expr.sql() for expr in group.expressions]
        
        # Check for subqueries
        metadata['has_subquery'] = bool(parsed.find(exp.Subquery))
        
        # Check for CTEs
        metadata['has_cte'] = bool(parsed.find(exp.CTE))
        
        # Check for window functions
        metadata['has_window_functions'] = bool(parsed.find(exp.Window))
        
        # Calculate complexity
        metadata['complexity_score'] = get_query_complexity_score(sql, dialect)
    
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
    
    # Convert sets to lists for JSON serialization
    metadata['tables'] = list(metadata['tables'])
    metadata['columns'] = list(metadata['columns'])
    
    return metadata