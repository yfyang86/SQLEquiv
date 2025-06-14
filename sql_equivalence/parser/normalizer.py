# sql_equivalence/parser/normalizer.py
"""SQL normalization utilities."""

import re
from typing import Dict, List, Optional, Set
import sqlglot
from sqlglot import expressions as exp

class SQLNormalizer:
    """Normalizes SQL queries for consistent comparison."""
    
    def __init__(self, preserve_semantics: bool = True):
        """
        Initialize SQL normalizer.
        
        Args:
            preserve_semantics: Whether to preserve semantic meaning during normalization
        """
        self.preserve_semantics = preserve_semantics
        
        # Normalization rules
        self.rules = [
            self._normalize_whitespace,
            self._normalize_case,
            self._normalize_quotes,
            self._normalize_parentheses,
            self._normalize_aliases,
            self._normalize_table_references,
            self._normalize_column_order_in_group_by,
            self._normalize_join_conditions,
        ]
        
        if not preserve_semantics:
            # Additional aggressive normalization
            self.rules.extend([
                self._remove_redundant_parentheses,
                self._normalize_boolean_expressions,
                self._normalize_null_comparisons,
            ])
    
    def normalize(self, sql: str) -> str:
        """
        Normalize SQL query string.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL string
        """
        normalized = sql
        
        # Apply normalization rules
        for rule in self.rules:
            normalized = rule(normalized)
        
        # Parse and reformat using sqlglot for consistent formatting
        try:
            parsed = sqlglot.parse_one(normalized)
            normalized = parsed.sql(pretty=False, normalize=True)
        except:
            # If parsing fails, return the result of text-based normalization
            pass
        
        return normalized.strip()
    
    def _normalize_whitespace(self, sql: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple whitespaces with single space
        sql = re.sub(r'\s+', ' ', sql)
        # Remove whitespace around operators
        sql = re.sub(r'\s*([,;()=<>!+\-*/])\s*', r'\1', sql)
        # Add space after commas
        sql = re.sub(r',(?! )', ', ', sql)
        # Add space around comparison operators
        sql = re.sub(r'([<>!=]+)', r' \1 ', sql)
        # Clean up multiple spaces again
        sql = re.sub(r'\s+', ' ', sql)
        return sql.strip()
    
    def _normalize_case(self, sql: str) -> str:
        """Normalize keyword case to uppercase."""
        keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL',
            'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'OFFSET',
            'UNION', 'INTERSECT', 'EXCEPT', 'ALL', 'DISTINCT',
            'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS',
            'SUM', 'COUNT', 'AVG', 'MIN', 'MAX',
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LEAD', 'LAG',
            'OVER', 'PARTITION', 'ASC', 'DESC'
        }
        
        # Use word boundaries to match whole words only
        pattern = r'\b(' + '|'.join(keywords) + r')\b'
        sql = re.sub(pattern, lambda m: m.group().upper(), sql, flags=re.IGNORECASE)
        
        return sql
    
    def _normalize_quotes(self, sql: str) -> str:
        """Normalize quote characters."""
        # Standardize to double quotes for identifiers
        sql = re.sub(r'`([^`]+)`', r'"\1"', sql)
        # Standardize to single quotes for strings
        sql = re.sub(r'"([^"]+)"(?![a-zA-Z_])', r"'\1'", sql)
        return sql
    
    def _normalize_parentheses(self, sql: str) -> str:
        """Normalize parentheses placement."""
        # Remove spaces inside parentheses
        sql = re.sub(r'\(\s+', '(', sql)
        sql = re.sub(r'\s+\)', ')', sql)
        # Add spaces around parentheses in certain contexts
        sql = re.sub(r'(\w)\(', r'\1 (', sql)
        sql = re.sub(r'\)(\w)', r') \1', sql)
        return sql
    
    def _normalize_aliases(self, sql: str) -> str:
        """Normalize table and column aliases."""
        # Remove optional AS keyword for aliases
        sql = re.sub(r'\s+AS\s+', ' ', sql, flags=re.IGNORECASE)
        return sql
    
    def _normalize_table_references(self, sql: str) -> str:
        """Normalize table references (schema.table)."""
        # This is a simple implementation; more complex logic might be needed
        # for handling different database dialects
        return sql
    
    def _normalize_column_order_in_group_by(self, sql: str) -> str:
        """Normalize column order in GROUP BY clauses."""
        # This requires parsing; for now, we'll skip it in text-based normalization
        return sql
    
    def _normalize_join_conditions(self, sql: str) -> str:
        """Normalize JOIN conditions."""
        # Normalize JOIN ... ON syntax
        sql = re.sub(r'\s+ON\s+', ' ON ', sql, flags=re.IGNORECASE)
        return sql
    
    def _remove_redundant_parentheses(self, sql: str) -> str:
        """Remove redundant parentheses (aggressive normalization)."""
        # This is complex to do correctly without parsing
        # For now, we'll only remove obviously redundant ones
        sql = re.sub(r'\((\w+)\)', r'\1', sql)
        return sql
    
    def _normalize_boolean_expressions(self, sql: str) -> str:
        """Normalize boolean expressions (aggressive normalization)."""
        # Convert 1=1 to TRUE, 0=1 to FALSE
        sql = re.sub(r'\b1\s*=\s*1\b', 'TRUE', sql)
        sql = re.sub(r'\b0\s*=\s*1\b', 'FALSE', sql)
        sql = re.sub(r'\b1\s*=\s*0\b', 'FALSE', sql)
        return sql
    
    def _normalize_null_comparisons(self, sql: str) -> str:
        """Normalize NULL comparisons (aggressive normalization)."""
        # Convert = NULL to IS NULL
        sql = re.sub(r'=\s*NULL\b', 'IS NULL', sql, flags=re.IGNORECASE)
        # Convert != NULL or <> NULL to IS NOT NULL
        sql = re.sub(r'(!=|<>)\s*NULL\b', 'IS NOT NULL', sql, flags=re.IGNORECASE)
        return sql
    
    def normalize_with_ast(self, sql: str) -> str:
        """
        Normalize SQL using AST transformation.
        This is more accurate but slower than text-based normalization.
        """
        try:
            # Parse SQL
            ast = sqlglot.parse_one(sql)
            
            # Apply AST-based transformations
            ast = self._normalize_ast(ast)
            
            # Convert back to SQL
            return ast.sql(pretty=False, normalize=True)
        except Exception as e:
            # Fall back to text-based normalization
            return self.normalize(sql)
    
    def _normalize_ast(self, ast: exp.Expression) -> exp.Expression:
        """Apply normalization transformations to AST."""
        # Sort columns in GROUP BY
        if isinstance(ast, exp.Select) and ast.group:
            # Sort group by expressions for consistent ordering
            ast.args['group'].expressions.sort(key=lambda x: x.sql())
        
        # Normalize JOIN conditions (a.id = b.id vs b.id = a.id)
        for join in ast.find_all(exp.Join):
            if join.on:
                join.args['on'] = self._normalize_comparison(join.on)
        
        # Recursively process subqueries
        for subquery in ast.find_all(exp.Subquery):
            if subquery.this:
                subquery.args['this'] = self._normalize_ast(subquery.this)
        
        return ast
    
    def _normalize_comparison(self, expr: exp.Expression) -> exp.Expression:
        """Normalize comparison expressions for consistent ordering."""
        if isinstance(expr, exp.EQ):
            # Sort operands lexicographically
            left_sql = expr.this.sql() if expr.this else ""
            right_sql = expr.expression.sql() if expr.expression else ""
            
            if left_sql > right_sql:
                # Swap operands
                expr.args['this'], expr.args['expression'] = expr.expression, expr.this
        
        return expr
