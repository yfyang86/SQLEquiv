# sql_equivalence/examples/basic_examples.py
"""Basic examples of SQL equivalence analysis."""

from sql_equivalence import SQLEquivalenceAnalyzer
from sql_equivalence.parser import SQLParser
from sql_equivalence.representations.algebraic import AlgebraicExpression
from sql_equivalence.utils import format_sql, visualize_query_graph

def example_simple_equivalence():
    """Example: Check if two simple queries are equivalent."""
    print("=== Simple Query Equivalence Example ===\n")
    
    # Initialize analyzer
    analyzer = SQLEquivalenceAnalyzer()
    
    # Define two equivalent queries
    sql1 = """
    SELECT name, age 
    FROM users 
    WHERE age > 18
    """
    
    sql2 = """
    SELECT name, age 
    FROM users 
    WHERE age >= 19
    """
    
    print("Query 1:")
    print(format_sql(sql1))
    print("\nQuery 2:")
    print(format_sql(sql2))
    
    # Check equivalence
    result = analyzer.analyze(sql1, sql2)
    
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    # Show method results
    print("\nMethod Results:")
    for method, method_result in result.method_results.items():
        print(f"  {method}: {method_result['is_equivalent']} "
              f"(confidence: {method_result['confidence']:.2%})")
    
    return result

def example_join_reordering():
    """Example: Join commutativity - different join orders."""
    print("\n=== Join Reordering Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Equivalent queries with different join orders
    sql1 = """
    SELECT u.name, o.order_id, p.product_name
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    JOIN products p ON o.product_id = p.product_id
    WHERE u.age > 25
    """
    
    sql2 = """
    SELECT u.name, o.order_id, p.product_name
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    JOIN products p ON o.product_id = p.product_id
    WHERE u.age > 25
    """
    
    print("Query 1 (users → orders → products):")
    print(format_sql(sql1))
    print("\nQuery 2 (orders → users → products):")
    print(format_sql(sql2))
    
    # Check equivalence
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    
    # Show proof steps if available
    if result.method_results['algebraic'].get('details', {}).get('proof_steps'):
        print("\nProof steps:")
        for step in result.method_results['algebraic']['details']['proof_steps']:
            print(f"  - {step}")
    
    return result

def example_predicate_pushdown():
    """Example: Selection/predicate pushdown optimization."""
    print("\n=== Predicate Pushdown Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Query with predicate after join
    sql1 = """
    SELECT *
    FROM (
        SELECT u.*, o.*
        FROM users u
        JOIN orders o ON u.user_id = o.user_id
    ) AS joined
    WHERE joined.age > 30 AND joined.status = 'active'
    """
    
    # Query with predicate pushed down
    sql2 = """
    SELECT u.*, o.*
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    WHERE u.age > 30 AND o.status = 'active'
    """
    
    print("Query 1 (predicate after join):")
    print(format_sql(sql1))
    print("\nQuery 2 (predicate pushed down):")
    print(format_sql(sql2))
    
    # Parse and build algebraic representations
    parser = SQLParser()
    parsed1 = parser.parse(sql1)
    parsed2 = parser.parse(sql2)
    
    alg1 = AlgebraicExpression(parsed1)
    alg2 = AlgebraicExpression(parsed2)
    alg1.build()
    alg2.build()
    
    print("\nAlgebraic representation 1:")
    print(alg1.to_string())
    print("\nAlgebraic representation 2:")
    print(alg2.to_string())
    
    # Check equivalence
    result = analyzer.analyze(sql1, sql2)
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    
    return result

def example_projection_elimination():
    """Example: Redundant projection elimination."""
    print("\n=== Projection Elimination Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Query with redundant projections
    sql1 = """
    SELECT name, age
    FROM (
        SELECT name, age, email
        FROM (
            SELECT *
            FROM users
        ) t1
    ) t2
    """
    
    # Simplified query
    sql2 = """
    SELECT name, age
    FROM users
    """
    
    print("Query 1 (with redundant projections):")
    print(format_sql(sql1))
    print("\nQuery 2 (simplified):")
    print(format_sql(sql2))
    
    # Check equivalence
    result = analyzer.analyze(sql1, sql2)
    
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    
    # Show complexity scores
    parser = SQLParser()
    complexity1 = parser.parse(sql1).metadata.complexity_score
    complexity2 = parser.parse(sql2).metadata.complexity_score
    
    print(f"\nComplexity scores:")
    print(f"  Query 1: {complexity1:.2f}")
    print(f"  Query 2: {complexity2:.2f}")
    
    return result

def example_set_operations():
    """Example: Set operations (UNION, INTERSECT, EXCEPT)."""
    print("\n=== Set Operations Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Example 1: Union commutativity
    print("Example 1: Union Commutativity")
    
    sql1 = """
    SELECT name FROM employees
    UNION
    SELECT name FROM contractors
    """
    
    sql2 = """
    SELECT name FROM contractors
    UNION
    SELECT name FROM employees
    """
    
    print("Query 1:")
    print(format_sql(sql1))
    print("\nQuery 2:")
    print(format_sql(sql2))
    
    result = analyzer.analyze(sql1, sql2)
    print(f"\nEquivalent: {result.is_equivalent}")
    
    # Example 2: Distributivity
    print("\n\nExample 2: Distributivity of Selection over Union")
    
    sql3 = """
    SELECT * FROM (
        SELECT * FROM employees
        UNION ALL
        SELECT * FROM contractors
    ) t
    WHERE age > 30
    """
    
    sql4 = """
    SELECT * FROM employees WHERE age > 30
    UNION ALL
    SELECT * FROM contractors WHERE age > 30
    """
    
    print("Query 3:")
    print(format_sql(sql3))
    print("\nQuery 4:")
    print(format_sql(sql4))
    
    result2 = analyzer.analyze(sql3, sql4)
    print(f"\nEquivalent: {result2.is_equivalent}")
    
    return result, result2

def example_aggregate_queries():
    """Example: Queries with aggregations."""
    print("\n=== Aggregate Queries Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Example 1: Simple aggregation
    sql1 = """
    SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
    FROM employees
    WHERE status = 'active'
    GROUP BY department
    HAVING COUNT(*) > 5
    """
    
    sql2 = """
    SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
    FROM employees
    WHERE status = 'active'
    GROUP BY department
    HAVING emp_count > 5
    """
    
    print("Query 1 (HAVING with COUNT(*)):")
    print(format_sql(sql1))
    print("\nQuery 2 (HAVING with alias):")
    print(format_sql(sql2))
    
    result = analyzer.analyze(sql1, sql2)
    print(f"\nEquivalent: {result.is_equivalent}")
    
    # Example 2: Aggregation with joins
    print("\n\nExample 2: Aggregation with Joins")
    
    sql3 = """
    SELECT d.name, COUNT(e.employee_id) as emp_count
    FROM departments d
    LEFT JOIN employees e ON d.dept_id = e.dept_id
    GROUP BY d.dept_id, d.name
    """
    
    sql4 = """
    SELECT d.name, 
           (SELECT COUNT(*) FROM employees e WHERE e.dept_id = d.dept_id) as emp_count
    FROM departments d
    """
    
    print("Query 3 (GROUP BY with LEFT JOIN):")
    print(format_sql(sql3))
    print("\nQuery 4 (Correlated subquery):")
    print(format_sql(sql4))
    
    result2 = analyzer.analyze(sql3, sql4)
    print(f"\nEquivalent: {result2.is_equivalent}")
    print(f"Confidence: {result2.confidence:.2%}")
    
    return result, result2

def run_all_basic_examples():
    """Run all basic examples."""
    print("=" * 60)
    print("SQL EQUIVALENCE ANALYSIS - BASIC EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_simple_equivalence()
    example_join_reordering()
    example_predicate_pushdown()
    example_projection_elimination()
    # example_set_operations()
    example_aggregate_queries()
    
    print("\n" + "=" * 60)
    print("All basic examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_basic_examples()
