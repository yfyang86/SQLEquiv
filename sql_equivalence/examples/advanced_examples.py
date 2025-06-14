# sql_equivalence/examples/advanced_examples.py
"""Advanced examples of SQL equivalence analysis."""

import os
from typing import List, Tuple
from sql_equivalence import SQLEquivalenceAnalyzer
from sql_equivalence.parser import SQLParser
from sql_equivalence.representations.algebraic import AlgebraicExpression
from sql_equivalence.representations.graph import QueryGraph
from sql_equivalence.utils import (
    visualize_query_graph, 
    visualize_algebraic_expression,
    create_query_comparison_plot
)
from ..utils.sql_utils import format_sql

def example_complex_joins():
    """Example: Complex multi-way joins with different patterns."""
    print("=== Complex Joins Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Star schema join
    sql1 = """
    SELECT 
        f.sale_id,
        d.date,
        p.product_name,
        c.customer_name,
        s.store_name,
        f.amount
    FROM fact_sales f
    JOIN dim_date d ON f.date_id = d.date_id
    JOIN dim_product p ON f.product_id = p.product_id
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_store s ON f.store_id = s.store_id
    WHERE d.year = 2023 AND p.category = 'Electronics'
    """
    
    # Same query with different join order
    sql2 = """
    SELECT 
        f.sale_id,
        d.date,
        p.product_name,
        c.customer_name,
        s.store_name,
        f.amount
    FROM dim_date d
    JOIN fact_sales f ON d.date_id = f.date_id
    JOIN dim_product p ON f.product_id = p.product_id
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_store s ON f.store_id = s.store_id
    WHERE d.year = 2023 AND p.category = 'Electronics'
    """
    
    print("Star Schema Query 1:")
    print(format_sql(sql1))
    print("\nStar Schema Query 2 (different join order):")
    print(format_sql(sql2))
    
    # Analyze with detailed results
    result = analyzer.analyze(sql1, sql2, detailed=True)
    
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    
    # Show detailed analysis
    for method, details in result.method_results.items():
        print(f"\n{method.upper()} Analysis:")
        if 'details' in details:
            for key, value in details['details'].items():
                print(f"  {key}: {value}")
    
    return result

def example_subqueries():
    """Example: Different subquery patterns."""
    print("\n=== Subqueries Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Example 1: IN subquery vs JOIN
    print("Example 1: IN Subquery vs JOIN")
    
    sql1 = """
    SELECT e.name, e.salary
    FROM employees e
    WHERE e.dept_id IN (
        SELECT d.dept_id
        FROM departments d
        WHERE d.location = 'New York'
    )
    """
    
    sql2 = """
    SELECT DISTINCT e.name, e.salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.location = 'New York'
    """
    
    result1 = analyzer.analyze(sql1, sql2)
    print(f"IN vs JOIN - Equivalent: {result1.is_equivalent}")
    
    # Example 2: EXISTS vs IN
    print("\n\nExample 2: EXISTS vs IN")
    
    sql3 = """
    SELECT d.name
    FROM departments d
    WHERE EXISTS (
        SELECT 1
        FROM employees e
        WHERE e.dept_id = d.dept_id
        AND e.salary > 100000
    )
    """
    
    sql4 = """
    SELECT DISTINCT d.name
    FROM departments d
    WHERE d.dept_id IN (
        SELECT e.dept_id
        FROM employees e
        WHERE e.salary > 100000
    )
    """
    
    result2 = analyzer.analyze(sql3, sql4)
    print(f"EXISTS vs IN - Equivalent: {result2.is_equivalent}")
    
    # Example 3: Correlated vs Non-correlated
    print("\n\nExample 3: Correlated vs Non-correlated Subquery")
    
    sql5 = """
    SELECT e.name, e.salary
    FROM employees e
    WHERE e.salary > (
        SELECT AVG(e2.salary)
        FROM employees e2
        WHERE e2.dept_id = e.dept_id
    )
    """
    
    sql6 = """
    SELECT e.name, e.salary
    FROM employees e
    JOIN (
        SELECT dept_id, AVG(salary) as avg_salary
        FROM employees
        GROUP BY dept_id
    ) dept_avg ON e.dept_id = dept_avg.dept_id
    WHERE e.salary > dept_avg.avg_salary
    """
    
    result3 = analyzer.analyze(sql5, sql6)
    print(f"Correlated vs Non-correlated - Equivalent: {result3.is_equivalent}")
    
    return result1, result2, result3

def example_window_functions():
    """Example: Window functions and equivalent queries."""
    print("\n=== Window Functions Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Window function query
    sql1 = """
    SELECT 
        employee_id,
        name,
        department,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
    FROM employees
    WHERE rank <= 3
    """
    
    # Equivalent query without window function (using correlated subquery)
    sql2 = """
    SELECT 
        e1.employee_id,
        e1.name,
        e1.department,
        e1.salary,
        (
            SELECT COUNT(*) + 1
            FROM employees e2
            WHERE e2.department = e1.department
            AND e2.salary > e1.salary
        ) as rank
    FROM employees e1
    WHERE (
        SELECT COUNT(*) + 1
        FROM employees e2
        WHERE e2.department = e1.department
        AND e2.salary > e1.salary
    ) <= 3
    """
    
    print("Query with Window Function:")
    print(format_sql(sql1))
    print("\nEquivalent Query without Window Function:")
    print(format_sql(sql2))
    
    result = analyzer.analyze(sql1, sql2)
    print(f"\nEquivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    
    return result

def example_cte_analysis():
    """Example: Common Table Expressions (CTEs)."""
    print("\n=== CTE Analysis Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Query with CTE
    sql1 = """
    WITH high_salary_employees AS (
        SELECT *
        FROM employees
        WHERE salary > 100000
    ),
    dept_counts AS (
        SELECT 
            dept_id,
            COUNT(*) as emp_count
        FROM high_salary_employees
        GROUP BY dept_id
    )
    SELECT 
        d.name,
        dc.emp_count
    FROM departments d
    JOIN dept_counts dc ON d.dept_id = dc.dept_id
    WHERE dc.emp_count > 5
    """
    
    # Equivalent query without CTE
    sql2 = """
    SELECT 
        d.name,
        subq.emp_count
    FROM departments d
    JOIN (
        SELECT 
            dept_id,
            COUNT(*) as emp_count
        FROM employees
        WHERE salary > 100000
        GROUP BY dept_id
        HAVING COUNT(*) > 5
    ) subq ON d.dept_id = subq.dept_id
    """
    
    print("Query with CTE:")
    print(format_sql(sql1))
    print("\nEquivalent query without CTE:")
    print(format_sql(sql2))
    
    result = analyzer.analyze(sql1, sql2)
    print(f"\nEquivalent: {result.is_equivalent}")
    
    return result

def example_batch_analysis():
    """Example: Batch analysis of multiple query pairs."""
    print("\n=== Batch Analysis Example ===\n")
    
    analyzer = SQLEquivalenceAnalyzer()
    
    # Define multiple query pairs
    query_pairs = [
        # Pair 1: Simple selection
        (
            "SELECT * FROM users WHERE age > 18",
            "SELECT * FROM users WHERE age >= 19"
        ),
        # Pair 2: Join order
        (
            "SELECT * FROM a JOIN b ON a.id = b.id",
            "SELECT * FROM b JOIN a ON b.id = a.id"
        ),
        ###UNION IS NOT IMP### # Pair 3: Union order
        ###UNION IS NOT IMP### (
        ###UNION IS NOT IMP###     "SELECT id FROM table1 UNION SELECT id FROM table2",
        ###UNION IS NOT IMP###     "SELECT id FROM table2 UNION SELECT id FROM table1"
        ###UNION IS NOT IMP### ),
        # Pair 4: Different queries
        (
            "SELECT name FROM users",
            "SELECT email FROM users"
        )
    ]
    
    print("Analyzing", len(query_pairs), "query pairs...\n")
    
    # Batch analyze
    results = analyzer.batch_analyze(query_pairs, n_jobs=2)
    
    # Display results
    for i, (result, (sql1, sql2)) in enumerate(zip(results, query_pairs)):
        print(f"Pair {i+1}:")
        print(f"  Query 1: {sql1}")
        print(f"  Query 2: {sql2}")
        print(f"  Equivalent: {result.is_equivalent}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Time: {result.execution_time:.3f}s")
        print()
    
    # Create comparison plot
    plot_path = "batch_analysis_results.html"
    create_query_comparison_plot(
        [r.to_dict() for r in results],
        output_path=plot_path
    )
    print(f"Comparison plot saved to: {plot_path}")
    
    return results

def example_visualization():
    """Example: Visualizing query structures."""
    print("\n=== Visualization Example ===\n")
    
    # Parse a complex query
    sql = """
    SELECT 
        c.customer_name,
        COUNT(o.order_id) as order_count,
        SUM(o.total_amount) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE c.registration_date >= '2023-01-01'
    GROUP BY c.customer_id, c.customer_name
    HAVING COUNT(o.order_id) > 0
    ORDER BY total_spent DESC
    """
    
    print("Analyzing query:")
    print(format_sql(sql))
    
    # Parse query
    parser = SQLParser()
    parsed = parser.parse(sql)
    
    # Build representations
    alg_expr = AlgebraicExpression(parsed)
    alg_expr.build()
    
    query_graph = QueryGraph(parsed)
    query_graph.build()
    
    # Create visualizations
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Algebraic expression tree
    alg_path = os.path.join(output_dir, "algebraic_expression")
    alg_viz = visualize_algebraic_expression(alg_expr, alg_path)
    print(f"\nAlgebraic expression saved to: {alg_viz}")
    
    # 2. Query graph
    graph_path = os.path.join(output_dir, "query_graph.png")
    graph_viz = visualize_query_graph(query_graph, graph_path)
    print(f"Query graph saved to: {graph_viz}")
    
    # 3. Interactive graph
    interactive_path = os.path.join(output_dir, "interactive_graph.html")
    interactive_viz = visualize_query_graph(
        query_graph, 
        interactive_path, 
        interactive=True
    )
    print(f"Interactive graph saved to: {interactive_viz}")
    
    return parsed, alg_expr, query_graph

def run_all_advanced_examples():
    """Run all advanced examples."""
    print("=" * 60)
    print("SQL EQUIVALENCE ANALYSIS - ADVANCED EXAMPLES")
    print("=" * 60)
    
    # Import here to avoid circular imports
    from sql_equivalence.utils import format_sql
    
    # Run examples
    example_complex_joins()
    example_subqueries()
    # example_window_functions()
    example_cte_analysis()
    example_batch_analysis()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("All advanced examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_advanced_examples()