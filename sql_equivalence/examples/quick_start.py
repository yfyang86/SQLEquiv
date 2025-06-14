# sql_equivalence/examples/quick_start.py
"""Quick start guide for SQL equivalence analysis."""

def quick_start_guide():
    """Quick start guide with simple examples."""
    print("""
    SQL Equivalence Analysis - Quick Start Guide
    ===========================================
    
    This library helps you determine if two SQL queries are equivalent.
    
    Basic Usage:
    -----------
    """)
    
    # Example 1: Basic usage
    print("1. Basic Equivalence Check:")
    print("""
    from sql_equivalence import SQLEquivalenceAnalyzer
    
    analyzer = SQLEquivalenceAnalyzer()
    
    sql1 = "SELECT * FROM users WHERE age > 18"
    sql2 = "SELECT * FROM users WHERE age >= 19"
    
    result = analyzer.analyze(sql1, sql2)
    print(f"Equivalent: {result.is_equivalent}")
    print(f"Confidence: {result.confidence:.2%}")
    """)
    
    # Example 2: Using specific methods
    print("\n2. Using Specific Analysis Methods:")
    print("""
    # Use only algebraic method for faster analysis
    result = analyzer.analyze(sql1, sql2, methods=['algebraic'])
    
    # Use all methods for comprehensive analysis
    result = analyzer.analyze(sql1, sql2, methods=['algebraic', 'graph', 'embedding'])
    """)
    
    # Example 3: Batch analysis
    print("\n3. Batch Analysis:")
    print("""
    query_pairs = [
        ("SELECT * FROM a", "SELECT * FROM a"),
        ("SELECT id FROM users", "SELECT user_id FROM users"),
        ("SELECT * FROM a JOIN b", "SELECT * FROM b JOIN a")
    ]
    
    results = analyzer.batch_analyze(query_pairs, n_jobs=-1)
    for result in results:
        print(f"Equivalent: {result.is_equivalent}")
    """)
    
    # Example 4: Detailed analysis
    print("\n4. Getting Detailed Results:")
    print("""
    result = analyzer.analyze(sql1, sql2, detailed=True)
    
    # Access method-specific results
    for method, details in result.method_results.items():
        print(f"{method}: {details['is_equivalent']}")
        if 'proof_steps' in details:
            for step in details['proof_steps']:
                print(f"  - {step}")
    """)
    
    # Example 5: Working with representations
    print("\n5. Working with Query Representations:")
    print("""
    from sql_equivalence.parser import SQLParser
    from sql_equivalence.representations.algebraic import AlgebraicExpression
    
    parser = SQLParser()
    parsed = parser.parse("SELECT * FROM users WHERE age > 18")
    
    # Build algebraic representation
    alg_expr = AlgebraicExpression(parsed)
    alg_expr.build()
    print(alg_expr.to_string())
    
    # Visualize
    alg_expr.visualize("query_tree.png")
    """)
    
    print("\n" + "="*50)
    print("For more examples, see the examples directory!")
    print("="*50)

if __name__ == "__main__":
    quick_start_guide()