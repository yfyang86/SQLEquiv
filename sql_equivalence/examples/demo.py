# sql_equivalence/examples/demo.py
"""Interactive demo of SQL equivalence analysis."""

def interactive_demo():
    """Run an interactive demo."""
    from sql_equivalence import SQLEquivalenceAnalyzer
    from sql_equivalence.utils import format_sql
    
    print("""
    =====================================
    SQL Equivalence Analysis Demo
    =====================================
    
    This demo will analyze some common SQL query patterns
    and show you whether they are equivalent.
    """)
    
    analyzer = SQLEquivalenceAnalyzer()
    
    demos = [
        {
            'name': 'Simple Filter Equivalence',
            'sql1': 'SELECT * FROM products WHERE price > 100',
            'sql2': 'SELECT * FROM products WHERE price >= 101',
            'expected': True
        },
        {
            'name': 'Column Selection',
            'sql1': 'SELECT name, price FROM products',
            'sql2': 'SELECT price, name FROM products',
            'expected': True  # Same columns, different order
        },
        {
            'name': 'Join Commutativity',
            'sql1': 'SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id',
            'sql2': 'SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id',
            'expected': True
        },
        {
            'name': 'Different Filters',
            'sql1': 'SELECT * FROM users WHERE age > 20',
            'sql2': 'SELECT * FROM users WHERE age < 20',
            'expected': False
        },
        ##UNION is not implemented## {
        ##UNION is not implemented##     'name': 'Union Order',
        ##UNION is not implemented##     'sql1': 'SELECT id FROM table1 UNION SELECT id FROM table2',
        ##UNION is not implemented##     'sql2': 'SELECT id FROM table2 UNION SELECT id FROM table1',
        ##UNION is not implemented##     'expected': True
        ##UNION is not implemented## }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\n{'-'*50}")
        print(f"Demo {i}: {demo['name']}")
        print(f"{'-'*50}")
        
        print("\nQuery 1:")
        print(format_sql(demo['sql1']))
        print("\nQuery 2:")
        print(format_sql(demo['sql2']))
        
        print("\nAnalyzing...")
        result = analyzer.analyze(demo['sql1'], demo['sql2'])
        
        print(f"\nResult:")
        print(f"  Equivalent: {result.is_equivalent}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Expected: {demo['expected']}")
        print(f"  Match: {'✓' if result.is_equivalent == demo['expected'] else '✗'}")
        
        input("\nPress Enter to continue...")
    
    print(f"\n{'='*50}")
    print("Demo completed!")
    print(f"{'='*50}")

def main():
    """Main entry point for examples."""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == 'basic':
            from .basic_examples import run_all_basic_examples
            run_all_basic_examples()
        elif example == 'advanced':
            from .advanced_examples import run_all_advanced_examples
            run_all_advanced_examples()
        elif example == 'demo':
            interactive_demo()
        elif example == 'quickstart':
            from .quick_start import quick_start_guide
            quick_start_guide()
        else:
            print(f"Unknown example: {example}")
            print("Available: basic, advanced, demo, quickstart")
    else:
        print("SQL Equivalence Analysis Examples")
        print("=================================")
        print("\nUsage: python -m sql_equivalence.examples [example]")
        print("\nAvailable examples:")
        print("  basic      - Run basic examples")
        print("  advanced   - Run advanced examples")
        print("  demo       - Run interactive demo")
        print("  quickstart - Show quick start guide")
        
        print("\nOr import specific examples:")
        print("  from sql_equivalence.examples import example_simple_equivalence")
        print("  example_simple_equivalence()")

if __name__ == "__main__":
    main()