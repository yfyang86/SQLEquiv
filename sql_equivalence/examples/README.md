# SQL Equivalence Analysis Examples

This directory contains examples demonstrating various features of the SQL equivalence analysis library.

## Quick Start

```python
from sql_equivalence import SQLEquivalenceAnalyzer

analyzer = SQLEquivalenceAnalyzer()

sql1 = "SELECT * FROM users WHERE age > 18"
sql2 = "SELECT * FROM users WHERE age >= 19"

result = analyzer.analyze(sql1, sql2)
print(f"Equivalent: {result.is_equivalent}")
```
