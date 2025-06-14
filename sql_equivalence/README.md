# README.md
# SQL Equivalence Analysis Library

A comprehensive Python library for analyzing SQL query equivalence using algebraic, graph-based, and embedding-based approaches.

## Features

- **Multiple Analysis Methods**:
  - Algebraic equivalence checking using relational algebra
  - Graph-based equivalence using query graphs and LQT
  - Embedding-based similarity using modern ML techniques

- **Comprehensive SQL Support**:
  - Complex queries with subqueries and CTEs
  - Join operations (INNER, LEFT, RIGHT, FULL)
  - Set operations (UNION, INTERSECT, EXCEPT)
  - Aggregate and window functions
  - Various scalar functions

- **Extensible Architecture**:
  - Easy to add new operators and functions
  - Pluggable ML models for embeddings
  - Customizable equivalence rules

## Installation

```bash
pip install sql-equivalence