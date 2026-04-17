# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Deep test suite (102 passing tests, 7 xfail-strict known-gap markers):
  per-checker tests for graph / embedding / algebraic plus integration
  tests over complex SQL (CTEs, window functions, set operations,
  subqueries, multi-way joins, aggregation with HAVING). See
  `tests/test_graph.py`, `tests/test_embedding.py`, `tests/test_algebraic.py`,
  `tests/test_complex_queries.py`, `tests/test_plugins.py`.
- `QueryGraph.node_type_histogram()` and `iter_nodes_of_type()` helpers
  used by the similarity metric and the tests.

### Changed
- `GraphEquivalenceChecker._compute_node_similarity` now blends a
  type-only histogram with a `(type, value)` histogram so table names,
  column names, and literal values actually influence similarity. This
  is what closes the "SELECT id FROM users" vs "SELECT id FROM customers"
  gap.

### Fixed
- Parser AST builder compatibility with current sqlglot:
  `Window.partition_by`, `Union.distinct`, `Limit.offset`, and
  `Select.distinct` are all methods or renamed args in current releases.
  The builder now reads them via `args.get(...)`. Also: JOINs migrated
  from `From.joins` to `Select.args['joins']` -- the builder now attaches
  them as FROM children.
- FROM key lookup: sqlglot renamed `'from'` to `'from_'`; the builder
  tries both so old and new versions work.

### Plugin registry (from earlier this release)
- `sql_equivalence.plugins` with
  `register_method` / `unregister_method` / `load_entry_points`. Third-party
  packages can now register custom analysis methods via the
  `sql_equivalence.methods` entry-point group.
- `HashingVectorEncoder` in `sql_equivalence.models.embedding_models` — a
  deterministic zero-extra-dependency embedding baseline usable as
  `SQLEquivalenceAnalyzer(embedding_model=HashingVectorEncoder())`.
- `Future-Plan.md` documenting the roadmap, go/no-go list, and known gaps.
- Fixture corpus at `sql_equivalence/tests/fixtures.py` with equivalent,
  non-equivalent, and known-gap pairs.
- Latency benchmark at `scripts/bench.py`.
- CI pipeline (`.github/workflows/ci.yml`): pytest matrix on Python
  3.8 / 3.10 / 3.12 + ruff lint + informational mypy.

### Changed
- `QueryGraph.build()`, `LogicalQueryTree.build()`, and
  `GraphEquivalenceChecker` are no longer stubs: they walk the parsed AST
  and produce meaningful equivalence / similarity scores.
- `EmbeddingSimilarityChecker` accepts `ParsedQuery` directly; cosine
  similarity is mapped to `[0, 1]` for cleaner thresholding.
- `SQLEquivalenceAnalyzer.analyze(..., detailed=...)` now memoizes per
  `detailed` value.
- `graphviz` and `plotly` are imported lazily so that installing the core
  package does not require the full visualization stack.

### Fixed
- `tuple[...]` Python 3.10+ generics replaced with `Tuple[...]` for 3.8
  compatibility in `operators/relational_operators.py` and
  `parser/sql_parser.py`.
- Bare `except:` clauses narrowed to specific exception types across the
  parser, sql_utils, and graph_utils modules.
- Stray `import time` at the bottom of
  `equivalence/algebraic_equivalence.py` moved to the top.
