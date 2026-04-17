# SQLEquiv — Future Plan

_Last updated: 2026-04-17 (author: code-review-refactor branch)_

This document records the current health of the library, the roadmap for
reaching a first "useful" release, and — just as importantly — the things we
have decided **not** to pursue. It is intended to be kept in sync with
`Readme.md` and revised whenever a phase closes or priorities shift.

## 1. Snapshot of the current code

After the refactor on branch `claude/code-review-refactor-nA3rB` the package
imports cleanly without the optional visualization stack, all bare `except:`
clauses are gone, Python 3.8 typing compatibility is restored, and the public
analyzer API is DRY. However, a meaningful amount of the behavior advertised
in the README is still stubbed out. A quick end-to-end check on the identical
query `SELECT id FROM t WHERE id > 5` vs. itself illustrates the gap:

| Method     | Status             | Notes                                                 |
|------------|--------------------|-------------------------------------------------------|
| algebraic  | Works              | Returns `is_equivalent=True, confidence=1.0`          |
| graph      | Stubbed            | `QueryGraph.build()` is a no-op; similarity constants |
| embedding  | Not meaningful     | Graph is empty → pooled embedding is the zero vector  |

Concrete stub hot-spots (confirmed by grep):

- `representations/graph/query_graph.py::build` — empty; no AST-to-graph logic.
- `representations/graph/lqt.py::build` — empty; `visualize` raises.
- `equivalence/graph_equivalence.py` — 9 `# Framework implementation`
  placeholders (`_check_isomorphism`, `_compute_*_similarity`,
  `compute_graph_edit_distance`, `check_subgraph_isomorphism`).
- `transformations/optimization_rules.py` — 6 "simplified / placeholder"
  sections across `IndexSuggestion`, `JoinOrderOptimization`,
  `PredicatePushdown`, `PartitionPruning`.
- `models/*` — intentional placeholders, no ML backends wired.
- `tests/*` — only `test_parser.py` has coverage (4 smoke tests).

These are not bugs in the refactor; they are the genuine work still required
to deliver on the README's three analysis methods.

## 2. Guiding principles

1. **Algebraic first.** The algebraic pipeline is the most finished and the
   most defensible academically. Stabilize it before the other two.
2. **Lean dependencies.** Keep core `import sql_equivalence` free of heavy or
   optional packages. Push `graphviz`, `plotly`, and any future ML/GPU stack
   behind `extras_require`.
3. **One dialect at a time.** Commit to PostgreSQL as the reference dialect
   for v0.x. Other dialects ride on sqlglot's transpilation support — we do
   not implement dialect-specific logic ourselves.
4. **No silent stubs.** Any function that does not yet do the thing its name
   claims must either raise `NotImplementedError` or be absent. Returning
   `0.5` by default has cost us real debugging time already.
5. **Readable > clever.** The refactor removed duplication; future work must
   keep the bar high (type hints, docstrings explaining *why*, no dead code).

## 3. Phased roadmap

### Phase A — Make the three methods honest (target: v0.2)

Goal: every method either produces a meaningful answer or raises. No more
constant-valued similarities.

| # | Task                                                                 | Where                                               | Effort |
|---|----------------------------------------------------------------------|-----------------------------------------------------|--------|
| 1 | Implement `QueryGraph.build()` from the AST (nodes per clause/table/predicate; edges for data flow and predicate attachment). | `representations/graph/query_graph.py`              | M      |
| 2 | Replace `GraphEquivalenceChecker` constants with real NetworkX calls: `is_isomorphic`, `graph_edit_distance` (with timeout), node/edge-attr aware matchers. | `equivalence/graph_equivalence.py`                  | M      |
| 3 | Implement `LogicalQueryTree.build()` and use it for `graph_edit_distance` in the tree-shaped case (faster than general graph GED). | `representations/graph/lqt.py`                      | S      |
| 4 | Teach `GraphEmbedding` to actually walk the graph built in (1) and pool per-node features; keep hash-based vectors as the default encoder but add a `Node2VecEncoder` behind the `models[graph]` extra. | `representations/embedding/*`                       | M      |
| 5 | Port the README badge items marked `[p]` (partial) from "partial" to "supported" with unit tests covering each. | `tests/test_*.py`                                   | M      |

Exit criteria for Phase A: identical queries produce `is_equivalent=True`
from all three methods; the example pairs in `examples/basic_examples.py`
give stable, non-degenerate numbers; `pytest` runs ≥ 30 tests green.

### Phase B — Test & benchmarking harness (target: v0.3)

| # | Task                                                                 | Effort |
|---|----------------------------------------------------------------------|--------|
| 1 | Curate a fixture corpus of SQL pairs: identical, renamed-aliases, commuted joins, equivalent-via-predicate-simplification, non-equivalent. | M      |
| 2 | Golden-file tests for the algebraic canonical form to pin down behavior before we extend rules. | S      |
| 3 | Coverage target ≥ 70% on `parser/`, `representations/algebraic/`, `equivalence/algebraic_equivalence.py`. | M      |
| 4 | CI: GitHub Actions matrix for Python 3.8 / 3.10 / 3.12. Add `ruff`, `mypy --strict` gates. | S      |
| 5 | A tiny benchmark script (`scripts/bench.py`) that records analyze latency for a fixed corpus, to catch regressions. | S      |

### Phase C — Extensibility & publication (target: v0.4 / first PyPI release)

| # | Task                                                                 | Effort |
|---|----------------------------------------------------------------------|--------|
| 1 | Plugin hook for custom `EquivalenceChecker`s registered via entry points. | S      |
| 2 | Optional `models/` backend: a single learned embedding (e.g., distilled CodeBERT or a small GNN) behind `pip install sql-equivalence[ml]`. | L      |
| 3 | Polish docs: pdoc-generated API reference, plus a tutorial notebook that replaces `demo.ipynb`. | S      |
| 4 | Release `0.4.0` to PyPI with a pinned `sqlglot` range and a deprecation policy. | S      |

### Phase D — Stretch goals (unplanned, ideas only)

- Proof export for algebraic equivalences as a human-readable transcript.
- Streamlit demo app.
- Hugging Face dataset card for the fixture corpus.

## 4. Go / No-go list

### 4.1 Go

- **Postgres dialect as reference.** We already rely on sqlglot; everything
  else is translation.
- **NetworkX for graphs.** Mature, covers isomorphism and GED out of the
  box, already a dependency.
- **sqlglot's AST as the single source of truth.** Our `ast_builder`
  wraps it; we should not build a second AST.
- **Phase A end-to-end correctness before any ML work.** Meaningful baselines
  are a prerequisite to having anything to compare a learned model against.
- **Lazy imports for heavy deps.** Continue the pattern we just introduced
  for `graphviz` and `plotly`; apply it to future `torch` / `transformers`
  dependencies too.

### 4.2 No-go (and why)

- **Writing our own SQL parser.** sqlglot is battle-tested and handles
  dozens of dialects; reinventing it would be 10k+ lines of thankless work.
- **Executing queries to prove equivalence by sampling.** That is a separate
  product category (query-diffing tools like `sqlfuzz` / `PQS`). We stay in
  the static-analysis lane.
- **Supporting DDL/DML equivalence.** Our target is `SELECT`. Statement
  types beyond that (CREATE, INSERT, MERGE, stored procs) are excluded.
- **A "universal" dialect layer.** Other dialects are supported only to the
  extent sqlglot transpiles them to our reference dialect. We explicitly do
  not own dialect-specific corner cases.
- **GPU / large-model embeddings in the default install.** Any learned
  model must live behind an optional extra and be load-time-optional.
- **Rebuilding the visualization stack.** `matplotlib` + optional
  `graphviz` / `plotly` is enough. We will not add a third library.
- **A custom cost-based optimizer.** The `optimization_rules.py` module
  hints at this direction, but a real cost model requires statistics we do
  not have. Keep the module as a thin rule set or delete it in Phase A.
- **"Agentic" LLM-based equivalence checks.** Out of scope; the value of
  this library is in the deterministic, explainable pipeline.
- **Type checker with `--strict` from day one.** Enabled in Phase B;
  enabling earlier would balloon the work without a clear payoff.

## 5. Open questions

1. **Are `operators/` and `representations/algebraic/operators.py` meant to
   converge?** Today they are two parallel operator hierarchies; before
   Phase A we must pick one as canonical (recommend the representations/
   one, since the algebraic checker already depends on it).
2. **What is the policy when the three methods disagree?** The current
   `analyze()` requires unanimity for `is_equivalent=True`. Once graph/
   embedding are honest, we may want a weighted or method-scoped policy
   (e.g., algebraic `True` short-circuits, embedding only contributes a
   similarity score not a vote).
3. **Who owns the ML plug-in story?** If no one has bandwidth for Phase C
   item 2, ship `0.4.0` with `models/` still empty and document it.
4. **Do we want a CLI?** `python -m sql_equivalence compare a.sql b.sql`
   would be easy, but it is only worth doing once Phase A lands.

## 6. How to keep this document honest

- Every PR that ticks a roadmap box should update this file in the same
  commit. Roadmap drift is the default failure mode of living documents.
- Phase exit criteria live here, not in PR descriptions.
- When a "No-go" item comes up again in review, reviewers should link to
  this file instead of relitigating it.
