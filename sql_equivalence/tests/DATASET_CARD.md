---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - sql
  - equivalence
  - regression-test
size_categories:
  - n<100
---

# SQLEquiv regression corpus

## Summary

A tiny, hand-curated corpus of SQL query pairs used as regression tests for
the SQLEquiv library. Each pair is labeled as **equivalent** or
**non-equivalent**; a small subset is tracked separately as a
**known-gap** set — pairs the library *ought* to recognise as
non-equivalent but currently does not.

## Supported tasks

- **SQL equivalence classification**: given two queries, predict whether
  they are semantically equivalent.
- **Query rewrite detection**: identify rewritings (whitespace, casing,
  column order, predicate flipping) that preserve or change semantics.

## Source

The corpus lives in `sql_equivalence/tests/fixtures.py`. It is loaded by
`scripts/bench.py` and `sql_equivalence/tests/test_corpus.py`.

## Fields

| field  | type   | description                                  |
|--------|--------|----------------------------------------------|
| `sql1` | string | First SQL query                              |
| `sql2` | string | Second SQL query                             |
| `label`| string | `equivalent`, `non_equivalent`, `known_gap`  |

(The on-disk format is a Python tuple of tuples, not Parquet or CSV. The
dataset card follows the Hugging Face template to make it easy to export
later.)

## Splits

There are no train/validation/test splits — the dataset is exhaustively
used as a regression suite.

| split        | size | notes                                               |
|--------------|-----:|-----------------------------------------------------|
| equivalent   |    2 | Identical SQL, whitespace-only differences          |
| non_equivalent |  2 | Different tables / different column-predicate mix  |
| known_gap    |    2 | Projection-edit and predicate-flip (xfail today)    |

## Intended use

- Unit / regression testing for `sql_equivalence`.
- Baseline for future work on learned SQL equivalence.

## Intended *non*-use

- Not a benchmark for general-purpose SQL understanding. The pair count is
  far too small.
- Not representative of production SQL -- queries are stripped-down
  synthetic examples.

## License

Apache 2.0, following the parent repository.

## Citation

See `Readme.md` at the repository root.
