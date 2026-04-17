# SQLEquiv technical report

This directory contains a reproducible Quarto manuscript for the
SQLEquiv library.

## Layout

```
manuscript/
├── _quarto.yml      # Project settings (freeze: auto, single render target)
├── main.qmd         # Manuscript source (executable, cites references.bib)
├── references.bib   # BibTeX citations
└── README.md        # This file
```

## Rendering

```bash
# One-time deps (Quarto + Jupyter + runtime libs)
pip install "sql-equivalence[dev]"   # or: pip install sql-equivalence pandas jupyter
# Install Quarto from https://quarto.org/docs/get-started/ if not already present.

# From the repository root:
quarto render manuscript/main.qmd --to html
quarto render manuscript/main.qmd --to pdf   # requires a LaTeX distribution
```

Every numeric value, table, and figure in the manuscript is produced
by a live Python block at render time. Swapping in a larger corpus is
a one-line edit in `sql_equivalence/tests/fixtures.py`.

## Notes on reproducibility

- The manuscript uses `execute.freeze: auto` so Quarto caches unchanged
  blocks between renders. Force a full re-run with
  `quarto render --cache-refresh manuscript/main.qmd`.
- The latency micro-benchmark in @sec-examples reports machine-local
  numbers; trends across revisions are more informative than absolute
  values.
- Citations follow the `bibliography:` key in the YAML header; extend
  `references.bib` in BibTeX syntax.
