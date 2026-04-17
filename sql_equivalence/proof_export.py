"""Export a human-readable transcript of an equivalence analysis.

Given an :class:`~sql_equivalence.analyzer.AnalysisResult` this module can
render a compact Markdown or JSON "proof" of *why* the analyzer returned the
verdict it did: the per-method votes, the proof steps recorded by the
algebraic checker, the canonical forms of the two inputs, and any graph
edit-distance or similarity scores that were computed.

The output is intended for humans (reviewers, unit-test authors) who need to
understand a single analysis in isolation, not for programmatic consumption
-- use :meth:`AnalysisResult.to_dict` for that.
"""

from __future__ import annotations

import json
from typing import Any

from .analyzer import AnalysisResult


def to_markdown(result: AnalysisResult) -> str:
    """Render ``result`` as a Markdown document."""
    verdict = "EQUIVALENT" if result.is_equivalent else "NOT EQUIVALENT"
    lines: list[str] = [
        "# SQL equivalence analysis",
        "",
        f"**Verdict**: {verdict} (confidence {result.confidence:.3f})",
        f"**Execution time**: {result.execution_time * 1000:.2f} ms",
        "",
        "## Queries",
        "```sql",
        f"-- Q1\n{result.sql1}",
        "```",
        "```sql",
        f"-- Q2\n{result.sql2}",
        "```",
        "",
        "## Per-method results",
    ]

    for method, outcome in result.method_results.items():
        symbol = "✅" if outcome.get('is_equivalent') else "❌"
        conf = outcome.get('confidence', 0.0)
        equiv_type = outcome.get('equivalence_type', 'n/a')
        lines.append(
            f"- {symbol} **{method}** — type={equiv_type}, confidence={conf:.3f}"
        )
        for step in outcome.get('proof_steps', []) or []:
            lines.append(f"  - {step}")
        for key in ('canonical_form1', 'canonical_form2',
                    'graph_edit_distance', 'similarity_score'):
            if key in outcome:
                lines.append(f"  - `{key}`: {_truncate(outcome[key])}")

    lines.append("")
    return "\n".join(lines)


def to_json(result: AnalysisResult, indent: int = 2) -> str:
    """Serialize ``result`` as a JSON document.

    Values that are not JSON-serializable (e.g. NumPy scalars) are coerced to
    their ``repr`` so the document always round-trips to text.
    """
    return json.dumps(result.to_dict(), indent=indent, default=repr)


def _truncate(value: Any, limit: int = 120) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
