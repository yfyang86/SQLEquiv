#!/usr/bin/env python
"""Tiny latency benchmark for :class:`SQLEquivalenceAnalyzer`.

Runs the analyzer across a fixed corpus of query pairs and prints min / mean
/ p95 latencies, both end-to-end and per method. Intended to be invoked from
CI (nightly) and in local development to catch regressions. The absolute
numbers are not meaningful across machines; trend across revisions is.

Example:

    python scripts/bench.py
    python scripts/bench.py --methods algebraic --runs 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

# Allow running the script directly from a git checkout without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_equivalence import SQLEquivalenceAnalyzer  # noqa: E402
from sql_equivalence.tests.fixtures import (  # noqa: E402
    EQUIVALENT_PAIRS,
    NON_EQUIVALENT_PAIRS,
)

CORPUS: List[Tuple[str, str]] = list(EQUIVALENT_PAIRS) + list(NON_EQUIVALENT_PAIRS)


def _percentile(samples: Sequence[float], pct: float) -> float:
    if not samples:
        return float('nan')
    ordered = sorted(samples)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


def run(methods: Sequence[str], runs: int) -> None:
    analyzer = SQLEquivalenceAnalyzer(enable_caching=False)
    latencies: List[float] = []
    for _ in range(runs):
        for sql1, sql2 in CORPUS:
            start = time.perf_counter()
            analyzer.analyze(sql1, sql2, methods=list(methods))
            latencies.append(time.perf_counter() - start)

    if not latencies:
        print("No samples collected.")
        return

    print(f"runs={runs}, pairs={len(CORPUS)}, samples={len(latencies)}")
    print(f"  min    : {min(latencies) * 1000:8.2f} ms")
    print(f"  mean   : {statistics.mean(latencies) * 1000:8.2f} ms")
    print(f"  median : {statistics.median(latencies) * 1000:8.2f} ms")
    print(f"  p95    : {_percentile(latencies, 95) * 1000:8.2f} ms")
    print(f"  max    : {max(latencies) * 1000:8.2f} ms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--methods',
        nargs='+',
        default=['algebraic', 'graph', 'embedding'],
        help="Methods to benchmark (default: all three).",
    )
    p.add_argument('--runs', type=int, default=10, help="Iterations over the corpus.")
    args = p.parse_args()
    run(args.methods, args.runs)


if __name__ == '__main__':
    main()
