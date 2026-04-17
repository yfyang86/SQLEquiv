"""Shared fixture corpus of SQL pairs.

Each entry in :data:`EQUIVALENT_PAIRS` is a pair of queries the library
*should* recognize as equivalent (exact or semantic). Each entry in
:data:`NON_EQUIVALENT_PAIRS` is a pair the library should flag as distinct.

The corpus is intentionally small and focused on the kinds of rewrites our
algebraic rules target today; extend it as we broaden coverage.
"""

from typing import Tuple

Pair = Tuple[str, str]

#: Queries that are known to be equivalent. Comments describe the rewrite.
EQUIVALENT_PAIRS = [
    # Identical queries -- trivial baseline.
    (
        "SELECT id FROM users",
        "SELECT id FROM users",
    ),
    # Whitespace/casing differences.
    (
        "SELECT   id,   name  FROM  users",
        "select id, name from users",
    ),
    # Column ordering in projection (still selecting the same set).
    # NOTE: algebraic rules today treat these as distinct; kept here as a
    # known-gap for future work.
    # (
    #     "SELECT id, name FROM users",
    #     "SELECT name, id FROM users",
    # ),
]


NON_EQUIVALENT_PAIRS: Tuple[Pair, ...] = (
    (
        "SELECT id FROM users",
        "SELECT id FROM customers",
    ),
    (
        "SELECT id FROM t",
        "SELECT a, b, c FROM users WHERE age > 10 GROUP BY a",
    ),
)


#: Pairs the library *ought* to flag as non-equivalent but currently does not.
#: Tracked as expected failures so regressions are still detected if the
#: behavior flips. Removing an entry from this list requires the
#: corresponding algebraic rule to be tightened.
KNOWN_GAP_NON_EQUIVALENT_PAIRS: Tuple[Pair, ...] = (
    # Different column selected -- algebraic checker currently ignores column
    # list mismatches in projection.
    (
        "SELECT id FROM users",
        "SELECT name FROM users",
    ),
    # Opposite WHERE predicates -- checker currently considers these equal.
    (
        "SELECT id FROM users WHERE age > 10",
        "SELECT id FROM users WHERE age < 10",
    ),
)
