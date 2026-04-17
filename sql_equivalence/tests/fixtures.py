"""Shared fixture corpus of SQL pairs.

Organized into three tiers:

- :data:`EQUIVALENT_PAIRS` — pairs the library **does** recognise as
  equivalent. Regression failures here mean something actually broke.
- :data:`NON_EQUIVALENT_PAIRS` — pairs the library **does** flag as
  distinct. Regression failures mean the checkers got more permissive.
- :data:`KNOWN_GAP_PAIRS` — pairs the library *should* flag as distinct
  but currently does not. Tracked as xfail(strict=True) so we hear about
  it the day a checker tightens enough to separate them.

Each entry is a ``(sql1, sql2)`` tuple of query strings.
"""

from typing import Tuple

Pair = Tuple[str, str]

# ---------------------------------------------------------------- equivalent
EQUIVALENT_PAIRS: Tuple[Pair, ...] = (
    # --- trivial baselines ----------------------------------------------
    (
        "SELECT id FROM users",
        "SELECT id FROM users",
    ),
    (
        "SELECT   id,   name  FROM  users",
        "select id, name from users",
    ),
    # --- identical complex queries --------------------------------------
    (
        "WITH adults AS (SELECT id, name FROM users WHERE age >= 18) "
        "SELECT id FROM adults",
        "WITH adults AS (SELECT id, name FROM users WHERE age >= 18) "
        "SELECT id FROM adults",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
        "FROM emp",
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
        "FROM emp",
    ),
    (
        "SELECT a.id, b.name FROM a INNER JOIN b ON a.id = b.aid",
        "SELECT a.id, b.name FROM a INNER JOIN b ON a.id = b.aid",
    ),
    (
        "SELECT dept, COUNT(*) AS c FROM emp GROUP BY dept HAVING COUNT(*) > 5",
        "SELECT dept, COUNT(*) AS c FROM emp GROUP BY dept HAVING COUNT(*) > 5",
    ),
)


# ---------------------------------------------------------- non-equivalent
NON_EQUIVALENT_PAIRS: Tuple[Pair, ...] = (
    # --- structurally different ----------------------------------------
    (
        "SELECT id FROM users",
        "SELECT id FROM customers",
    ),
    (
        "SELECT id FROM t",
        "SELECT a, b, c FROM users WHERE age > 10 GROUP BY a",
    ),
    # --- HAVING vs no HAVING -------------------------------------------
    (
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5",
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept",
    ),
    # --- extra JOIN (three-way vs two-way) -----------------------------
    (
        "SELECT a.id FROM a JOIN b ON a.id = b.aid JOIN c ON b.cid = c.id",
        "SELECT a.id FROM a JOIN b ON a.id = b.aid",
    ),
    # --- completely different queries ----------------------------------
    (
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept",
        "SELECT name FROM users WHERE active = TRUE",
    ),
)


# ------------------------------------------------------------- known gaps
#: Pairs the library *ought* to flag as non-equivalent but currently does
#: not. Tracked as expected failures so regressions are still detected if
#: the behavior flips. Removing an entry from this list requires the
#: corresponding checker to be tightened.
KNOWN_GAP_PAIRS: Tuple[Pair, ...] = (
    # Opposite WHERE predicate -- checker treats both as "a predicate exists".
    (
        "SELECT id FROM users WHERE age > 10",
        "SELECT id FROM users WHERE age < 10",
    ),
    # JOIN-type semantics are not compared.
    (
        "SELECT a.id FROM a INNER JOIN b ON a.id = b.aid",
        "SELECT a.id FROM a LEFT JOIN b ON a.id = b.aid",
    ),
    # UNION vs UNION ALL (duplicate-preservation changes result multiplicity).
    (
        "SELECT id FROM a UNION SELECT id FROM b",
        "SELECT id FROM a UNION ALL SELECT id FROM b",
    ),
    # LIMIT value not inspected.
    (
        "SELECT id FROM t LIMIT 10",
        "SELECT id FROM t LIMIT 100",
    ),
    # Different aggregate function.
    (
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept",
        "SELECT dept, SUM(salary) FROM emp GROUP BY dept",
    ),
)


# Back-compat alias (earlier tests referred to this name).
KNOWN_GAP_NON_EQUIVALENT_PAIRS = KNOWN_GAP_PAIRS
