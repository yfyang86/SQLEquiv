"""Minimal Streamlit demo for SQL equivalence analysis.

Not installed as part of the library -- run it with::

    pip install streamlit
    streamlit run scripts/streamlit_demo.py

The demo accepts two SQL queries, runs the analyzer, and renders the
Markdown proof produced by :mod:`sql_equivalence.proof_export`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    raise SystemExit(
        "streamlit is required for this demo. Install with: pip install streamlit"
    )

from sql_equivalence import SQLEquivalenceAnalyzer  # noqa: E402
from sql_equivalence.proof_export import to_markdown  # noqa: E402

DEFAULT_SQL1 = "SELECT id, name FROM users WHERE age > 18"
DEFAULT_SQL2 = "select id, name from users where age > 18"

st.set_page_config(page_title="SQL Equivalence Analyzer", layout="wide")
st.title("SQL Equivalence Analyzer")
st.caption("Compare two queries via algebraic, graph, and embedding methods.")

col1, col2 = st.columns(2)
with col1:
    sql1 = st.text_area("Query 1", value=DEFAULT_SQL1, height=180)
with col2:
    sql2 = st.text_area("Query 2", value=DEFAULT_SQL2, height=180)

methods = st.multiselect(
    "Methods",
    options=['algebraic', 'graph', 'embedding'],
    default=['algebraic', 'graph', 'embedding'],
)
detailed = st.checkbox("Detailed output", value=True)

if st.button("Analyze", type="primary"):
    analyzer = SQLEquivalenceAnalyzer()
    with st.spinner("Analyzing..."):
        result = analyzer.analyze(sql1, sql2, methods=methods, detailed=detailed)
    st.success(
        f"Verdict: {'EQUIVALENT' if result.is_equivalent else 'NOT EQUIVALENT'} "
        f"(confidence {result.confidence:.3f})"
    )
    st.markdown(to_markdown(result))
