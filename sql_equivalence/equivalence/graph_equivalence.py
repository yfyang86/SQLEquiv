"""Graph-based equivalence checking for SQL queries."""

import logging
import time
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

from ..representations.graph.query_graph import QueryGraph
from .base import EquivalenceChecker, EquivalenceResult, EquivalenceType

logger = logging.getLogger(__name__)


class GraphEquivalenceChecker(EquivalenceChecker):
    """Check equivalence using graph isomorphism and structural similarity.

    The checker runs a cascade:

    1. *Quick reject* on node/edge counts and degree sequences.
    2. *Isomorphism* via :func:`networkx.is_isomorphic`, optionally using
       node- and edge-attribute matchers so that e.g. a SELECT cannot match a
       WHERE even if the structure lines up.
    3. *Structural similarity* — a weighted combination of node-type overlap,
       edge-type overlap, degree-sequence overlap and Laplacian spectrum
       distance. The same metric feeds :attr:`EquivalenceResult.confidence`
       when isomorphism fails.
    """

    SIMILARITY_WEIGHTS: Dict[str, float] = {
        'node': 0.3,
        'edge': 0.3,
        'structural': 0.2,
        'spectral': 0.2,
    }

    def __init__(
        self,
        use_node_attributes: bool = True,
        use_edge_attributes: bool = True,
        ged_timeout: float = 2.0,
        similarity_threshold: float = 0.9,
    ):
        super().__init__()
        self.use_node_attributes = use_node_attributes
        self.use_edge_attributes = use_edge_attributes
        self.ged_timeout = ged_timeout
        self.similarity_threshold = similarity_threshold

    # ---------------------------------------------------------------- matchers
    @staticmethod
    def _node_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return a.get('type') == b.get('type') and a.get('value') == b.get('value')

    @staticmethod
    def _edge_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return a.get('type') == b.get('type')

    # --------------------------------------------------------------- entry pt
    def check_equivalence(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> EquivalenceResult:
        start_time = time.time()
        result = EquivalenceResult(
            is_equivalent=False,
            equivalence_type=EquivalenceType.NOT_EQUIVALENT,
            confidence=0.0,
        )

        try:
            if self._quick_reject(graph1, graph2):
                result.details['reason'] = 'Failed quick rejection tests'
                # Still compute a similarity score so callers see *how* far apart.
                result.confidence = self.compute_similarity(graph1, graph2)
                return result

            if self._check_isomorphism(graph1, graph2):
                result.is_equivalent = True
                result.equivalence_type = EquivalenceType.EXACT
                result.confidence = 1.0
                result.add_proof_step("Graphs are isomorphic (attribute-aware)")
                return result

            similarity = self.compute_similarity(graph1, graph2)
            result.confidence = similarity
            result.details['similarity_score'] = similarity

            if similarity >= self.similarity_threshold:
                result.is_equivalent = True
                result.equivalence_type = EquivalenceType.APPROXIMATE
                result.add_proof_step(
                    f"Structural similarity {similarity:.3f} ≥ threshold "
                    f"{self.similarity_threshold:.3f}"
                )

        except Exception as exc:  # pragma: no cover - logged for diagnostics
            logger.exception("Error checking graph equivalence")
            result.details['error'] = str(exc)
        finally:
            result.execution_time = time.time() - start_time

        return result

    # ------------------------------------------------------------- similarity
    def compute_similarity(self, graph1: QueryGraph, graph2: QueryGraph) -> float:
        components = {
            'node': self._compute_node_similarity(graph1, graph2),
            'edge': self._compute_edge_similarity(graph1, graph2),
            'structural': self._compute_structural_similarity(graph1, graph2),
            'spectral': self._compute_spectral_similarity(graph1, graph2),
        }
        return sum(components[k] * self.SIMILARITY_WEIGHTS[k] for k in components)

    # --------------------------------------------------------------- internals
    def _quick_reject(self, graph1: QueryGraph, graph2: QueryGraph) -> bool:
        if graph1.graph.number_of_nodes() != graph2.graph.number_of_nodes():
            return True
        if graph1.graph.number_of_edges() != graph2.graph.number_of_edges():
            return True
        deg1 = sorted(d for _, d in graph1.graph.degree())
        deg2 = sorted(d for _, d in graph2.graph.degree())
        if deg1 != deg2:
            return True
        return False

    def _check_isomorphism(self, graph1: QueryGraph, graph2: QueryGraph) -> bool:
        node_match = self._node_match if self.use_node_attributes else None
        edge_match = self._edge_match if self.use_edge_attributes else None
        try:
            return nx.is_isomorphic(
                graph1.graph,
                graph2.graph,
                node_match=node_match,
                edge_match=edge_match,
            )
        except nx.NetworkXError as err:
            logger.debug("Isomorphism check failed: %s", err)
            return False

    def _compute_node_similarity(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> float:
        return self._jaccard_on_histogram(
            graph1.node_type_histogram(), graph2.node_type_histogram()
        )

    def _compute_edge_similarity(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> float:
        hist1 = _histogram(a.get('type') for a in graph1.edge_attributes.values())
        hist2 = _histogram(a.get('type') for a in graph2.edge_attributes.values())
        return self._jaccard_on_histogram(hist1, hist2)

    def _compute_structural_similarity(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> float:
        deg1 = sorted(d for _, d in graph1.graph.degree())
        deg2 = sorted(d for _, d in graph2.graph.degree())
        if not deg1 and not deg2:
            return 1.0
        max_len = max(len(deg1), len(deg2))
        padded1 = np.array(deg1 + [0] * (max_len - len(deg1)), dtype=float)
        padded2 = np.array(deg2 + [0] * (max_len - len(deg2)), dtype=float)
        denom = np.linalg.norm(padded1) + np.linalg.norm(padded2)
        if denom == 0:
            return 1.0
        distance = np.linalg.norm(padded1 - padded2) / denom
        return float(max(0.0, 1.0 - distance))

    def _compute_spectral_similarity(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> float:
        try:
            undirected1 = graph1.graph.to_undirected()
            undirected2 = graph2.graph.to_undirected()
            if undirected1.number_of_nodes() == 0 and undirected2.number_of_nodes() == 0:
                return 1.0
            if undirected1.number_of_nodes() == 0 or undirected2.number_of_nodes() == 0:
                return 0.0
            spec1 = np.sort(nx.laplacian_spectrum(undirected1))
            spec2 = np.sort(nx.laplacian_spectrum(undirected2))
            max_len = max(len(spec1), len(spec2))
            spec1 = np.pad(spec1, (0, max_len - len(spec1)))
            spec2 = np.pad(spec2, (0, max_len - len(spec2)))
            denom = np.linalg.norm(spec1 + spec2)
            if denom == 0:
                return 1.0
            return float(max(0.0, 1.0 - np.linalg.norm(spec1 - spec2) / denom))
        except (nx.NetworkXError, ValueError) as err:
            logger.debug("Spectral similarity failed: %s", err)
            return 0.0

    @staticmethod
    def _jaccard_on_histogram(
        hist1: Dict[str, int], hist2: Dict[str, int]
    ) -> float:
        keys = set(hist1) | set(hist2)
        if not keys:
            return 1.0
        intersection = sum(min(hist1.get(k, 0), hist2.get(k, 0)) for k in keys)
        union = sum(max(hist1.get(k, 0), hist2.get(k, 0)) for k in keys)
        return intersection / union if union else 1.0

    # ------------------------------------------------------------- utilities
    def check_subgraph_isomorphism(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> bool:
        """Return True if ``graph2`` contains a subgraph isomorphic to ``graph1``."""
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            graph2.graph,
            graph1.graph,
            node_match=self._node_match if self.use_node_attributes else None,
            edge_match=self._edge_match if self.use_edge_attributes else None,
        )
        return matcher.subgraph_is_isomorphic()

    def compute_graph_edit_distance(
        self, graph1: QueryGraph, graph2: QueryGraph
    ) -> float:
        """Return an upper-bound graph edit distance.

        Uses :func:`networkx.optimize_graph_edit_distance` with a wall-clock
        timeout to keep this bounded, since general GED is NP-hard.
        """
        deadline = time.time() + self.ged_timeout if self.ged_timeout else None
        try:
            best: Optional[float] = None
            for value in nx.optimize_graph_edit_distance(
                graph1.graph,
                graph2.graph,
                node_match=self._node_match if self.use_node_attributes else None,
                edge_match=self._edge_match if self.use_edge_attributes else None,
            ):
                best = value
                if deadline is not None and time.time() > deadline:
                    break
            return float(best) if best is not None else float('inf')
        except nx.NetworkXError as err:
            logger.debug("GED computation failed: %s", err)
            return float('inf')


def _histogram(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        key = v if v is not None else 'unknown'
        out[key] = out.get(key, 0) + 1
    return out
