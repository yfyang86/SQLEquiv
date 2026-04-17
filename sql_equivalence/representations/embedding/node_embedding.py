"""Node embedding utilities."""

from typing import Any, Dict, List

import numpy as np


class NodeEmbedding:
    """Embedding for individual nodes in a query representation.

    The default implementation returns deterministic random vectors seeded by
    a stable hash of each node so that repeated calls yield identical
    embeddings. Subclasses or plugins may override :meth:`embed_node` to
    provide learned embeddings.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, int] = {}
        self.embeddings: Dict[int, np.ndarray] = {}

    @staticmethod
    def _node_key(node: Dict[str, Any]) -> str:
        return str(node.get('type', '')) + ':' + str(node.get('name', ''))

    def build_vocabulary(self, nodes: List[Dict[str, Any]]) -> None:
        """Assign a stable integer id to each distinct node key."""
        for node in nodes:
            key = self._node_key(node)
            if key not in self.vocabulary:
                self.vocabulary[key] = len(self.vocabulary)

    def embed_node(self, node: Dict[str, Any]) -> np.ndarray:
        """Return a deterministic pseudo-random embedding for ``node``."""
        key = self._node_key(node)
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        return rng.standard_normal(self.embedding_dim)

    def embed_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """Embed a list of nodes, keyed by their index in the input list."""
        return {idx: self.embed_node(node) for idx, node in enumerate(nodes)}
