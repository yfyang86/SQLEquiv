"""Graph embedding for queries."""

from typing import Dict, Optional

import numpy as np

from .encoder import QueryEncoder
from .node_embedding import NodeEmbedding


class GraphEmbedding(QueryEncoder):
    """Graph-based embedding for SQL queries.

    The default implementation is intentionally lightweight: it builds per-node
    embeddings via :class:`NodeEmbedding` and aggregates them by mean pooling.
    Production-grade implementations (GNNs, node2vec, ...) should subclass this
    and override :meth:`encode`.
    """

    def __init__(
        self,
        parsed_query: 'ParsedQuery',
        embedding_dim: int = 128,
        method: str = 'node2vec',
    ):
        super().__init__(parsed_query, embedding_dim)
        self.method = method
        self._node_embedder = NodeEmbedding(embedding_dim=embedding_dim)
        self.node_embeddings: Optional[Dict[int, np.ndarray]] = None
        self.graph_embedding: Optional[np.ndarray] = None

    def encode(self) -> np.ndarray:
        """Encode the query graph into a single embedding vector."""
        self.node_embeddings = self.get_node_embeddings()
        self.graph_embedding = self.aggregate_node_embeddings(self.node_embeddings)
        return self.graph_embedding

    def get_node_embeddings(self) -> Dict[int, np.ndarray]:
        """Return embeddings for the nodes of the underlying query graph."""
        graph = self.parsed_query.to_graph()
        nodes = [graph.node_attributes.get(n, {}) for n in graph.graph.nodes()]
        self._node_embedder.build_vocabulary(nodes)
        return self._node_embedder.embed_nodes(nodes)

    def aggregate_node_embeddings(
        self, node_embeddings: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Aggregate node embeddings via mean pooling."""
        if not node_embeddings:
            return np.zeros(self.embedding_dim)
        stacked = np.stack(list(node_embeddings.values()))
        return stacked.mean(axis=0)
