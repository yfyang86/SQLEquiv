"""Query encoder base class."""

from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..base import QueryRepresentation


class QueryEncoder(QueryRepresentation):
    """Base class for query encoders."""

    def __init__(self, parsed_query: 'ParsedQuery', embedding_dim: int = 128):
        super().__init__(parsed_query)
        self.embedding_dim = embedding_dim
        self.embedding: Optional[np.ndarray] = None

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Encode query into an embedding vector."""

    def build(self) -> None:
        self.embedding = self.encode()
        self._built = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'query_embedding',
            'embedding_dim': self.embedding_dim,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'is_built': self._built,
        }

    def to_string(self) -> str:
        return f"QueryEncoder(dim={self.embedding_dim}, built={self._built})"

    def visualize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError(
            "QueryEncoder.visualize must be implemented by subclasses "
            "(e.g. PCA/t-SNE projection of the embedding)."
        )
