# sql_equivalence/representations/embedding/encoder.py
"""Query encoder base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from ..base import QueryRepresentation

class QueryEncoder(QueryRepresentation):
    """Base class for query encoders."""
    
    def __init__(self, parsed_query: 'ParsedQuery', 
                 embedding_dim: int = 128):
        super().__init__(parsed_query)
        self.embedding_dim = embedding_dim
        self.embedding = None
    
    @abstractmethod
    def encode(self) -> np.ndarray:
        """Encode query into embedding vector."""
        pass
    
    def build(self) -> None:
        """Build embedding representation."""
        self.embedding = self.encode()
        self._built = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': 'query_embedding',
            'embedding_dim': self.embedding_dim,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'is_built': self._built
        }
    
    def to_string(self) -> str:
        """Convert to string representation."""
        return f"QueryEncoder(dim={self.embedding_dim}, built={self._built})"
    
    def visualize(self, output_path: Optional[str] = None) -> Any:
        """Visualize the embedding."""
        # Framework implementation - could show PCA/t-SNE visualization
        pass