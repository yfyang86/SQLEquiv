"""Embedding-based similarity checking for SQL queries."""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ..parser.sql_parser import ParsedQuery
from ..representations.embedding.graph_embedding import GraphEmbedding
from .base import EquivalenceChecker, EquivalenceResult, EquivalenceType

logger = logging.getLogger(__name__)


class EmbeddingSimilarityChecker(EquivalenceChecker):
    """Check similarity using embedding representations.

    Accepts :class:`~sql_equivalence.parser.sql_parser.ParsedQuery`,
    :class:`GraphEmbedding`, or raw ``numpy`` vectors. When a parsed query
    is passed, the checker builds the default :class:`GraphEmbedding` on
    demand — callers that need a richer model can pass one via
    ``embedding_model`` (must expose ``encode(query) -> np.ndarray``).
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.95,
        use_ensemble: bool = False,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.use_ensemble = use_ensemble
        self.distance_metrics = ['cosine', 'euclidean', 'manhattan']

    def check_equivalence(
        self,
        query1: Union[ParsedQuery, GraphEmbedding, np.ndarray, Any],
        query2: Union[ParsedQuery, GraphEmbedding, np.ndarray, Any],
    ) -> EquivalenceResult:
        start_time = time.time()
        result = EquivalenceResult(
            is_equivalent=False,
            equivalence_type=EquivalenceType.NOT_EQUIVALENT,
            confidence=0.0,
        )

        try:
            embedding1 = self._get_embedding(query1)
            embedding2 = self._get_embedding(query2)

            if embedding1 is None or embedding2 is None:
                result.details['error'] = 'Failed to generate embeddings'
                return result

            similarity = self.compute_similarity(embedding1, embedding2)
            result.confidence = similarity
            result.details['similarity_score'] = similarity

            if similarity >= self.similarity_threshold:
                result.is_equivalent = True
                if similarity >= 0.99:
                    result.equivalence_type = EquivalenceType.EXACT
                elif similarity >= 0.95:
                    result.equivalence_type = EquivalenceType.SEMANTIC
                else:
                    result.equivalence_type = EquivalenceType.APPROXIMATE
                result.add_proof_step(f"Embedding similarity: {similarity:.4f}")

            if self.use_ensemble:
                result.details['ensemble_scores'] = self._ensemble_similarity(
                    embedding1, embedding2
                )

        except Exception as exc:  # pragma: no cover
            logger.exception("Error checking embedding similarity")
            result.details['error'] = str(exc)
        finally:
            result.execution_time = time.time() - start_time

        return result

    # --------------------------------------------------------------- metrics
    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, Any],
        embedding2: Union[np.ndarray, Any],
    ) -> float:
        """Cosine similarity, mapped to ``[0.0, 1.0]`` for easier thresholding."""
        if isinstance(embedding1, GraphEmbedding):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, GraphEmbedding):
            embedding2 = embedding2.embedding

        arr1 = np.asarray(embedding1, dtype=float).reshape(1, -1)
        arr2 = np.asarray(embedding2, dtype=float).reshape(1, -1)

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 and norm2 == 0:
            return 1.0
        if norm1 == 0 or norm2 == 0:
            return 0.0

        arr1 = normalize(arr1)
        arr2 = normalize(arr2)
        cosine = float(cosine_similarity(arr1, arr2)[0, 0])
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    # --------------------------------------------------------------- helpers
    def _get_embedding(self, query: Any) -> Optional[np.ndarray]:
        if isinstance(query, ParsedQuery):
            if self.embedding_model is not None and hasattr(self.embedding_model, 'encode'):
                return np.asarray(self.embedding_model.encode(query))
            embedding = GraphEmbedding(query)
            embedding.build()
            return embedding.embedding

        if isinstance(query, GraphEmbedding):
            if query.embedding is None:
                query.build()
            return query.embedding

        if isinstance(query, np.ndarray):
            return query

        if self.embedding_model is not None and hasattr(self.embedding_model, 'encode'):
            return np.asarray(self.embedding_model.encode(query))

        return None

    def _ensemble_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        results['cosine'] = self.compute_similarity(embedding1, embedding2)

        euclidean_dist = float(np.linalg.norm(embedding1 - embedding2))
        results['euclidean'] = 1.0 / (1.0 + euclidean_dist)

        manhattan_dist = float(np.sum(np.abs(embedding1 - embedding2)))
        results['manhattan'] = 1.0 / (1.0 + manhattan_dist)

        if np.std(embedding1) == 0 or np.std(embedding2) == 0:
            results['pearson'] = 0.0
        else:
            results['pearson'] = float(np.corrcoef(embedding1, embedding2)[0, 1])

        return results

    # ---------------------------------------------------------------- config
    def set_embedding_model(self, model: Any) -> None:
        self.embedding_model = model

    def set_similarity_threshold(self, threshold: float) -> None:
        self.similarity_threshold = max(0.0, min(1.0, threshold))

    def batch_compute_similarity(
        self,
        embeddings1: List[np.ndarray],
        embeddings2: List[np.ndarray],
    ) -> np.ndarray:
        """Pairwise cosine similarity matrix between two batches."""
        matrix1 = normalize(np.vstack(embeddings1))
        matrix2 = normalize(np.vstack(embeddings2))
        return cosine_similarity(matrix1, matrix2)
