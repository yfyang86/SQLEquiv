# sql_equivalence/equivalence/embedding_similarity.py
"""Embedding-based similarity checking for SQL queries."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .base import EquivalenceChecker, EquivalenceResult, EquivalenceType
from ..representations.embedding.graph_embedding import GraphEmbedding

logger = logging.getLogger(__name__)

class EmbeddingSimilarityChecker(EquivalenceChecker):
    """Check similarity using embedding representations."""
    
    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.95
        self.use_ensemble = False
        self.distance_metrics = ['cosine', 'euclidean', 'manhattan']
    
    def check_equivalence(self, query1: Union[GraphEmbedding, Any], 
                         query2: Union[GraphEmbedding, Any]) -> EquivalenceResult:
        """Check equivalence using embedding similarity."""
        import time
        start_time = time.time()
        
        result = EquivalenceResult(
            is_equivalent=False,
            equivalence_type=EquivalenceType.NOT_EQUIVALENT,
            confidence=0.0
        )
        
        try:
            # Get embeddings
            embedding1 = self._get_embedding(query1)
            embedding2 = self._get_embedding(query2)
            
            if embedding1 is None or embedding2 is None:
                result.details['error'] = 'Failed to generate embeddings'
                return result
            
            # Compute similarity
            similarity = self.compute_similarity(embedding1, embedding2)
            result.confidence = similarity
            result.details['similarity_score'] = similarity
            
            # Determine equivalence
            if similarity >= self.similarity_threshold:
                result.is_equivalent = True
                if similarity >= 0.99:
                    result.equivalence_type = EquivalenceType.EXACT
                elif similarity >= 0.95:
                    result.equivalence_type = EquivalenceType.SEMANTIC
                else:
                    result.equivalence_type = EquivalenceType.APPROXIMATE
                
                result.add_proof_step(f"Embedding similarity: {similarity:.4f}")
            
            # Additional analysis
            if self.use_ensemble:
                ensemble_result = self._ensemble_similarity(embedding1, embedding2)
                result.details['ensemble_scores'] = ensemble_result
        
        except Exception as e:
            logger.error(f"Error checking embedding similarity: {e}")
            result.details['error'] = str(e)
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def compute_similarity(self, embedding1: Union[np.ndarray, Any], 
                          embedding2: Union[np.ndarray, Any]) -> float:
        """Compute similarity between embeddings."""
        if isinstance(embedding1, GraphEmbedding):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, GraphEmbedding):
            embedding2 = embedding2.embedding
        
        # Ensure numpy arrays
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Normalize embeddings
        embedding1 = normalize(embedding1.reshape(1, -1))[0]
        embedding2 = normalize(embedding2.reshape(1, -1))[0]
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def _get_embedding(self, query: Any) -> Optional[np.ndarray]:
        """Get embedding for a query."""
        if isinstance(query, GraphEmbedding):
            if query.embedding is not None:
                return query.embedding
            else:
                # Generate embedding
                query.build()
                return query.embedding
        
        elif isinstance(query, np.ndarray):
            return query
        
        elif self.embedding_model is not None:
            # Use provided model to generate embedding
            return self.embedding_model.encode(query)
        
        return None
    
    def _ensemble_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> Dict[str, float]:
        """Compute similarity using multiple metrics."""
        results = {}
        
        # Cosine similarity
        results['cosine'] = self.compute_similarity(embedding1, embedding2)
        
        # Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        results['euclidean'] = 1.0 / (1.0 + euclidean_dist)
        
        # Manhattan distance (converted to similarity)
        manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
        results['manhattan'] = 1.0 / (1.0 + manhattan_dist)
        
        # Pearson correlation
        results['pearson'] = np.corrcoef(embedding1, embedding2)[0, 1]
        
        return results
    
    def set_embedding_model(self, model: Any) -> None:
        """Set the embedding model."""
        self.embedding_model = model
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set similarity threshold for equivalence."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
    
    def batch_compute_similarity(self, embeddings1: List[np.ndarray], 
                               embeddings2: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise similarities for batches."""
        # Stack embeddings
        matrix1 = np.vstack(embeddings1)
        matrix2 = np.vstack(embeddings2)
        
        # Normalize
        matrix1 = normalize(matrix1)
        matrix2 = normalize(matrix2)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(matrix1, matrix2)
        
        return similarities
    
    def find_nearest_neighbors(self, query_embedding: np.ndarray, 
                             database_embeddings: List[np.ndarray], 
                             k: int = 5) -> List[Tuple[int, float]]:
        """Find k nearest neighbors in embedding space."""
        query_embedding = normalize(query_embedding.reshape(1, -1))[0]
        
        similarities = []
        for i, db_embedding in enumerate(database_embeddings):
            db_embedding = normalize(db_embedding.reshape(1, -1))[0]
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                db_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]