# sql_equivalence/analyzer.py (修复后的版本)
"""Main analyzer module for SQL equivalence checking."""

from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import time

from .parser.sql_parser import SQLParser, ParsedQuery
from .representations.algebraic.relational_algebra import AlgebraicExpression
from .representations.graph.query_graph import QueryGraph
from .representations.graph.lqt import LogicalQueryTree
from .representations.embedding.graph_embedding import GraphEmbedding
from .equivalence.algebraic_equivalence import AlgebraicEquivalenceChecker
from .equivalence.graph_equivalence import GraphEquivalenceChecker
from .equivalence.embedding_similarity import EmbeddingSimilarityChecker

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of SQL equivalence analysis."""
    sql1: str
    sql2: str
    is_equivalent: bool
    confidence: float
    method_results: Dict[str, Any]
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'sql1': self.sql1,
            'sql2': self.sql2,
            'is_equivalent': self.is_equivalent,
            'confidence': self.confidence,
            'method_results': self.method_results,
            'execution_time': self.execution_time
        }

class SQLEquivalenceAnalyzer:
    """Main SQL equivalence analyzer class."""
    
    def __init__(self, 
                 dialect: str = 'postgres',
                 enable_caching: bool = True,
                 embedding_model: Optional[Any] = None):
        """
        Initialize the SQL equivalence analyzer.
        
        Args:
            dialect: SQL dialect to use
            enable_caching: Whether to enable caching of results
            embedding_model: Pre-trained embedding model (optional)
        """
        self.dialect = dialect
        self.enable_caching = enable_caching
        
        # Initialize components
        self.parser = SQLParser(dialect=dialect)
        self.algebraic_checker = AlgebraicEquivalenceChecker()
        self.graph_checker = GraphEquivalenceChecker()
        self.embedding_checker = EmbeddingSimilarityChecker(embedding_model)
        
        # Cache for parsed queries and results
        self._cache = {} if enable_caching else None
        
        logger.info(f"Initialized SQLEquivalenceAnalyzer with dialect: {dialect}")
    
    def analyze(self, 
                sql1: str, 
                sql2: str, 
                methods: List[str] = None,
                detailed: bool = False) -> AnalysisResult:
        """
        Analyze two SQL queries for equivalence.
        
        Args:
            sql1: First SQL query
            sql2: Second SQL query
            methods: List of methods to use ['algebraic', 'graph', 'embedding']
            detailed: Whether to include detailed analysis results
        
        Returns:
            AnalysisResult object containing the analysis results
        """
        start_time = time.time()
        
        if methods is None:
            methods = ['algebraic', 'graph', 'embedding']
        
        # Check cache
        cache_key = (sql1, sql2, tuple(sorted(methods)))
        if self._cache and cache_key in self._cache:
            logger.debug("Returning cached result")
            return self._cache[cache_key]
        
        try:
            # Parse queries
            parsed1 = self.parser.parse(sql1)
            parsed2 = self.parser.parse(sql2)
            
            method_results = {}
            equivalence_votes = []
            
            # Algebraic method
            if 'algebraic' in methods:
                alg_result = self._check_algebraic_equivalence(parsed1, parsed2, detailed)
                method_results['algebraic'] = alg_result
                equivalence_votes.append(alg_result['is_equivalent'])
            
            # Graph method
            if 'graph' in methods:
                graph_result = self._check_graph_equivalence(parsed1, parsed2, detailed)
                method_results['graph'] = graph_result
                equivalence_votes.append(graph_result['is_equivalent'])
            
            # Embedding method
            if 'embedding' in methods:
                emb_result = self._check_embedding_similarity(parsed1, parsed2, detailed)
                method_results['embedding'] = emb_result
                equivalence_votes.append(emb_result['is_equivalent'])
            
            # Determine overall equivalence
            is_equivalent = all(equivalence_votes) if equivalence_votes else False
            confidence = sum(1 for v in equivalence_votes if v) / len(equivalence_votes) if equivalence_votes else 0.0
            
            execution_time = time.time() - start_time
            
            result = AnalysisResult(
                sql1=sql1,
                sql2=sql2,
                is_equivalent=is_equivalent,
                confidence=confidence,
                method_results=method_results,
                execution_time=execution_time
            )
            
            # Cache result
            if self._cache is not None:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing queries: {e}")
            raise
    
    def _check_algebraic_equivalence(self, 
                                   parsed1: ParsedQuery, 
                                   parsed2: ParsedQuery,
                                   detailed: bool) -> Dict[str, Any]:
        """Check equivalence using algebraic method."""
        expr1 = parsed1.to_algebraic()
        expr2 = parsed2.to_algebraic()
        
        # Get EquivalenceResult object
        equiv_result = self.algebraic_checker.check_equivalence(expr1, expr2)
        
        result = {
            'is_equivalent': equiv_result.is_equivalent,
            'confidence': equiv_result.confidence,
            'equivalence_type': equiv_result.equivalence_type.value
        }
        
        if detailed:
            result['details'] = equiv_result.details
            result['proof_steps'] = equiv_result.proof_steps
            result['canonical_form1'] = expr1.to_canonical_form()
            result['canonical_form2'] = expr2.to_canonical_form()
            
        return result
    
    def _check_graph_equivalence(self, 
                                parsed1: ParsedQuery, 
                                parsed2: ParsedQuery,
                                detailed: bool) -> Dict[str, Any]:
        """Check equivalence using graph method."""
        graph1 = parsed1.to_graph()
        graph2 = parsed2.to_graph()
        
        # Get EquivalenceResult object
        equiv_result = self.graph_checker.check_equivalence(graph1, graph2)
        
        result = {
            'is_equivalent': equiv_result.is_equivalent,
            'confidence': equiv_result.confidence,
            'equivalence_type': equiv_result.equivalence_type.value
        }
        
        if detailed:
            result['details'] = equiv_result.details
            result['graph_edit_distance'] = self.graph_checker.compute_graph_edit_distance(graph1, graph2)
            
        return result
    
    def _check_embedding_similarity(self, 
                                   parsed1: ParsedQuery, 
                                   parsed2: ParsedQuery,
                                   detailed: bool) -> Dict[str, Any]:
        """Check similarity using embedding method."""
        # Get EquivalenceResult object
        equiv_result = self.embedding_checker.check_equivalence(parsed1, parsed2)
        
        result = {
            'is_equivalent': equiv_result.is_equivalent,
            'similarity_score': equiv_result.confidence,
            'confidence': equiv_result.confidence,
            'equivalence_type': equiv_result.equivalence_type.value
        }
        
        if detailed:
            result['details'] = equiv_result.details
            
        return result
    
    def batch_analyze(self, 
                     query_pairs: List[Tuple[str, str]], 
                     methods: List[str] = None,
                     n_jobs: int = 1) -> List[AnalysisResult]:
        """
        Analyze multiple query pairs for equivalence.
        
        Args:
            query_pairs: List of (sql1, sql2) tuples
            methods: Methods to use for analysis
            n_jobs: Number of parallel jobs (-1 for all cores)
        
        Returns:
            List of AnalysisResult objects
        """
        results = []
        
        if n_jobs == 1:
            # Sequential processing
            for sql1, sql2 in query_pairs:
                result = self.analyze(sql1, sql2, methods)
                results.append(result)
        else:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(self.analyze, sql1, sql2, methods): (sql1, sql2)
                    for sql1, sql2 in query_pairs
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        
        return results
    
    def clear_cache(self):
        """Clear the analysis cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Cache cleared")
