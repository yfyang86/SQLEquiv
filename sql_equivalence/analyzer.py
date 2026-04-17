"""Main analyzer module for SQL equivalence checking."""

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .equivalence.algebraic_equivalence import AlgebraicEquivalenceChecker
from .equivalence.embedding_similarity import EmbeddingSimilarityChecker
from .equivalence.graph_equivalence import GraphEquivalenceChecker
from .parser.sql_parser import ParsedQuery, SQLParser
from .plugins import iter_methods, load_entry_points, register_method, unregister_method

logger = logging.getLogger(__name__)

#: All analysis methods known to the analyzer, in their default order.
DEFAULT_METHODS: Tuple[str, ...] = ('algebraic', 'graph', 'embedding')


@dataclass
class AnalysisResult:
    """Result of SQL equivalence analysis."""

    sql1: str
    sql2: str
    is_equivalent: bool
    confidence: float
    method_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sql1': self.sql1,
            'sql2': self.sql2,
            'is_equivalent': self.is_equivalent,
            'confidence': self.confidence,
            'method_results': self.method_results,
            'execution_time': self.execution_time,
        }


#: Signature for a method runner: ``(parsed1, parsed2, detailed) -> Dict``.
MethodRunner = Callable[[ParsedQuery, ParsedQuery, bool], Dict[str, Any]]


class SQLEquivalenceAnalyzer:
    """Main SQL equivalence analyzer class."""

    def __init__(
        self,
        dialect: str = 'postgres',
        enable_caching: bool = True,
        embedding_model: Optional[Any] = None,
    ):
        """Initialize the SQL equivalence analyzer.

        Args:
            dialect: SQL dialect to use.
            enable_caching: Whether to memoize analysis results.
            embedding_model: Pre-trained embedding model, if any.
        """
        self.dialect = dialect
        self.enable_caching = enable_caching

        self.parser = SQLParser(dialect=dialect)
        self.algebraic_checker = AlgebraicEquivalenceChecker()
        self.graph_checker = GraphEquivalenceChecker()
        self.embedding_checker = EmbeddingSimilarityChecker(embedding_model)

        self._cache: Optional[
            Dict[Tuple[str, str, Tuple[str, ...], bool], AnalysisResult]
        ] = ({} if enable_caching else None)
        self._builtin_runners: Dict[str, MethodRunner] = {
            'algebraic': self._run_algebraic,
            'graph': self._run_graph,
            'embedding': self._run_embedding,
        }

        # Expose the built-ins to the plugin registry so that third-party
        # plugins and built-ins share a single dispatch table.
        for name, runner in self._builtin_runners.items():
            try:
                register_method(name, runner)
            except ValueError:
                # Built-ins may already be registered if the analyzer is
                # instantiated multiple times -- that is fine.
                pass
        load_entry_points()

        logger.info("Initialized SQLEquivalenceAnalyzer with dialect: %s", dialect)

    @property
    def available_methods(self) -> Tuple[str, ...]:
        """Return the names of every method currently callable via ``analyze``."""
        return tuple(sorted(name for name, _ in iter_methods()))

    def register_method(self, name: str, runner: MethodRunner) -> None:
        """Register a custom analysis method on this analyzer instance.

        The method is added to the global plugin registry; other analyzers
        created after this call will see it too. Use
        :meth:`unregister_method` to remove it again.
        """
        register_method(name, runner)

    def unregister_method(self, name: str) -> None:
        """Remove a previously registered method from the registry."""
        unregister_method(name)

    # ------------------------------------------------------------------ public
    def analyze(
        self,
        sql1: str,
        sql2: str,
        methods: Optional[Sequence[str]] = None,
        detailed: bool = False,
    ) -> AnalysisResult:
        """Analyze two SQL queries for equivalence.

        Args:
            sql1: First SQL query.
            sql2: Second SQL query.
            methods: Methods to run; defaults to all. Unknown methods are rejected.
            detailed: Include extended analysis details in the method results.

        Returns:
            :class:`AnalysisResult` summarizing the analysis.
        """
        methods = tuple(methods) if methods is not None else DEFAULT_METHODS
        self._validate_methods(methods)

        cache_key = (sql1, sql2, tuple(sorted(methods)), bool(detailed))
        if self._cache is not None and cache_key in self._cache:
            logger.debug("Returning cached result")
            return self._cache[cache_key]

        start_time = time.time()
        try:
            parsed1 = self.parser.parse(sql1)
            parsed2 = self.parser.parse(sql2)

            method_results: Dict[str, Any] = {}
            votes: List[bool] = []
            for method in methods:
                runner = self._resolve_runner(method)
                outcome = runner(parsed1, parsed2, detailed)
                method_results[method] = outcome
                votes.append(bool(outcome['is_equivalent']))

            result = AnalysisResult(
                sql1=sql1,
                sql2=sql2,
                is_equivalent=bool(votes) and all(votes),
                confidence=(sum(votes) / len(votes)) if votes else 0.0,
                method_results=method_results,
                execution_time=time.time() - start_time,
            )
        except Exception:
            logger.exception("Error analyzing queries")
            raise

        if self._cache is not None:
            self._cache[cache_key] = result
        return result

    def batch_analyze(
        self,
        query_pairs: Sequence[Tuple[str, str]],
        methods: Optional[Sequence[str]] = None,
        n_jobs: int = 1,
    ) -> List[AnalysisResult]:
        """Analyze multiple query pairs, optionally in parallel.

        Args:
            query_pairs: Pairs of queries to compare.
            methods: Methods to run.
            n_jobs: Number of parallel workers; ``-1`` uses all CPUs.
        """
        if n_jobs == 1:
            return [self.analyze(a, b, methods) for a, b in query_pairs]

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        results: List[AnalysisResult] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self.analyze, a, b, methods) for a, b in query_pairs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Cache cleared")

    # ----------------------------------------------------------------- private
    def _resolve_runner(self, method: str) -> MethodRunner:
        # Built-ins first -- they are bound methods and cheapest to dispatch.
        if method in self._builtin_runners:
            return self._builtin_runners[method]
        # Fall back to the plugin registry for third-party methods.
        for name, runner in iter_methods():
            if name == method:
                return runner
        raise KeyError(method)

    def _validate_methods(self, methods: Sequence[str]) -> None:
        known = set(self._builtin_runners) | {name for name, _ in iter_methods()}
        unknown = [m for m in methods if m not in known]
        if unknown:
            raise ValueError(
                f"Unknown analysis method(s): {unknown}. "
                f"Supported: {sorted(known)}"
            )

    @staticmethod
    def _base_result(equiv_result: Any) -> Dict[str, Any]:
        return {
            'is_equivalent': equiv_result.is_equivalent,
            'confidence': equiv_result.confidence,
            'equivalence_type': equiv_result.equivalence_type.value,
        }

    def _run_algebraic(
        self, parsed1: ParsedQuery, parsed2: ParsedQuery, detailed: bool
    ) -> Dict[str, Any]:
        expr1 = parsed1.to_algebraic()
        expr2 = parsed2.to_algebraic()
        equiv_result = self.algebraic_checker.check_equivalence(expr1, expr2)

        result = self._base_result(equiv_result)
        if detailed:
            result['details'] = equiv_result.details
            result['proof_steps'] = equiv_result.proof_steps
            result['canonical_form1'] = expr1.to_canonical_form()
            result['canonical_form2'] = expr2.to_canonical_form()
        return result

    def _run_graph(
        self, parsed1: ParsedQuery, parsed2: ParsedQuery, detailed: bool
    ) -> Dict[str, Any]:
        graph1 = parsed1.to_graph()
        graph2 = parsed2.to_graph()
        equiv_result = self.graph_checker.check_equivalence(graph1, graph2)

        result = self._base_result(equiv_result)
        if detailed:
            result['details'] = equiv_result.details
            result['graph_edit_distance'] = self.graph_checker.compute_graph_edit_distance(
                graph1, graph2
            )
        return result

    def _run_embedding(
        self, parsed1: ParsedQuery, parsed2: ParsedQuery, detailed: bool
    ) -> Dict[str, Any]:
        equiv_result = self.embedding_checker.check_equivalence(parsed1, parsed2)
        result = self._base_result(equiv_result)
        # Embedding API traditionally exposes a ``similarity_score``.
        result['similarity_score'] = equiv_result.confidence
        if detailed:
            result['details'] = equiv_result.details
        return result
