"""Embedding model interfaces.

The default :class:`HashingVectorEncoder` is a zero-dependency baseline that
treats each SQL query as a bag of word-level tokens and hashes them into a
fixed-size vector. It is deterministic, requires no training, and is good
enough as a sanity baseline.

Richer backends (a GNN on the query graph, a pretrained CodeBERT-style
encoder, ...) are expected to live in separate packages and plug in via the
:mod:`sql_equivalence.plugins` entry-point group. This module intentionally
stays lean so the core install has no additional heavy dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


class HashingVectorEncoder:
    """Deterministic hashing encoder for SQL queries.

    Implements the informal ``encode(query) -> np.ndarray`` contract expected
    by :class:`~sql_equivalence.equivalence.embedding_similarity.EmbeddingSimilarityChecker`.
    The encoder accepts either a raw SQL string or any object exposing a
    ``sql`` attribute (e.g. :class:`~sql_equivalence.parser.sql_parser.ParsedQuery`).
    """

    def __init__(self, dim: int = 256, ngram_range: tuple = (1, 2)):
        if dim <= 0:
            raise ValueError("dim must be positive")
        lo, hi = ngram_range
        if lo < 1 or hi < lo:
            raise ValueError(f"Invalid ngram_range: {ngram_range!r}")
        self.dim = dim
        self.ngram_range = ngram_range

    # ----------------------------------------------------------------- public
    def encode(self, query: Any) -> np.ndarray:
        sql = self._extract_sql(query)
        vector = np.zeros(self.dim, dtype=float)
        for token in self._iter_ngrams(self._tokenize(sql)):
            index, sign = self._hash(token)
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    # ------------------------------------------------------------- internals
    @staticmethod
    def _extract_sql(query: Any) -> str:
        if isinstance(query, str):
            return query
        sql = getattr(query, 'sql', None)
        if isinstance(sql, str):
            return sql
        raise TypeError(
            "HashingVectorEncoder.encode expects a SQL string or an object "
            "with a 'sql' attribute"
        )

    @staticmethod
    def _tokenize(sql: str) -> Iterable[str]:
        return [m.group().lower() for m in _TOKEN_RE.finditer(sql)]

    def _iter_ngrams(self, tokens: list) -> Iterable[str]:
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n <= 0 or n > len(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                yield ' '.join(tokens[i:i + n])

    def _hash(self, token: str) -> tuple:
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        index = int.from_bytes(digest[:6], 'big') % self.dim
        # Use a sign bit from the tail of the digest so collisions partially
        # cancel out, following Weinberger et al.'s "hashing trick".
        sign = 1.0 if (digest[-1] & 1) else -1.0
        return index, sign


def default_encoder(dim: int | None = None) -> HashingVectorEncoder:
    """Return a :class:`HashingVectorEncoder` with library-wide defaults."""
    return HashingVectorEncoder(dim=dim or 256)
