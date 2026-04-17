"""Pluggable ML-backed encoders and similarity models.

The core install ships with :class:`HashingVectorEncoder` -- a deterministic,
zero-extra-dependency baseline. Heavier backends (GNNs, pretrained code
encoders) are expected to live in separate packages and register themselves
via the ``sql_equivalence.methods`` entry-point group documented in
:mod:`sql_equivalence.plugins`.
"""

from .embedding_models import HashingVectorEncoder, default_encoder

__all__ = ['HashingVectorEncoder', 'default_encoder']
