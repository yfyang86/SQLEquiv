"""Plugin registry for custom equivalence methods.

Third-party packages can add an analysis method without modifying the core
library by either:

1. Calling :func:`register_method` directly at import time, or
2. Declaring the method in their ``pyproject.toml`` under the
   ``sql_equivalence.methods`` entry-point group, e.g.::

       [project.entry-points."sql_equivalence.methods"]
       my_method = "my_package.my_module:MyChecker"

The :class:`SQLEquivalenceAnalyzer` looks up methods through this registry,
so a registered plugin becomes usable immediately::

    analyzer.analyze(sql1, sql2, methods=["algebraic", "my_method"])

The plugin contract is intentionally minimal: a plugin is any callable
``(parsed1, parsed2, detailed) -> dict`` whose returned dict contains at
least an ``is_equivalent`` boolean and a ``confidence`` float. The analyzer
wraps built-in methods with the same contract so there is a single code
path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

#: Signature every registered method must satisfy.
#: ``(parsed_query_1, parsed_query_2, detailed) -> dict``
MethodCallable = Callable[[Any, Any, bool], dict]

_REGISTRY: dict[str, MethodCallable] = {}


def register_method(name: str, runner: MethodCallable) -> None:
    """Register ``runner`` under ``name``.

    Raises :class:`ValueError` if ``name`` is already registered -- plugins
    should pick unique names or override explicitly via
    :func:`unregister_method`.
    """
    if name in _REGISTRY:
        raise ValueError(f"Analysis method {name!r} is already registered")
    _REGISTRY[name] = runner
    logger.debug("Registered analysis method: %s", name)


def unregister_method(name: str) -> None:
    """Remove a method from the registry (no-op if absent)."""
    _REGISTRY.pop(name, None)


def get_method(name: str) -> MethodCallable:
    """Return the registered callable for ``name``.

    Raises :class:`KeyError` when the method is unknown.
    """
    return _REGISTRY[name]


def iter_methods() -> Iterator[tuple[str, MethodCallable]]:
    """Iterate ``(name, runner)`` pairs for every registered method."""
    return iter(_REGISTRY.items())


def registered_names() -> tuple[str, ...]:
    """Return a stable tuple of the currently registered method names."""
    return tuple(sorted(_REGISTRY))


def load_entry_points(group: str = "sql_equivalence.methods") -> int:
    """Discover methods advertised via Python entry points.

    Returns the number of methods newly registered. Called automatically by
    :class:`SQLEquivalenceAnalyzer` at construction time; safe to call again
    to pick up methods registered after the analyzer was built.
    """
    try:
        # Python 3.10+
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8 unsupported
        return 0

    loaded = 0
    try:
        eps = entry_points(group=group)
    except TypeError:
        # Python 3.8/3.9 style
        eps = entry_points().get(group, [])  # type: ignore[assignment]

    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            runner = ep.load()
        except Exception:
            logger.exception("Failed to load plugin %r", ep.name)
            continue
        try:
            register_method(ep.name, runner)
            loaded += 1
        except ValueError:
            # Benign race: another entry point registered it first.
            pass
    return loaded
