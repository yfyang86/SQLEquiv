"""Tests for the plugin registry and analyzer extension points."""

from typing import Any, Dict

import pytest

from sql_equivalence import SQLEquivalenceAnalyzer, plugins


@pytest.fixture(autouse=True)
def _reset_registry():
    """Snapshot-and-restore the registry so tests do not leak state."""
    saved = dict(plugins._REGISTRY)  # noqa: SLF001
    yield
    plugins._REGISTRY.clear()  # noqa: SLF001
    plugins._REGISTRY.update(saved)  # noqa: SLF001


def _always_equivalent(_parsed1: Any, _parsed2: Any, _detailed: bool) -> Dict[str, Any]:
    return {'is_equivalent': True, 'confidence': 1.0, 'equivalence_type': 'exact'}


def test_custom_method_is_invoked_by_analyzer() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    analyzer.register_method('always_eq', _always_equivalent)

    result = analyzer.analyze("SELECT 1", "SELECT 2", methods=['always_eq'])
    assert result.is_equivalent is True
    assert 'always_eq' in result.method_results


def test_duplicate_registration_raises() -> None:
    plugins.register_method('dup', _always_equivalent)
    with pytest.raises(ValueError):
        plugins.register_method('dup', _always_equivalent)


def test_unknown_method_still_raises() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    with pytest.raises(ValueError, match='Unknown analysis method'):
        analyzer.analyze("SELECT 1", "SELECT 1", methods=['missing'])


def test_unregister_removes_method() -> None:
    plugins.register_method('temp', _always_equivalent)
    plugins.unregister_method('temp')
    assert 'temp' not in plugins.registered_names()
