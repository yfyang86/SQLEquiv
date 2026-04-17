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


def _always_equivalent(
    _parsed1: Any, _parsed2: Any, _detailed: bool
) -> Dict[str, Any]:
    return {'is_equivalent': True, 'confidence': 1.0, 'equivalence_type': 'exact'}


def _always_non_equivalent(
    _parsed1: Any, _parsed2: Any, _detailed: bool
) -> Dict[str, Any]:
    return {'is_equivalent': False, 'confidence': 0.0, 'equivalence_type': 'not_equivalent'}


# ---------------------------------------------------------------- basics
def test_custom_method_is_invoked_by_analyzer() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    analyzer.register_method('always_eq', _always_equivalent)

    result = analyzer.analyze("SELECT 1", "SELECT 2", methods=['always_eq'])
    assert result.is_equivalent is True
    assert 'always_eq' in result.method_results


def test_duplicate_registration_raises() -> None:
    plugins.register_method('dup', _always_equivalent)
    with pytest.raises(ValueError, match='already registered'):
        plugins.register_method('dup', _always_equivalent)


def test_unknown_method_still_raises() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    with pytest.raises(ValueError, match='Unknown analysis method'):
        analyzer.analyze("SELECT 1", "SELECT 1", methods=['missing'])


def test_unregister_removes_method() -> None:
    plugins.register_method('temp', _always_equivalent)
    plugins.unregister_method('temp')
    assert 'temp' not in plugins.registered_names()


def test_unregister_unknown_is_noop() -> None:
    # Must not raise.
    plugins.unregister_method('never_registered')


def test_registered_names_is_sorted() -> None:
    plugins.register_method('zebra', _always_equivalent)
    plugins.register_method('apple', _always_equivalent)
    names = plugins.registered_names()
    assert names == tuple(sorted(names))


# ----------------------------------------------------------- analyzer wiring
def test_analyzer_available_methods_includes_builtins() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    names = set(analyzer.available_methods)
    assert {'algebraic', 'graph', 'embedding'} <= names


def test_analyzer_available_methods_updates_after_registration() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    analyzer.register_method('mock', _always_equivalent)
    assert 'mock' in analyzer.available_methods


def test_mixing_builtin_and_custom_methods() -> None:
    analyzer = SQLEquivalenceAnalyzer()
    analyzer.register_method('always_no', _always_non_equivalent)
    result = analyzer.analyze(
        "SELECT id FROM t", "SELECT id FROM t",
        methods=['algebraic', 'always_no'],
    )
    # Unanimous vote requires all methods to agree; the custom method votes No.
    assert result.is_equivalent is False
    assert result.confidence == pytest.approx(0.5)


def test_plugin_return_value_must_have_is_equivalent() -> None:
    """A plugin that returns the wrong shape should cause analyze() to fail
    loudly, not produce a nonsense result."""
    def bad(_p1, _p2, _d):
        return {'confidence': 1.0}  # missing 'is_equivalent'

    analyzer = SQLEquivalenceAnalyzer()
    analyzer.register_method('bad', bad)
    with pytest.raises(KeyError):
        analyzer.analyze("SELECT 1", "SELECT 1", methods=['bad'])


# -------------------------------------------------------- entry-point loading
def test_load_entry_points_is_idempotent() -> None:
    # Twice should not double-register or raise.
    first = plugins.load_entry_points('nonexistent_group_name_for_tests')
    second = plugins.load_entry_points('nonexistent_group_name_for_tests')
    assert first == 0
    assert second == 0
