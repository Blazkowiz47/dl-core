"""Tests for component registry resolution behavior."""

from __future__ import annotations

import pytest

from dl_core.core.registry import ComponentRegistry


def test_registry_rejects_unregistered_prefix_variants() -> None:
    """A misspelled or suffixed name must not select another component."""
    registry = ComponentRegistry("Metric manager")

    @registry.register("standard")
    class StandardManager:
        pass

    @registry.register("standard_act")
    class StandardActManager:
        pass

    assert registry.get_class("standard") is StandardManager
    assert registry.get_class("standard_act") is StandardActManager
    assert not registry.is_registered("standard_act_custom")
    with pytest.raises(NotImplementedError, match="standard_act_custom"):
        registry.get_class("standard_act_custom")
    with pytest.raises(NotImplementedError, match="standard_custom"):
        registry.get("standard_custom")


def test_registry_resolves_exact_match() -> None:
    """An explicitly registered name resolves to its class."""
    registry = ComponentRegistry("Model")

    @registry.register("resnet")
    class ResNet:
        pass

    @registry.register("resnet50")
    class ResNet50:
        pass

    assert registry.get_class("resnet50") is ResNet50
    assert isinstance(registry.get("resnet50"), ResNet50)


def test_multi_name_registration_is_atomic_on_collision() -> None:
    """A rejected alias must not leave earlier names partially registered."""

    registry = ComponentRegistry("Model")

    class ExistingModel:
        pass

    class NewModel:
        pass

    registry.register_class("taken", ExistingModel)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(["new", "taken"])(NewModel)

    assert not registry.is_registered("new")
    assert registry.get_class("taken") is ExistingModel
