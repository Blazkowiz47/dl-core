"""Tests for component registry resolution behavior."""

from __future__ import annotations

from dl_core.core.registry import ComponentRegistry


def test_registry_prefers_the_most_specific_matching_prefix() -> None:
    """Overlapping registrations should resolve to the longest prefix."""
    registry = ComponentRegistry("Metric manager")

    @registry.register("standard")
    class StandardManager:
        pass

    @registry.register("standard_act")
    class StandardActManager:
        pass

    assert registry.get_class("standard_act_custom") is StandardActManager
    assert isinstance(registry.get("standard_act_custom"), StandardActManager)
    assert registry.get_class("standard_custom") is StandardManager


def test_registry_still_prefers_an_exact_match() -> None:
    """Exact registration names should win before prefix resolution."""
    registry = ComponentRegistry("Model")

    @registry.register("resnet")
    class ResNet:
        pass

    @registry.register("resnet50")
    class ResNet50:
        pass

    assert registry.get_class("resnet50") is ResNet50
    assert isinstance(registry.get("resnet50"), ResNet50)
