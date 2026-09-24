"""Tests for installed runtime extension discovery."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from dl_core import load_builtin_components, load_runtime_extensions
from dl_core.core import CALLBACK_REGISTRY


def test_builtin_loading_imports_runtime_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed integration should register without a project bootstrap."""

    class RuntimeCallback:
        pass

    class Extension:
        name = "test_runtime_extension"
        value = "test_extension"

        def load(self) -> type[RuntimeCallback]:
            CALLBACK_REGISTRY.register_class("test_runtime_callback", RuntimeCallback)
            return RuntimeCallback

    monkeypatch.setattr(
        "dl_core.entry_points",
        lambda *, group: [Extension()] if group == "dl_core.runtime_extensions" else [],
    )

    try:
        load_builtin_components()
        assert CALLBACK_REGISTRY.get_class("test_runtime_callback") is RuntimeCallback
    finally:
        CALLBACK_REGISTRY.unregister("test_runtime_callback")


def test_broken_runtime_extension_rolls_back_and_does_not_block_others(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken optional package must not poison registries or other imports."""

    class PartialCallback:
        pass

    class HealthyCallback:
        pass

    class BrokenExtension:
        name = "broken"
        value = "broken_package:load"

        def load(self) -> Any:
            CALLBACK_REGISTRY.register_class("partial_callback", PartialCallback)
            sys.modules["broken_package.partial"] = ModuleType("broken_package.partial")
            raise ImportError("missing dependency")

    class HealthyExtension:
        name = "healthy"
        value = "healthy_package:load"

        def load(self) -> type[HealthyCallback]:
            CALLBACK_REGISTRY.register_class("healthy_callback", HealthyCallback)
            return HealthyCallback

    monkeypatch.setattr(
        "dl_core.entry_points",
        lambda *, group: [BrokenExtension(), HealthyExtension()],
    )

    try:
        assert load_runtime_extensions() == ["healthy"]
        assert not CALLBACK_REGISTRY.is_registered("partial_callback")
        assert CALLBACK_REGISTRY.get_class("healthy_callback") is HealthyCallback
        assert "broken_package.partial" not in sys.modules
        assert "Skipping runtime extension 'broken'" in caplog.text
        assert "missing dependency" in caplog.text
    finally:
        CALLBACK_REGISTRY.unregister("healthy_callback")


def test_builtin_loading_continues_past_broken_runtime_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated CLI paths can still load their built-in components."""

    class BrokenExtension:
        name = "broken"
        value = "broken_package"

        def load(self) -> Any:
            raise ImportError("missing dependency")

    monkeypatch.setattr("dl_core.entry_points", lambda *, group: [BrokenExtension()])

    load_builtin_components()

    assert CALLBACK_REGISTRY.is_registered("checkpoint")
