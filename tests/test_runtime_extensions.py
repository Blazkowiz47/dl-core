"""Tests for installed runtime extension discovery."""

from __future__ import annotations

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


def test_runtime_extension_import_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken installed integration must not disappear silently."""

    class BrokenExtension:
        name = "broken"
        value = "broken_package"

        def load(self) -> Any:
            raise ImportError("missing dependency")

    monkeypatch.setattr(
        "dl_core.entry_points",
        lambda *, group: [BrokenExtension()],
    )

    with pytest.raises(RuntimeError, match="broken.*broken_package") as error:
        load_runtime_extensions()

    assert isinstance(error.value.__cause__, ImportError)
