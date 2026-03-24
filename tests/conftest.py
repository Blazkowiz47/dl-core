"""Shared pytest fixtures for dl-core tests."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest

from dl_core.core import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def clear_scaffolded_resnet_wrapper() -> Iterator[None]:
    """Remove the scaffolded local ResNet wrapper between tests."""
    _remove_scaffolded_resnet_wrapper()
    yield
    _remove_scaffolded_resnet_wrapper()


def _remove_scaffolded_resnet_wrapper() -> None:
    """Clear the test-local ResNetExample wrapper from the registry and import cache."""
    MODEL_REGISTRY._components.pop("resnet_example", None)
    module_names = [
        module_name
        for module_name in sys.modules
        if module_name.endswith(".resnet_example")
        and not module_name.startswith("dl_core.")
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)
