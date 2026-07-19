"""Built-in environment adapters and environment creation helpers."""

from __future__ import annotations

from typing import Any

from dl_core.core.registry import ENVIRONMENT_REGISTRY
from dl_core.core.rl_types import Environment

from .gymnasium_environment import GymnasiumEnvironment


def make_environment(config: dict[str, Any]) -> Environment[Any, Any]:
    """Create a registered environment from a configuration mapping."""
    return _make_environment(config)


def _make_environment(config: dict[str, Any]) -> Environment[Any, Any]:
    if not isinstance(config, dict):
        raise TypeError("Environment config must be a mapping")

    environment_name = config.get("name")
    if not isinstance(environment_name, str) or not environment_name:
        raise ValueError("environment.name must be a non-empty string")

    environment_config = {
        key: value for key, value in config.items() if key != "name"
    }
    return ENVIRONMENT_REGISTRY.get(environment_name, environment_config)


__all__ = ["GymnasiumEnvironment", "make_environment"]
