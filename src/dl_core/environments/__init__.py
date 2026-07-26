"""Built-in environment adapters and environment creation helpers."""

from __future__ import annotations

from typing import Any

from dl_core.core.registry import ENVIRONMENT_REGISTRY
from dl_core.core.rl_types import Environment, VectorEnvironment

from .action_history_wrapper import ActionHistoryWrapper
from .gymnasium_environment import GymnasiumEnvironment
from .gymnasium_vector_environment import GymnasiumVectorEnvironment


def make_environment(
    config: dict[str, Any],
) -> Environment[Any, Any] | VectorEnvironment[Any, Any]:
    """Create a registered environment from a configuration mapping."""
    if not isinstance(config, dict):
        raise TypeError("Environment config must be a mapping")

    environment_name = config.get("name")
    if not isinstance(environment_name, str) or not environment_name:
        raise ValueError("environment.name must be a non-empty string")

    environment_config = {
        key: value for key, value in config.items() if key != "name"
    }
    action_history_config = environment_config.pop("action_history", None)
    history_length = 4
    if action_history_config is not None:
        if not isinstance(action_history_config, dict):
            raise TypeError("environment.action_history must be a mapping")
        unknown_keys = set(action_history_config) - {"length"}
        if unknown_keys:
            raise ValueError(
                "environment.action_history contains unsupported settings: "
                f"{sorted(unknown_keys)}"
            )
        configured_length = action_history_config.get("length", 4)
        if isinstance(configured_length, bool) or not isinstance(
            configured_length,
            int,
        ):
            raise TypeError("environment.action_history.length must be an integer")
        if configured_length <= 0:
            raise ValueError("environment.action_history.length must be positive")
        history_length = configured_length
    environment = ENVIRONMENT_REGISTRY.get(environment_name, environment_config)
    if action_history_config is None:
        return environment
    try:
        return ActionHistoryWrapper(
            environment,
            history_length=history_length,
        )
    except Exception:
        environment.close()
        raise


__all__ = [
    "ActionHistoryWrapper",
    "GymnasiumEnvironment",
    "GymnasiumVectorEnvironment",
    "make_environment",
]
