"""Gymnasium environment adapter."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from dl_core.core.registry import register_environment


@register_environment("gymnasium")
class GymnasiumEnvironment(gym.Wrapper):
    """Create and expose a Gymnasium environment through the core registry."""

    def __init__(self, config: dict[str, Any]):
        environment_id = config.get("id")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("environment.id must be a non-empty string")

        make_kwargs = config.get("kwargs", {})
        if not isinstance(make_kwargs, dict):
            raise TypeError("environment.kwargs must be a mapping")

        super().__init__(gym.make(environment_id, **make_kwargs))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the wrapped environment."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the wrapped environment by one action."""
        return self._step(action)

    def _step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> Any:
        """Render the wrapped environment."""
        return self._render()

    def _render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        """Close the wrapped environment."""
        self._close()

    def _close(self) -> None:
        self.env.close()
