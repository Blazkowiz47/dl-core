"""Gymnasium vector-environment adapter."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from dl_core.core.registry import register_environment


@register_environment("gymnasium_vector")
class GymnasiumVectorEnvironment(gym.vector.VectorWrapper):
    """Create a same-step autoreset vector environment, asynchronously by default."""

    def __init__(self, config: dict[str, Any]):
        environment_id = config.get("id")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("environment.id must be a non-empty string")
        num_envs = int(config.get("num_envs", 1))
        if num_envs <= 0:
            raise ValueError("environment.num_envs must be positive")
        vectorization_mode = str(config.get("vectorization_mode", "async"))
        if vectorization_mode not in {"sync", "async"}:
            raise ValueError("vectorization_mode must be 'sync' or 'async'")
        make_kwargs = config.get("kwargs", {})
        if not isinstance(make_kwargs, dict):
            raise TypeError("environment.kwargs must be a mapping")
        vector_environment = gym.make_vec(
            environment_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
            },
            **make_kwargs,
        )
        super().__init__(vector_environment)
        self.supports_async_step = isinstance(
            vector_environment,
            gym.vector.AsyncVectorEnv,
        )

    def step_async(self, actions: Any) -> None:
        """Dispatch actions without waiting for asynchronous workers."""
        self._step_async(actions)

    def _step_async(self, actions: Any) -> None:
        if not self.supports_async_step:
            raise RuntimeError(
                "Asynchronous stepping requires vectorization_mode='async'"
            )
        self.env.step_async(actions)

    def step_wait(
        self,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Wait for a previously dispatched asynchronous vector step."""
        return self._step_wait()

    def _step_wait(
        self,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        if not self.supports_async_step:
            raise RuntimeError(
                "Asynchronous stepping requires vectorization_mode='async'"
            )
        return self.env.step_wait()
