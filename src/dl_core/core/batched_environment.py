"""Scalar and vector environment normalization for RL trainers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium.vector import AutoresetMode

from .rl_types import Environment, VectorEnvironment


class BatchedEnvironment:
    """Expose scalar and Gymnasium vector environments through one batch API."""

    def __init__(
        self,
        environment: Environment[Any, Any] | VectorEnvironment[Any, Any],
    ):
        self.environment = environment
        self.is_vector = isinstance(environment, VectorEnvironment)
        if self.is_vector:
            mode = environment.metadata.get("autoreset_mode")
            if mode != AutoresetMode.SAME_STEP:
                raise ValueError(
                    "Vector RL collection requires same-step autoreset and "
                    "metadata['autoreset_mode'] = AutoresetMode.SAME_STEP"
                )
            self.num_envs = int(environment.num_envs)
            self.observation_space = environment.single_observation_space
            self.action_space = environment.single_action_space
        else:
            self.num_envs = 1
            self.observation_space = environment.observation_space
            self.action_space = environment.action_space

    def reset_batch(
        self,
        seeds: list[int | None],
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Reset every environment lane and return batched observations."""
        return self._reset_batch(seeds)

    def _reset_batch(
        self,
        seeds: list[int | None],
    ) -> tuple[Any, list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count must match the number of environments")
        if self.is_vector:
            observations, infos = self.environment.reset(seed=seeds)
            return observations, self._unbatch_infos(infos)
        observation, info = self.environment.reset(seed=seeds[0])
        return self._stack_values([observation]), [info]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset a scalar underlying environment for compatibility."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.is_vector:
            raise RuntimeError("Use reset_batch for a vector environment")
        return self.environment.reset(seed=seed, options=options)

    def reset_lanes(self, reset_mask: np.ndarray) -> tuple[Any, list[dict[str, Any]]]:
        """Reset selected vector lanes after trainer-enforced truncation."""
        return self._reset_lanes(reset_mask)

    def _reset_lanes(
        self,
        reset_mask: np.ndarray,
    ) -> tuple[Any, list[dict[str, Any]]]:
        mask = np.asarray(reset_mask, dtype=np.bool_)
        if mask.shape != (self.num_envs,):
            raise ValueError("Reset mask shape must match the number of environments")
        if not mask.any():
            raise ValueError("At least one environment lane must be reset")
        if not self.is_vector:
            observation, info = self.environment.reset()
            return self._stack_values([observation]), [info]
        observations, infos = self.environment.reset(
            options={"reset_mask": mask},
        )
        return observations, self._unbatch_infos(infos)

    def step_batch(
        self,
        actions: list[Any],
    ) -> tuple[
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        list[Any],
    ]:
        """Step every lane and preserve terminal observations."""
        return self._step_batch(actions)

    def _step_batch(
        self,
        actions: list[Any],
    ) -> tuple[
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        list[Any],
    ]:
        if len(actions) != self.num_envs:
            raise ValueError("Action count must match the number of environments")
        if self.is_vector:
            observations, rewards, terminated, truncated, infos = (
                self.environment.step(self._stack_values(actions))
            )
            lane_infos = self._unbatch_infos(infos)
        else:
            observation, reward, lane_terminated, lane_truncated, info = (
                self.environment.step(actions[0])
            )
            observations = self._stack_values([observation])
            rewards = np.asarray([reward], dtype=np.float32)
            terminated = np.asarray([lane_terminated], dtype=np.bool_)
            truncated = np.asarray([lane_truncated], dtype=np.bool_)
            lane_infos = [info]
        rewards_array = np.asarray(rewards, dtype=np.float32)
        terminated_array = np.asarray(terminated, dtype=np.bool_)
        truncated_array = np.asarray(truncated, dtype=np.bool_)
        final_observations = [
            self.batch_item(observations, index)
            for index in range(self.num_envs)
        ]
        for index in np.flatnonzero(
            np.logical_or(terminated_array, truncated_array)
        ):
            if "final_obs" in lane_infos[int(index)]:
                final_observations[int(index)] = deepcopy(
                    lane_infos[int(index)]["final_obs"]
                )
        return (
            observations,
            rewards_array,
            terminated_array,
            truncated_array,
            lane_infos,
            final_observations,
        )

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step a scalar underlying environment for compatibility."""
        return self._step(action)

    def _step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.is_vector:
            raise RuntimeError("Use step_batch for a vector environment")
        return self.environment.step(action)

    def batch_item(self, values: Any, index: int) -> Any:
        """Return one environment lane from nested batched values."""
        return self._batch_item(values, index)

    def _batch_item(self, values: Any, index: int) -> Any:
        if isinstance(values, dict):
            return {
                key: self._batch_item(value, index)
                for key, value in values.items()
            }
        if isinstance(values, tuple):
            return tuple(self._batch_item(value, index) for value in values)
        value = values[index]
        return value.item() if isinstance(value, np.generic) else value

    def replace_batch_items(
        self,
        values: Any,
        replacements: Any,
        mask: np.ndarray,
    ) -> Any:
        """Replace selected lanes in nested batched values."""
        return self._replace_batch_items(values, replacements, mask)

    def _replace_batch_items(
        self,
        values: Any,
        replacements: Any,
        mask: np.ndarray,
    ) -> Any:
        if isinstance(values, dict):
            return {
                key: self._replace_batch_items(
                    value,
                    replacements[key],
                    mask,
                )
                for key, value in values.items()
            }
        if isinstance(values, tuple):
            return tuple(
                self._replace_batch_items(value, replacements[index], mask)
                for index, value in enumerate(values)
            )
        updated = np.asarray(values).copy()
        updated[mask] = np.asarray(replacements)[mask]
        return updated

    def stack_values(self, values: list[Any]) -> Any:
        """Stack per-lane values using the vector environment structure."""
        return self._stack_values(values)

    def _stack_values(self, values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, dict):
            return {
                key: self._stack_values([value[key] for value in values])
                for key in first
            }
        if isinstance(first, tuple):
            return tuple(
                self._stack_values([value[index] for value in values])
                for index in range(len(first))
            )
        return np.stack([np.asarray(value) for value in values])

    def _unbatch_infos(self, infos: dict[str, Any]) -> list[dict[str, Any]]:
        lane_infos = [{} for _ in range(self.num_envs)]
        for key, values in infos.items():
            if key.startswith("_"):
                continue
            mask = infos.get(f"_{key}")
            if mask is None:
                mask = np.ones(self.num_envs, dtype=np.bool_)
            for index in np.flatnonzero(np.asarray(mask, dtype=np.bool_)):
                lane_infos[int(index)][key] = self._batch_item(values, int(index))
        return lane_infos

    def render(self) -> Any:
        """Render the underlying environment."""
        return self._render()

    def _render(self) -> Any:
        return self.environment.render()

    def close(self) -> None:
        """Close the underlying environment."""
        self._close()

    def _close(self) -> None:
        self.environment.close()
