"""Observation wrapper that appends a fixed window of previous actions."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector.utils import batch_space

from dl_core.core.rl_types import Environment, VectorEnvironment


class ActionHistoryWrapper:
    """Append recent actions to scalar or vector environment observations."""

    def __init__(
        self,
        environment: Environment[Any, Any] | VectorEnvironment[Any, Any],
        *,
        history_length: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("Action history length must be positive")

        self.environment = environment
        self.history_length = history_length
        self.is_vector = isinstance(environment, VectorEnvironment)
        if self.is_vector:
            source_observation_space = environment.single_observation_space
            source_action_space = environment.single_action_space
            self.num_envs = int(environment.num_envs)
            self.metadata = environment.metadata
        else:
            source_observation_space = environment.observation_space
            source_action_space = environment.action_space
            self.num_envs = 1

        if not isinstance(source_observation_space, (Box, Discrete)):
            raise TypeError(
                "ActionHistoryWrapper requires a Box or Discrete observation space"
            )
        if not isinstance(source_action_space, (Box, Discrete)):
            raise TypeError(
                "ActionHistoryWrapper requires a Box or Discrete action space"
            )

        self.source_observation_space = source_observation_space
        self.source_action_space = source_action_space
        if isinstance(source_observation_space, Discrete):
            observation_low = np.zeros(source_observation_space.n, dtype=np.float32)
            observation_high = np.ones(source_observation_space.n, dtype=np.float32)
        else:
            observation_low = np.asarray(
                source_observation_space.low,
                dtype=np.float32,
            ).reshape(-1)
            observation_high = np.asarray(
                source_observation_space.high,
                dtype=np.float32,
            ).reshape(-1)

        if isinstance(source_action_space, Discrete):
            self.action_width = int(source_action_space.n)
            action_low = np.zeros(self.action_width, dtype=np.float32)
            action_high = np.ones(self.action_width, dtype=np.float32)
        else:
            self.action_width = int(np.prod(source_action_space.shape))
            action_low = np.minimum(
                np.asarray(source_action_space.low, dtype=np.float32).reshape(-1),
                0.0,
            )
            action_high = np.maximum(
                np.asarray(source_action_space.high, dtype=np.float32).reshape(-1),
                0.0,
            )

        augmented_observation_space = Box(
            low=np.concatenate(
                [observation_low, np.tile(action_low, history_length)]
            ),
            high=np.concatenate(
                [observation_high, np.tile(action_high, history_length)]
            ),
            dtype=np.float32,
        )
        self.action_history = np.zeros(
            (self.num_envs, history_length, self.action_width),
            dtype=np.float32,
        )
        if self.is_vector:
            self.single_observation_space = augmented_observation_space
            self.single_action_space = source_action_space
            self.observation_space = batch_space(
                augmented_observation_space,
                self.num_envs,
            )
            self.action_space = environment.action_space
        else:
            self.observation_space = augmented_observation_space
            self.action_space = source_action_space

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the wrapped environment and the selected action histories."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observations, info = self.environment.reset(seed=seed, options=options)
        reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        if self.is_vector and options is not None and "reset_mask" in options:
            reset_mask = np.asarray(options["reset_mask"], dtype=np.bool_)
            if reset_mask.shape != (self.num_envs,):
                raise ValueError(
                    "Action history reset mask must match the number of environments"
                )
        self.action_history[reset_mask] = 0.0
        if self.is_vector:
            observations = np.stack(
                [
                    self._augment_observation(
                        np.asarray(observations)[index],
                        self.action_history[index],
                    )
                    for index in range(self.num_envs)
                ]
            )
        else:
            observations = self._augment_observation(
                observations,
                self.action_history[0],
            )
        return observations, info

    def step(
        self,
        actions: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Advance the environment and append the executed actions."""
        return self._step(actions)

    def _step(
        self,
        actions: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        action_batch = actions if self.is_vector else [actions]
        action_batch = np.asarray(action_batch)
        if action_batch.shape[0] != self.num_envs:
            raise ValueError(
                "Action history batch must match the number of environments"
            )
        encoded_actions = np.empty(
            (self.num_envs, self.action_width),
            dtype=np.float32,
        )
        for index in range(self.num_envs):
            if isinstance(self.source_action_space, Discrete):
                if not self.source_action_space.contains(action_batch[index]):
                    raise ValueError("Action is outside the configured space")
                encoded_action = np.zeros(self.action_width, dtype=np.float32)
                action_index = (
                    int(action_batch[index]) - int(self.source_action_space.start)
                )
                encoded_action[action_index] = 1.0
            else:
                action = np.asarray(action_batch[index])
                if not self.source_action_space.contains(action):
                    raise ValueError("Action is outside the configured space")
                encoded_action = action.astype(np.float32).reshape(-1)
            encoded_actions[index] = encoded_action

        observations, rewards, terminated, truncated, info = (
            self.environment.step(actions)
        )
        self.action_history[:, :-1] = self.action_history[:, 1:]
        self.action_history[:, -1] = encoded_actions
        if not self.is_vector:
            return (
                self._augment_observation(observations, self.action_history[0]),
                rewards,
                terminated,
                truncated,
                info,
            )

        completed_history = self.action_history.copy()
        done = np.logical_or(terminated, truncated)
        if "final_obs" in info:
            final_observations = np.asarray(info["final_obs"], dtype=object).copy()
            final_mask = np.asarray(
                info.get("_final_obs", done),
                dtype=np.bool_,
            )
            for index in np.flatnonzero(final_mask):
                final_observations[int(index)] = self._augment_observation(
                    final_observations[int(index)],
                    completed_history[int(index)],
                )
            info = dict(info)
            info["final_obs"] = final_observations
        self.action_history[done] = 0.0
        observations = np.stack(
            [
                self._augment_observation(
                    np.asarray(observations)[index],
                    self.action_history[index],
                )
                for index in range(self.num_envs)
            ]
        )
        return observations, rewards, terminated, truncated, info

    def _augment_observation(
        self,
        observation: Any,
        action_history: np.ndarray,
    ) -> np.ndarray:
        if isinstance(self.source_observation_space, Discrete):
            if not self.source_observation_space.contains(observation):
                raise ValueError("Observation is outside the configured space")
            encoded_observation = np.zeros(
                self.source_observation_space.n,
                dtype=np.float32,
            )
            observation_index = (
                int(observation) - int(self.source_observation_space.start)
            )
            encoded_observation[observation_index] = 1.0
        else:
            source_observation = np.asarray(observation)
            if not self.source_observation_space.contains(source_observation):
                raise ValueError("Observation is outside the configured space")
            encoded_observation = np.asarray(
                source_observation,
                dtype=np.float32,
            ).reshape(-1)
        return np.concatenate(
            [encoded_observation, action_history.reshape(-1)]
        ).astype(np.float32, copy=False)

    def render(self) -> Any:
        """Render the wrapped environment."""
        return self._render()

    def _render(self) -> Any:
        return self.environment.render()

    def close(self) -> None:
        """Close the wrapped environment."""
        self._close()

    def _close(self) -> None:
        self.environment.close()
