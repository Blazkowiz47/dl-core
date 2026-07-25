"""Preallocated vector rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    """Flattened tensor rollout with returns and generalized advantages."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    """Fixed-capacity on-policy storage for independent environment streams."""

    def __init__(self, capacity: int = 2048, num_envs: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("Rollout capacity must be positive")
        if num_envs <= 0:
            raise ValueError("Rollout num_envs must be positive")
        self.capacity = capacity
        self.num_envs = num_envs
        self.position = 0
        self.observations: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.rewards = np.empty((capacity, num_envs), dtype=np.float32)
        self.values = np.empty((capacity, num_envs), dtype=np.float32)
        self.log_probabilities = np.empty((capacity, num_envs), dtype=np.float32)
        self.next_values = np.empty((capacity, num_envs), dtype=np.float32)
        self.terminated = np.empty((capacity, num_envs), dtype=np.bool_)
        self.truncated = np.empty((capacity, num_envs), dtype=np.bool_)

    def __len__(self) -> int:
        """Return the number of collected time steps per environment."""
        return self.position

    def add(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        value: float,
        log_probability: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one scalar-environment policy transition."""
        self._add(
            observation=observation,
            action=action,
            reward=reward,
            value=value,
            log_probability=log_probability,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

    def _add(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        value: float,
        log_probability: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.num_envs != 1:
            raise RuntimeError("Use add_batch when num_envs is greater than one")
        self._add_batch(
            observations=np.expand_dims(np.asarray(observation), 0),
            actions=np.expand_dims(np.asarray(action), 0),
            rewards=np.asarray([reward], dtype=np.float32),
            values=np.asarray([value], dtype=np.float32),
            log_probabilities=np.asarray([log_probability], dtype=np.float32),
            next_values=np.asarray([next_value], dtype=np.float32),
            terminated=np.asarray([terminated], dtype=np.bool_),
            truncated=np.asarray([truncated], dtype=np.bool_),
        )

    def add_batch(
        self,
        *,
        observations: Any,
        actions: Any,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probabilities: np.ndarray,
        next_values: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Append one synchronized time step across all environment streams."""
        self._add_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probabilities=log_probabilities,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
        )

    def _add_batch(
        self,
        *,
        observations: Any,
        actions: Any,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probabilities: np.ndarray,
        next_values: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        if self.position >= self.capacity:
            raise RuntimeError("Rollout buffer is full")
        observation_array = np.asarray(observations)
        action_array = np.asarray(actions)
        expected_prefix = (self.num_envs,)
        fields = {
            "observations": observation_array,
            "actions": action_array,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "log_probabilities": np.asarray(
                log_probabilities,
                dtype=np.float32,
            ),
            "next_values": np.asarray(next_values, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "truncated": np.asarray(truncated, dtype=np.bool_),
        }
        for name, field in fields.items():
            if field.shape[:1] != expected_prefix:
                raise ValueError(
                    f"Rollout {name} must start with shape ({self.num_envs},)"
                )
        for name in (
            "rewards",
            "values",
            "log_probabilities",
            "next_values",
            "terminated",
            "truncated",
        ):
            if fields[name].shape != expected_prefix:
                raise ValueError(f"Rollout {name} must have shape {expected_prefix}")
        if self.observations is None:
            self.observations = np.empty(
                (self.capacity, *observation_array.shape),
                dtype=observation_array.dtype,
            )
            self.actions = np.empty(
                (self.capacity, *action_array.shape),
                dtype=action_array.dtype,
            )
        if observation_array.shape != self.observations.shape[1:]:
            raise ValueError("Rollout observation shape changed during collection")
        if self.actions is None or action_array.shape != self.actions.shape[1:]:
            raise ValueError("Rollout action shape changed during collection")
        self.observations[self.position] = observation_array
        self.actions[self.position] = action_array
        self.rewards[self.position] = fields["rewards"]
        self.values[self.position] = fields["values"]
        self.log_probabilities[self.position] = fields["log_probabilities"]
        self.next_values[self.position] = fields["next_values"]
        self.terminated[self.position] = fields["terminated"]
        self.truncated[self.position] = fields["truncated"]
        self.position += 1

    def compute_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        """Compute per-stream GAE and flatten time/environment for updates."""
        return self._compute_batch(
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
        )

    def _compute_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        if self.position == 0 or self.observations is None or self.actions is None:
            raise RuntimeError("Cannot compute an empty rollout")
        rewards = self.rewards[: self.position]
        values = self.values[: self.position]
        next_values = self.next_values[: self.position]
        terminated = self.terminated[: self.position]
        truncated = self.truncated[: self.position]
        advantages = np.empty_like(rewards)
        running_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for index in range(self.position - 1, -1, -1):
            bootstrap_mask = (~terminated[index]).astype(np.float32)
            continuation_mask = (~np.logical_or(
                terminated[index],
                truncated[index],
            )).astype(np.float32)
            delta = (
                rewards[index]
                + gamma * bootstrap_mask * next_values[index]
                - values[index]
            )
            running_advantage = delta + (
                gamma
                * gae_lambda
                * continuation_mask
                * running_advantage
            )
            advantages[index] = running_advantage
        returns = advantages + values
        observations = self.observations[: self.position]
        actions = self.actions[: self.position]
        return RolloutBatch(
            observations=torch.as_tensor(
                observations.reshape(-1, *observations.shape[2:]),
                device=device,
            ),
            actions=torch.as_tensor(
                actions.reshape(-1, *actions.shape[2:]),
                device=device,
            ),
            old_log_probabilities=torch.as_tensor(
                self.log_probabilities[: self.position].reshape(-1),
                device=device,
            ),
            old_values=torch.as_tensor(values.reshape(-1), device=device),
            returns=torch.as_tensor(returns.reshape(-1), device=device),
            advantages=torch.as_tensor(advantages.reshape(-1), device=device),
        )

    def clear(self) -> None:
        """Remove all collected transitions without reallocating storage."""
        self._clear()

    def _clear(self) -> None:
        self.position = 0

    def state_dict(self) -> dict[str, Any]:
        """Return a copy of a potentially partial vector rollout."""
        return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "position": self.position,
            "observations": (
                None
                if self.position == 0
                else self.observations[: self.position].copy()
            ),
            "actions": (
                None
                if self.position == 0
                else self.actions[: self.position].copy()
            ),
            "rewards": self.rewards[: self.position].copy(),
            "values": self.values[: self.position].copy(),
            "log_probabilities": self.log_probabilities[: self.position].copy(),
            "next_values": self.next_values[: self.position].copy(),
            "terminated": self.terminated[: self.position].copy(),
            "truncated": self.truncated[: self.position].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a partial rollout after validating its vector contract."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        if "capacity" not in state:
            field_names = (
                "observations",
                "actions",
                "rewards",
                "values",
                "log_probabilities",
                "next_values",
                "terminated",
                "truncated",
            )
            if self.num_envs != 1 or any(name not in state for name in field_names):
                raise ValueError("Legacy rollout checkpoint state is incomplete")
            lengths = {len(state[name]) for name in field_names}
            if len(lengths) != 1:
                raise ValueError(
                    "Legacy rollout checkpoint fields have inconsistent lengths"
                )
            if next(iter(lengths)) > self.capacity:
                raise ValueError("Legacy rollout exceeds configured capacity")
            if any(
                not isinstance(value, (bool, np.bool_))
                for name in ("terminated", "truncated")
                for value in state[name]
            ):
                raise ValueError(
                    "Legacy rollout boundary fields must be booleans"
                )
            scalar_values = np.asarray(
                [
                    state["rewards"],
                    state["values"],
                    state["log_probabilities"],
                    state["next_values"],
                ],
                dtype=np.float64,
            )
            if not np.isfinite(scalar_values).all():
                raise ValueError("Legacy rollout scalar fields must be finite")
            self._clear()
            for values in zip(*(state[name] for name in field_names), strict=True):
                self._add(
                    observation=values[0],
                    action=values[1],
                    reward=float(values[2]),
                    value=float(values[3]),
                    log_probability=float(values[4]),
                    next_value=float(values[5]),
                    terminated=bool(values[6]),
                    truncated=bool(values[7]),
                )
            return
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Rollout capacity does not match configuration")
        if int(state.get("num_envs", -1)) != self.num_envs:
            raise ValueError("Rollout num_envs does not match configuration")
        position = int(state.get("position", -1))
        if not 0 <= position <= self.capacity:
            raise ValueError("Rollout position is invalid")
        observation_values = state.get("observations")
        action_values = state.get("actions")
        fields = {
            "rewards": np.asarray(state.get("rewards")),
            "values": np.asarray(state.get("values")),
            "log_probabilities": np.asarray(state.get("log_probabilities")),
            "next_values": np.asarray(state.get("next_values")),
            "terminated": np.asarray(state.get("terminated")),
            "truncated": np.asarray(state.get("truncated")),
        }
        if position == 0:
            if observation_values is not None or action_values is not None:
                raise ValueError("Empty rollout must not contain observations or actions")
            expected_shape = (0, self.num_envs)
            if any(field.shape != expected_shape for field in fields.values()):
                raise ValueError("Empty rollout scalar field shapes are invalid")
            if fields["terminated"].dtype != np.bool_:
                raise ValueError("Rollout terminated values must be boolean")
            if fields["truncated"].dtype != np.bool_:
                raise ValueError("Rollout truncated values must be boolean")
            self.position = 0
            return
        observations = np.asarray(observation_values)
        actions = np.asarray(action_values)
        if observations.shape[:2] != (position, self.num_envs):
            raise ValueError("Rollout observation shape is invalid")
        if actions.shape[:2] != (position, self.num_envs):
            raise ValueError("Rollout action shape is invalid")
        expected_shape = (position, self.num_envs)
        if any(field.shape != expected_shape for field in fields.values()):
            raise ValueError("Rollout scalar field shapes are invalid")
        if not np.isfinite(
            np.stack(
                [
                    fields["rewards"],
                    fields["values"],
                    fields["log_probabilities"],
                    fields["next_values"],
                ]
            )
        ).all():
            raise ValueError("Rollout scalar fields must be finite")
        if fields["terminated"].dtype != np.bool_:
            raise ValueError("Rollout terminated values must be boolean")
        if fields["truncated"].dtype != np.bool_:
            raise ValueError("Rollout truncated values must be boolean")
        self.observations = np.empty(
            (self.capacity, *observations.shape[1:]),
            dtype=observations.dtype,
        )
        self.actions = np.empty(
            (self.capacity, *actions.shape[1:]),
            dtype=actions.dtype,
        )
        self.observations[:position] = observations
        self.actions[:position] = actions
        self.rewards[:position] = fields["rewards"]
        self.values[:position] = fields["values"]
        self.log_probabilities[:position] = fields["log_probabilities"]
        self.next_values[:position] = fields["next_values"]
        self.terminated[:position] = fields["terminated"]
        self.truncated[:position] = fields["truncated"]
        self.position = position
