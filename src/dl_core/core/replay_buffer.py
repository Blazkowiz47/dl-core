"""Bounded replay storage shared by off-policy reinforcement-learning trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rl_types import Transition


@dataclass(slots=True)
class ReplayBatch:
    """Tensor batch sampled from a replay buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity NumPy replay buffer with reproducible sampling."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        observation_dtype: np.dtype[Any] | type[np.generic] = np.float32,
        action_dtype: np.dtype[Any] | type[np.generic] = np.float32,
        seed: int = 42,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observation_dtype = np.dtype(observation_dtype)
        self.action_dtype = np.dtype(action_dtype)
        self.observations = np.empty(
            (capacity, *observation_shape),
            dtype=self.observation_dtype,
        )
        self.actions = np.empty((capacity, *action_shape), dtype=self.action_dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, *observation_shape),
            dtype=self.observation_dtype,
        )
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.random_generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition[Any, Any]) -> None:
        """Append one transition, replacing the oldest entry when full."""
        self._add(transition)

    def _add(self, transition: Transition[Any, Any]) -> None:
        self.observations[self.position] = np.asarray(
            transition.observation,
            dtype=self.observation_dtype,
        )
        self.actions[self.position] = np.asarray(
            transition.action,
            dtype=self.action_dtype,
        )
        self.rewards[self.position] = transition.reward
        self.next_observations[self.position] = np.asarray(
            transition.next_observation,
            dtype=self.observation_dtype,
        )
        self.terminated[self.position] = transition.terminated
        self.truncated[self.position] = transition.truncated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample a tensor batch uniformly with replacement."""
        return self._sample(batch_size, device)

    def _sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("Replay batch size must be positive")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        indices = self.random_generator.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                device=device,
            ),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
            truncated=torch.as_tensor(self.truncated[indices], device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return replay contents, cursor, and sampling-generator state."""
        return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "observation_dtype": self.observation_dtype.str,
            "action_dtype": self.action_dtype.str,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "random_generator_state": self.random_generator.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay contents after validating its storage contract."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "observation_dtype": self.observation_dtype.str,
            "action_dtype": self.action_dtype.str,
        }
        for key, value in expected.items():
            saved_value = state.get(key)
            if key in {"observation_shape", "action_shape"}:
                saved_value = tuple(saved_value or ())
            if saved_value != value:
                raise ValueError(f"Replay buffer {key} does not match configuration")

        size = int(state.get("size", -1))
        position = int(state.get("position", -1))
        if not 0 <= size <= self.capacity:
            raise ValueError("Replay buffer size is invalid")
        if not 0 <= position < self.capacity:
            raise ValueError("Replay buffer position is invalid")
        if size < self.capacity and position != size:
            raise ValueError("Replay buffer position is inconsistent with its size")
        arrays = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        for key, destination in arrays.items():
            source = np.asarray(state.get(key))
            if source.shape != destination[:size].shape:
                raise ValueError(f"Replay buffer {key} shape is invalid")
            if source.dtype != destination.dtype:
                raise ValueError(f"Replay buffer {key} dtype is invalid")
            destination[:size] = source
        generator_state = state.get("random_generator_state")
        if not isinstance(generator_state, dict):
            raise ValueError("Replay buffer random generator state is invalid")
        self.size = size
        self.position = position
        self.random_generator.bit_generator.state = generator_state
