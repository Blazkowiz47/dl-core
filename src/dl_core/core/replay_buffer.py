"""Bounded replay storage shared by off-policy reinforcement-learning trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rl_types import Transition, TransitionBatch


@dataclass(slots=True)
class ReplayBatch:
    """Tensor batch sampled from a replay buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
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
        gamma: float = 0.99,
        n_step: int = 1,
        seed: int = 42,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Replay buffer gamma must be in [0, 1]")
        if n_step <= 0:
            raise ValueError("Replay buffer n_step must be positive")
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observation_dtype = np.dtype(observation_dtype)
        self.action_dtype = np.dtype(action_dtype)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.observations = np.empty(
            (capacity, *observation_shape),
            dtype=self.observation_dtype,
        )
        self.actions = np.empty((capacity, *action_shape), dtype=self.action_dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, *observation_shape),
            dtype=self.observation_dtype,
        )
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.random_generator = np.random.default_rng(seed)
        self.pending_transitions: list[list[Transition[Any, Any]]] = []

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition[Any, Any]) -> int:
        """Append one transition and return the number of matured entries."""
        return self._add(transition)

    def _add(self, transition: Transition[Any, Any]) -> int:
        added_per_environment = self._add_batch(
            TransitionBatch(
                observations=np.expand_dims(
                    np.asarray(transition.observation),
                    0,
                ),
                actions=np.expand_dims(np.asarray(transition.action), 0),
                rewards=np.asarray([transition.reward], dtype=np.float32),
                next_observations=np.expand_dims(
                    np.asarray(transition.next_observation),
                    0,
                ),
                terminated=np.asarray(
                    [transition.terminated],
                    dtype=np.bool_,
                ),
                truncated=np.asarray(
                    [transition.truncated],
                    dtype=np.bool_,
                ),
            )
        )
        return int(added_per_environment[0])

    def add_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> np.ndarray:
        """Append one vector step and return matured entries per lane."""
        return self._add_batch(transitions)

    def _add_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> np.ndarray:
        observations = np.asarray(transitions.observations)
        actions = np.asarray(transitions.actions)
        rewards = np.asarray(transitions.rewards)
        next_observations = np.asarray(transitions.next_observations)
        terminated = np.asarray(transitions.terminated)
        truncated = np.asarray(transitions.truncated)
        batch_size = int(rewards.shape[0])
        expected_shapes = {
            "observations": (batch_size, *self.observation_shape),
            "actions": (batch_size, *self.action_shape),
            "rewards": (batch_size,),
            "next_observations": (batch_size, *self.observation_shape),
            "terminated": (batch_size,),
            "truncated": (batch_size,),
        }
        values = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminated": terminated,
            "truncated": truncated,
        }
        for name, expected_shape in expected_shapes.items():
            if values[name].shape != expected_shape:
                raise ValueError(
                    f"Replay batch {name} shape {values[name].shape} does not "
                    f"match {expected_shape}"
                )
        added_per_environment = np.ones(batch_size, dtype=np.int64)
        if batch_size == 0:
            return added_per_environment
        discounts = np.full(batch_size, self.gamma, dtype=np.float32)
        if self.n_step > 1:
            added_per_environment.fill(0)
            if len(self.pending_transitions) != batch_size:
                if any(self.pending_transitions):
                    raise ValueError(
                        "N-step replay requires a stable environment batch size"
                    )
                self.pending_transitions = [[] for _ in range(batch_size)]

            stored_observations: list[Any] = []
            stored_actions: list[Any] = []
            stored_rewards: list[float] = []
            stored_discounts: list[float] = []
            stored_next_observations: list[Any] = []
            stored_terminated: list[bool] = []
            stored_truncated: list[bool] = []
            for environment_index in range(batch_size):
                previous_stored_size = len(stored_rewards)
                pending = self.pending_transitions[environment_index]
                pending.append(
                    Transition(
                        observation=np.asarray(
                            observations[environment_index]
                        ).copy(),
                        action=np.asarray(actions[environment_index]).copy(),
                        reward=float(rewards[environment_index]),
                        next_observation=np.asarray(
                            next_observations[environment_index]
                        ).copy(),
                        terminated=bool(terminated[environment_index]),
                        truncated=bool(truncated[environment_index]),
                    )
                )
                episode_done = bool(
                    terminated[environment_index]
                    or truncated[environment_index]
                )
                while len(pending) >= self.n_step or (
                    episode_done and pending
                ):
                    rollout_length = min(self.n_step, len(pending))
                    first_transition = pending[0]
                    final_transition = pending[rollout_length - 1]
                    stored_observations.append(first_transition.observation)
                    stored_actions.append(first_transition.action)
                    stored_rewards.append(
                        float(
                            sum(
                                (self.gamma**offset)
                                * pending[offset].reward
                                for offset in range(rollout_length)
                            )
                        )
                    )
                    stored_discounts.append(self.gamma**rollout_length)
                    stored_next_observations.append(
                        final_transition.next_observation
                    )
                    stored_terminated.append(final_transition.terminated)
                    stored_truncated.append(final_transition.truncated)
                    pending.pop(0)
                    if not episode_done:
                        break
                added_per_environment[environment_index] = (
                    len(stored_rewards) - previous_stored_size
                )

            if not stored_rewards:
                return added_per_environment
            observations = np.asarray(stored_observations)
            actions = np.asarray(stored_actions)
            rewards = np.asarray(stored_rewards, dtype=np.float32)
            discounts = np.asarray(stored_discounts, dtype=np.float32)
            next_observations = np.asarray(stored_next_observations)
            terminated = np.asarray(stored_terminated, dtype=np.bool_)
            truncated = np.asarray(stored_truncated, dtype=np.bool_)
            batch_size = len(stored_rewards)

        original_batch_size = batch_size
        if batch_size >= self.capacity:
            start = batch_size - self.capacity
            observations = observations[start:]
            actions = actions[start:]
            rewards = rewards[start:]
            discounts = discounts[start:]
            next_observations = next_observations[start:]
            terminated = terminated[start:]
            truncated = truncated[start:]
            batch_size = self.capacity
            write_position = (
                self.position + original_batch_size - self.capacity
            ) % self.capacity
        else:
            write_position = self.position
        indices = (write_position + np.arange(batch_size)) % self.capacity
        self.observations[indices] = observations.astype(
            self.observation_dtype,
            copy=False,
        )
        self.actions[indices] = actions.astype(self.action_dtype, copy=False)
        self.rewards[indices] = rewards.astype(np.float32, copy=False)
        self.discounts[indices] = discounts.astype(np.float32, copy=False)
        self.next_observations[indices] = next_observations.astype(
            self.observation_dtype,
            copy=False,
        )
        self.terminated[indices] = terminated.astype(np.bool_, copy=False)
        self.truncated[indices] = truncated.astype(np.bool_, copy=False)
        self.position = (
            self.position + original_batch_size
        ) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        return added_per_environment

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
            discounts=torch.as_tensor(self.discounts[indices], device=device),
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
            "gamma": self.gamma,
            "n_step": self.n_step,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "discounts": self.discounts[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "random_generator_state": self.random_generator.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay contents after validating its storage contract."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        legacy_state = "n_step" not in state
        expected = {
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "observation_dtype": self.observation_dtype.str,
            "action_dtype": self.action_dtype.str,
            "gamma": self.gamma,
            "n_step": self.n_step,
        }
        for key, value in expected.items():
            saved_value = state.get(key)
            if legacy_state and key == "gamma":
                saved_value = self.gamma
            if legacy_state and key == "n_step":
                saved_value = 1
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
            "discounts": self.discounts,
            "next_observations": self.next_observations,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        for key, destination in arrays.items():
            if key == "discounts" and legacy_state:
                source = np.full(size, self.gamma, dtype=np.float32)
            else:
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
        self.pending_transitions = []
        self.random_generator.bit_generator.state = generator_state
