"""Episode-safe sequence replay for recurrent reinforcement-learning agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rl_types import Transition, TransitionBatch


@dataclass(slots=True)
class SequenceBatch:
    """Fixed-length replay sequences with one bootstrap observation."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    is_first: torch.Tensor
    episode_ids: torch.Tensor
    start_steps: torch.Tensor
    sample_ages: torch.Tensor
    burn_in: int


@dataclass(slots=True)
class SequenceReplayAddResult:
    """Sequence-window changes across one ordered vector insertion."""

    added_per_environment: np.ndarray
    available_after_environment: np.ndarray


class SequenceReplayBuffer:
    """Fixed-capacity replay that samples within individual episodes."""

    def __init__(
        self,
        capacity: int,
        num_environments: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        sequence_length: int,
        burn_in: int = 0,
        observation_dtype: np.dtype[Any] | type[np.generic] = np.float32,
        action_dtype: np.dtype[Any] | type[np.generic] = np.float32,
        seed: int = 42,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Sequence replay capacity must be positive")
        if num_environments <= 0:
            raise ValueError("Sequence replay environment count must be positive")
        if capacity < num_environments:
            raise ValueError(
                "Sequence replay capacity must fit every environment lane"
            )
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        if burn_in < 0:
            raise ValueError("Sequence burn-in cannot be negative")

        self.capacity = int(capacity)
        self.num_environments = int(num_environments)
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.total_sequence_length = self.sequence_length + self.burn_in
        self.observation_dtype = np.dtype(observation_dtype)
        self.action_dtype = np.dtype(action_dtype)

        base_capacity, extra_lanes = divmod(capacity, num_environments)
        self.lane_capacities = np.asarray(
            [
                base_capacity + int(index < extra_lanes)
                for index in range(num_environments)
            ],
            dtype=np.int64,
        )
        if int(self.lane_capacities.min()) < self.total_sequence_length:
            raise ValueError(
                "Sequence replay capacity must hold sequence_length + burn_in "
                "transitions in every environment lane"
            )
        self.observations = [
            np.empty(
                (lane_capacity, *self.observation_shape),
                dtype=self.observation_dtype,
            )
            for lane_capacity in self.lane_capacities
        ]
        self.actions = [
            np.empty(
                (lane_capacity, *self.action_shape),
                dtype=self.action_dtype,
            )
            for lane_capacity in self.lane_capacities
        ]
        self.rewards = [
            np.empty(lane_capacity, dtype=np.float32)
            for lane_capacity in self.lane_capacities
        ]
        self.next_observations = [
            np.empty(
                (lane_capacity, *self.observation_shape),
                dtype=self.observation_dtype,
            )
            for lane_capacity in self.lane_capacities
        ]
        self.terminated = [
            np.empty(lane_capacity, dtype=np.bool_)
            for lane_capacity in self.lane_capacities
        ]
        self.truncated = [
            np.empty(lane_capacity, dtype=np.bool_)
            for lane_capacity in self.lane_capacities
        ]
        self.episode_ids = [
            np.empty(lane_capacity, dtype=np.int64)
            for lane_capacity in self.lane_capacities
        ]
        self.episode_steps = [
            np.empty(lane_capacity, dtype=np.int64)
            for lane_capacity in self.lane_capacities
        ]
        self.insertion_ids = [
            np.empty(lane_capacity, dtype=np.int64)
            for lane_capacity in self.lane_capacities
        ]

        self.lane_sizes = np.zeros(num_environments, dtype=np.int64)
        self.lane_total_added = np.zeros(num_environments, dtype=np.int64)
        self.current_episode_ids = np.arange(
            num_environments,
            dtype=np.int64,
        )
        self.current_episode_steps = np.zeros(
            num_environments,
            dtype=np.int64,
        )
        self.next_episode_id = num_environments
        self.insertion_count = 0
        self.candidate_starts: list[list[int]] = [
            [] for _ in range(num_environments)
        ]
        self.candidate_offsets = np.zeros(num_environments, dtype=np.int64)
        self.random_generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.lane_sizes.sum())

    @property
    def num_sequences(self) -> int:
        """Return the number of currently sampleable sequence windows."""
        return int(
            sum(
                len(starts) - int(self.candidate_offsets[index])
                for index, starts in enumerate(self.candidate_starts)
            )
        )

    def add(
        self,
        transition: Transition[Any, Any],
        *,
        environment_index: int = 0,
    ) -> int:
        """Append one transition and return the number of new sequence windows."""
        return self._add(
            transition,
            environment_index=environment_index,
        )

    def _add(
        self,
        transition: Transition[Any, Any],
        *,
        environment_index: int,
    ) -> int:
        if not 0 <= environment_index < self.num_environments:
            raise IndexError("Sequence replay environment index is out of range")
        lane_capacity = int(self.lane_capacities[environment_index])
        serial = int(self.lane_total_added[environment_index])
        position = serial % lane_capacity
        self.observations[environment_index][position] = np.asarray(
            transition.observation,
            dtype=self.observation_dtype,
        )
        self.actions[environment_index][position] = np.asarray(
            transition.action,
            dtype=self.action_dtype,
        )
        self.rewards[environment_index][position] = float(transition.reward)
        self.next_observations[environment_index][position] = np.asarray(
            transition.next_observation,
            dtype=self.observation_dtype,
        )
        self.terminated[environment_index][position] = bool(
            transition.terminated
        )
        self.truncated[environment_index][position] = bool(
            transition.truncated
        )
        self.episode_ids[environment_index][position] = (
            self.current_episode_ids[environment_index]
        )
        self.episode_steps[environment_index][position] = (
            self.current_episode_steps[environment_index]
        )
        self.insertion_ids[environment_index][position] = self.insertion_count

        sequence_added = int(
            int(self.current_episode_steps[environment_index]) + 1
            >= self.total_sequence_length
        )
        if sequence_added:
            self.candidate_starts[environment_index].append(
                serial - self.total_sequence_length + 1
            )

        self.lane_total_added[environment_index] += 1
        self.lane_sizes[environment_index] = min(
            int(self.lane_sizes[environment_index]) + 1,
            lane_capacity,
        )
        self.insertion_count += 1

        earliest_serial = (
            int(self.lane_total_added[environment_index]) - lane_capacity
        )
        starts = self.candidate_starts[environment_index]
        offset = int(self.candidate_offsets[environment_index])
        while offset < len(starts) and starts[offset] < earliest_serial:
            offset += 1
        if offset > 1024 and offset * 2 > len(starts):
            del starts[:offset]
            offset = 0
        self.candidate_offsets[environment_index] = offset

        if transition.terminated or transition.truncated:
            self.current_episode_ids[environment_index] = self.next_episode_id
            self.current_episode_steps[environment_index] = 0
            self.next_episode_id += 1
        else:
            self.current_episode_steps[environment_index] += 1
        return sequence_added

    def add_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> SequenceReplayAddResult:
        """Append a vector step and report per-lane window availability."""
        return self._add_batch(transitions)

    def _add_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> SequenceReplayAddResult:
        observations = np.asarray(transitions.observations)
        actions = np.asarray(transitions.actions)
        rewards = np.asarray(transitions.rewards)
        next_observations = np.asarray(transitions.next_observations)
        terminated = np.asarray(transitions.terminated)
        truncated = np.asarray(transitions.truncated)
        expected_shapes = {
            "observations": (
                self.num_environments,
                *self.observation_shape,
            ),
            "actions": (self.num_environments, *self.action_shape),
            "rewards": (self.num_environments,),
            "next_observations": (
                self.num_environments,
                *self.observation_shape,
            ),
            "terminated": (self.num_environments,),
            "truncated": (self.num_environments,),
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
                    f"Sequence replay batch {name} shape "
                    f"{values[name].shape} does not match {expected_shape}"
                )

        added_sequences = np.zeros(
            self.num_environments,
            dtype=np.int64,
        )
        available_sequences = np.zeros(
            self.num_environments,
            dtype=np.int64,
        )
        total_available = self.num_sequences
        for environment_index in range(self.num_environments):
            previous_lane_count = (
                len(self.candidate_starts[environment_index])
                - int(self.candidate_offsets[environment_index])
            )
            added_sequences[environment_index] = self._add(
                Transition(
                    observation=observations[environment_index],
                    action=actions[environment_index],
                    reward=float(rewards[environment_index]),
                    next_observation=next_observations[environment_index],
                    terminated=bool(terminated[environment_index]),
                    truncated=bool(truncated[environment_index]),
                ),
                environment_index=environment_index,
            )
            current_lane_count = (
                len(self.candidate_starts[environment_index])
                - int(self.candidate_offsets[environment_index])
            )
            total_available += current_lane_count - previous_lane_count
            available_sequences[environment_index] = total_available
        return SequenceReplayAddResult(
            added_per_environment=added_sequences,
            available_after_environment=available_sequences,
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> SequenceBatch:
        """Sample episode-safe sequences uniformly with replacement."""
        return self._sample(batch_size, device)

    def _sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> SequenceBatch:
        if batch_size <= 0:
            raise ValueError("Sequence replay batch size must be positive")
        counts = np.asarray(
            [
                len(starts) - int(self.candidate_offsets[index])
                for index, starts in enumerate(self.candidate_starts)
            ],
            dtype=np.int64,
        )
        total_candidates = int(counts.sum())
        if total_candidates == 0:
            raise RuntimeError(
                "Sequence replay does not contain a complete sequence"
            )
        cumulative_counts = np.cumsum(counts)
        sampled_candidates = self.random_generator.integers(
            0,
            total_candidates,
            size=batch_size,
        )

        observation_sequences: list[np.ndarray] = []
        action_sequences: list[np.ndarray] = []
        reward_sequences: list[np.ndarray] = []
        terminated_sequences: list[np.ndarray] = []
        truncated_sequences: list[np.ndarray] = []
        first_sequences: list[np.ndarray] = []
        sampled_episode_ids: list[int] = []
        sampled_start_steps: list[int] = []
        sampled_ages: list[int] = []
        for candidate in sampled_candidates:
            environment_index = int(
                np.searchsorted(
                    cumulative_counts,
                    candidate,
                    side="right",
                )
            )
            previous_count = (
                int(cumulative_counts[environment_index - 1])
                if environment_index > 0
                else 0
            )
            local_candidate = int(candidate) - previous_count
            start_serial = self.candidate_starts[environment_index][
                int(self.candidate_offsets[environment_index])
                + local_candidate
            ]
            lane_capacity = int(self.lane_capacities[environment_index])
            indices = (
                start_serial + np.arange(self.total_sequence_length)
            ) % lane_capacity

            observation_sequences.append(
                np.stack(
                    [
                        self.observations[environment_index][indices[0]],
                        *self.next_observations[environment_index][indices],
                    ]
                )
            )
            action_sequences.append(
                self.actions[environment_index][indices].copy()
            )
            reward_sequences.append(
                self.rewards[environment_index][indices].copy()
            )
            terminated_sequences.append(
                self.terminated[environment_index][indices].copy()
            )
            truncated_sequences.append(
                self.truncated[environment_index][indices].copy()
            )
            is_first = np.zeros(
                self.total_sequence_length + 1,
                dtype=np.bool_,
            )
            is_first[0] = (
                self.episode_steps[environment_index][indices[0]] == 0
            )
            first_sequences.append(is_first)
            sampled_episode_ids.append(
                int(self.episode_ids[environment_index][indices[0]])
            )
            sampled_start_steps.append(
                int(self.episode_steps[environment_index][indices[0]])
            )
            sampled_ages.append(
                self.insertion_count
                - 1
                - int(self.insertion_ids[environment_index][indices[-1]])
            )

        return SequenceBatch(
            observations=torch.as_tensor(
                np.stack(observation_sequences),
                device=device,
            ),
            actions=torch.as_tensor(
                np.stack(action_sequences),
                device=device,
            ),
            rewards=torch.as_tensor(
                np.stack(reward_sequences),
                device=device,
            ),
            terminated=torch.as_tensor(
                np.stack(terminated_sequences),
                device=device,
            ),
            truncated=torch.as_tensor(
                np.stack(truncated_sequences),
                device=device,
            ),
            is_first=torch.as_tensor(
                np.stack(first_sequences),
                device=device,
            ),
            episode_ids=torch.as_tensor(
                sampled_episode_ids,
                dtype=torch.long,
                device=device,
            ),
            start_steps=torch.as_tensor(
                sampled_start_steps,
                dtype=torch.long,
                device=device,
            ),
            sample_ages=torch.as_tensor(
                sampled_ages,
                dtype=torch.long,
                device=device,
            ),
            burn_in=self.burn_in,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return replay contents, episode cursors, and sampling state."""
        return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        arrays: dict[str, list[np.ndarray]] = {}
        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
            "episode_ids",
            "episode_steps",
            "insertion_ids",
        ):
            lane_arrays = getattr(self, name)
            arrays[name] = [
                (
                    values.copy()
                    if int(self.lane_sizes[index])
                    == int(self.lane_capacities[index])
                    else values[: int(self.lane_sizes[index])].copy()
                )
                for index, values in enumerate(lane_arrays)
            ]
        return {
            "capacity": self.capacity,
            "num_environments": self.num_environments,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "sequence_length": self.sequence_length,
            "burn_in": self.burn_in,
            "observation_dtype": self.observation_dtype.str,
            "action_dtype": self.action_dtype.str,
            "lane_capacities": self.lane_capacities.copy(),
            "lane_sizes": self.lane_sizes.copy(),
            "lane_total_added": self.lane_total_added.copy(),
            "current_episode_ids": self.current_episode_ids.copy(),
            "current_episode_steps": self.current_episode_steps.copy(),
            "next_episode_id": self.next_episode_id,
            "insertion_count": self.insertion_count,
            "random_generator_state": self.random_generator.bit_generator.state,
            **arrays,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay state after validating its storage contract."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "capacity": self.capacity,
            "num_environments": self.num_environments,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "sequence_length": self.sequence_length,
            "burn_in": self.burn_in,
            "observation_dtype": self.observation_dtype.str,
            "action_dtype": self.action_dtype.str,
        }
        for key, expected_value in expected.items():
            saved_value = state.get(key)
            if key in {"observation_shape", "action_shape"}:
                saved_value = tuple(saved_value or ())
            if saved_value != expected_value:
                raise ValueError(
                    f"Sequence replay {key} does not match configuration"
                )

        lane_capacities = np.asarray(state.get("lane_capacities"))
        lane_sizes = np.asarray(state.get("lane_sizes"))
        lane_total_added = np.asarray(state.get("lane_total_added"))
        if not np.array_equal(lane_capacities, self.lane_capacities):
            raise ValueError("Sequence replay lane capacities are invalid")
        if lane_sizes.shape != (self.num_environments,) or np.any(
            (lane_sizes < 0) | (lane_sizes > self.lane_capacities)
        ):
            raise ValueError("Sequence replay lane sizes are invalid")
        if lane_total_added.shape != (self.num_environments,) or np.any(
            lane_total_added < lane_sizes
        ):
            raise ValueError("Sequence replay lane cursors are invalid")

        validated_arrays: dict[str, list[np.ndarray]] = {}
        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
            "episode_ids",
            "episode_steps",
            "insertion_ids",
        ):
            saved_lanes = state.get(name)
            if not isinstance(saved_lanes, list) or (
                len(saved_lanes) != self.num_environments
            ):
                raise ValueError(f"Sequence replay {name} lanes are invalid")
            destination_lanes = getattr(self, name)
            validated_arrays[name] = []
            for environment_index, saved_values in enumerate(saved_lanes):
                source = np.asarray(saved_values)
                size = int(lane_sizes[environment_index])
                destination = destination_lanes[environment_index]
                expected_shape = destination.shape if (
                    size == int(self.lane_capacities[environment_index])
                ) else destination[:size].shape
                if source.shape != expected_shape:
                    raise ValueError(
                        f"Sequence replay {name} shape is invalid"
                    )
                if source.dtype != destination.dtype:
                    raise ValueError(
                        f"Sequence replay {name} dtype is invalid"
                    )
                validated_arrays[name].append(source.copy())

        current_episode_ids = np.asarray(state.get("current_episode_ids"))
        current_episode_steps = np.asarray(state.get("current_episode_steps"))
        if (
            current_episode_ids.shape != (self.num_environments,)
            or current_episode_ids.dtype != np.dtype(np.int64)
        ):
            raise ValueError("Sequence replay episode IDs are invalid")
        if (
            current_episode_steps.shape != (self.num_environments,)
            or current_episode_steps.dtype != np.dtype(np.int64)
            or np.any(current_episode_steps < 0)
        ):
            raise ValueError("Sequence replay episode steps are invalid")
        next_episode_id = int(state.get("next_episode_id", -1))
        insertion_count = int(state.get("insertion_count", -1))
        if next_episode_id < self.num_environments:
            raise ValueError("Sequence replay next episode ID is invalid")
        if insertion_count != int(lane_total_added.sum()):
            raise ValueError("Sequence replay insertion count is invalid")
        generator_state = state.get("random_generator_state")
        if not isinstance(generator_state, dict):
            raise ValueError(
                "Sequence replay random generator state is invalid"
            )

        validated_generator = np.random.default_rng()
        try:
            validated_generator.bit_generator.state = generator_state
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Sequence replay random generator state is invalid"
            ) from error

        candidate_starts: list[list[int]] = [
            [] for _ in range(self.num_environments)
        ]
        for environment_index in range(self.num_environments):
            lane_capacity = int(self.lane_capacities[environment_index])
            earliest_serial = (
                int(lane_total_added[environment_index])
                - int(lane_sizes[environment_index])
            )
            latest_start = (
                int(lane_total_added[environment_index])
                - self.total_sequence_length
            )
            for start_serial in range(earliest_serial, latest_start + 1):
                indices = (
                    start_serial
                    + np.arange(self.total_sequence_length)
                ) % lane_capacity
                episode_ids = validated_arrays["episode_ids"][
                    environment_index
                ][indices]
                episode_steps = validated_arrays["episode_steps"][
                    environment_index
                ][indices]
                if np.all(episode_ids == episode_ids[0]) and np.array_equal(
                    episode_steps,
                    episode_steps[0]
                    + np.arange(self.total_sequence_length),
                ):
                    candidate_starts[environment_index].append(start_serial)

        for name, saved_lanes in validated_arrays.items():
            destination_lanes = getattr(self, name)
            for environment_index, source in enumerate(saved_lanes):
                size = int(lane_sizes[environment_index])
                if size == int(self.lane_capacities[environment_index]):
                    destination_lanes[environment_index][:] = source
                else:
                    destination_lanes[environment_index][:size] = source
        self.lane_sizes = lane_sizes.astype(np.int64, copy=True)
        self.lane_total_added = lane_total_added.astype(np.int64, copy=True)
        self.current_episode_ids = current_episode_ids.astype(
            np.int64,
            copy=True,
        )
        self.current_episode_steps = current_episode_steps.astype(
            np.int64,
            copy=True,
        )
        self.next_episode_id = next_episode_id
        self.insertion_count = insertion_count
        self.random_generator.bit_generator.state = (
            validated_generator.bit_generator.state
        )
        self.candidate_starts = candidate_starts
        self.candidate_offsets.fill(0)
