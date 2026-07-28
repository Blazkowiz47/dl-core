"""Tests for episode-safe recurrent sequence replay."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_core.core import SequenceReplayBuffer, Transition, TransitionBatch


def _transition(
    step: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition[np.ndarray, int]:
    return Transition(
        observation=np.asarray([step], dtype=np.float32),
        action=step % 2,
        reward=float(step),
        next_observation=np.asarray([step + 1], dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )


def test_sequence_replay_samples_only_complete_episode_windows() -> None:
    buffer = SequenceReplayBuffer(
        capacity=12,
        num_environments=1,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=3,
        burn_in=1,
        action_dtype=np.int64,
        seed=7,
    )
    for step in range(3):
        buffer.add(_transition(step, terminated=step == 2))
    for step in range(10, 15):
        buffer.add(_transition(step, truncated=step == 14))

    assert len(buffer) == 8
    assert buffer.num_sequences == 2

    batch = buffer.sample(64, torch.device("cpu"))

    assert batch.observations.shape == (64, 5, 1)
    assert batch.actions.shape == (64, 4)
    assert batch.rewards.shape == (64, 4)
    assert batch.burn_in == 1
    assert torch.all(batch.episode_ids == 1)
    assert set(batch.start_steps.tolist()) <= {0, 1}
    assert torch.equal(
        batch.is_first[:, 0],
        batch.start_steps == 0,
    )
    assert not torch.any(batch.is_first[:, 1:])
    assert torch.all(
        batch.observations[:, 1:] - batch.observations[:, :-1] == 1
    )
    assert not torch.any(batch.terminated)
    assert not torch.any(batch.truncated[:, :-1])


def test_sequence_replay_tracks_vector_episode_boundaries_independently() -> None:
    buffer = SequenceReplayBuffer(
        capacity=12,
        num_environments=2,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
        seed=11,
    )
    added_windows: list[list[int]] = []
    for step in range(4):
        added_windows.append(
            buffer.add_batch(
                TransitionBatch(
                    observations=np.asarray(
                        [[step], [100 + step]],
                        dtype=np.float32,
                    ),
                    actions=np.asarray([0, 1], dtype=np.int64),
                    rewards=np.asarray([0.0, 1.0], dtype=np.float32),
                    next_observations=np.asarray(
                        [[step + 1], [101 + step]],
                        dtype=np.float32,
                    ),
                    terminated=np.asarray(
                        [step == 0, False],
                        dtype=np.bool_,
                    ),
                    truncated=np.asarray([False, False], dtype=np.bool_),
                )
            ).added_per_environment.tolist()
        )

    assert added_windows == [[0, 0], [0, 1], [1, 1], [1, 1]]
    assert buffer.num_sequences == 5
    batch = buffer.sample(100, torch.device("cpu"))

    assert torch.all(batch.actions[:, 0] == batch.actions[:, 1])
    assert torch.all(
        batch.observations[:, 1:] - batch.observations[:, :-1] == 1
    )
    assert not torch.any(batch.terminated)
    assert not torch.any(batch.truncated)


def test_sequence_replay_discards_windows_overwritten_by_ring_storage() -> None:
    buffer = SequenceReplayBuffer(
        capacity=5,
        num_environments=1,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=3,
        action_dtype=np.int64,
        seed=19,
    )
    for step in range(8):
        buffer.add(_transition(step))

    assert len(buffer) == 5
    assert buffer.num_sequences == 3
    batch = buffer.sample(100, torch.device("cpu"))

    assert set(batch.start_steps.tolist()) <= {3, 4, 5}
    assert torch.equal(
        batch.observations[:, 0, 0].long(),
        batch.start_steps,
    )


def test_vector_add_reports_availability_after_expiry_and_replacement() -> None:
    buffer = SequenceReplayBuffer(
        capacity=4,
        num_environments=2,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
    )
    buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([[0], [10]], dtype=np.float32),
            actions=np.asarray([0, 0]),
            rewards=np.zeros(2, dtype=np.float32),
            next_observations=np.asarray([[1], [11]], dtype=np.float32),
            terminated=np.asarray([False, True]),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )
    second_result = buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([[1], [20]], dtype=np.float32),
            actions=np.asarray([0, 0]),
            rewards=np.zeros(2, dtype=np.float32),
            next_observations=np.asarray([[2], [21]], dtype=np.float32),
            terminated=np.asarray([True, False]),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )
    replacement_result = buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([[30], [21]], dtype=np.float32),
            actions=np.asarray([0, 0]),
            rewards=np.zeros(2, dtype=np.float32),
            next_observations=np.asarray([[31], [22]], dtype=np.float32),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )

    assert second_result.added_per_environment.tolist() == [1, 0]
    assert second_result.available_after_environment.tolist() == [1, 1]
    assert replacement_result.added_per_environment.tolist() == [0, 1]
    assert replacement_result.available_after_environment.tolist() == [0, 1]
    assert buffer.num_sequences == 1


def test_sequence_replay_checkpoint_restores_contents_and_sampling() -> None:
    buffer = SequenceReplayBuffer(
        capacity=9,
        num_environments=2,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
        seed=23,
    )
    for step in range(6):
        buffer.add_batch(
            TransitionBatch(
                observations=np.asarray(
                    [[step], [100 + step]],
                    dtype=np.float32,
                ),
                actions=np.asarray([0, 1], dtype=np.int64),
                rewards=np.asarray([step, -step], dtype=np.float32),
                next_observations=np.asarray(
                    [[step + 1], [101 + step]],
                    dtype=np.float32,
                ),
                terminated=np.asarray([step == 2, False]),
                truncated=np.asarray([False, step == 4]),
            )
        )
    restored = SequenceReplayBuffer(
        capacity=9,
        num_environments=2,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
        seed=999,
    )
    restored.load_state_dict(buffer.state_dict())

    expected = buffer.sample(32, torch.device("cpu"))
    actual = restored.sample(32, torch.device("cpu"))

    assert len(restored) == len(buffer)
    assert restored.num_sequences == buffer.num_sequences
    assert restored.lane_capacities.tolist() == [5, 4]
    for field in (
        "observations",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "is_first",
        "episode_ids",
        "start_steps",
        "sample_ages",
    ):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_sequence_replay_rejects_capacity_that_cannot_fit_burn_in() -> None:
    with pytest.raises(ValueError, match="sequence_length \\+ burn_in"):
        SequenceReplayBuffer(
            capacity=7,
            num_environments=2,
            observation_shape=(1,),
            action_shape=(),
            sequence_length=3,
            burn_in=1,
        )


def test_sequence_replay_rejects_checkpoint_atomically() -> None:
    buffer = SequenceReplayBuffer(
        capacity=6,
        num_environments=1,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
    )
    for step in range(4):
        buffer.add(_transition(step))
    original_state = buffer.state_dict()
    malformed_state = buffer.state_dict()
    malformed_state["rewards"][0][0] = 999.0
    malformed_state["next_episode_id"] = -1

    with pytest.raises(ValueError, match="next episode ID"):
        buffer.load_state_dict(malformed_state)

    current_state = buffer.state_dict()
    assert np.array_equal(
        current_state["rewards"][0],
        original_state["rewards"][0],
    )
    assert current_state["next_episode_id"] == original_state["next_episode_id"]
    expected = SequenceReplayBuffer(
        capacity=6,
        num_environments=1,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
        action_dtype=np.int64,
    )
    expected.load_state_dict(original_state)
    assert torch.equal(
        buffer.sample(16, torch.device("cpu")).observations,
        expected.sample(16, torch.device("cpu")).observations,
    )


def test_sequence_replay_rejects_batches_with_wrong_lane_count() -> None:
    buffer = SequenceReplayBuffer(
        capacity=8,
        num_environments=2,
        observation_shape=(1,),
        action_shape=(),
        sequence_length=2,
    )
    transitions = TransitionBatch(
        observations=np.asarray([[0.0]], dtype=np.float32),
        actions=np.asarray([0.0], dtype=np.float32),
        rewards=np.asarray([0.0], dtype=np.float32),
        next_observations=np.asarray([[1.0]], dtype=np.float32),
        terminated=np.asarray([False]),
        truncated=np.asarray([False]),
    )

    with pytest.raises(ValueError, match="does not match"):
        buffer.add_batch(transitions)
