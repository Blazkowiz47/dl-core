"""Tests for complete reinforcement-learning episode tracking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dl_core.core import (
    EpisodeContext,
    EpisodeResult,
    Transition,
)
from dl_core.episode_managers.standard import StandardEpisodeManager
from dl_core.utils import ArtifactManager


def test_standard_episode_manager_tracks_and_persists_complete_episode(
    tmp_path: Path,
) -> None:
    """Captured trajectories retain T+1 observations and T transition fields."""
    artifact_manager = ArtifactManager(
        run_name="episode-test",
        output_dir=str(tmp_path),
    )
    manager = StandardEpisodeManager(
        {
            "capture_phases": ["train"],
            "info_keys": ["is_success"],
            "capture_action_info": True,
        },
        artifact_manager=artifact_manager,
    )
    manager.begin_episode(
        EpisodeContext(
            episode_id="train-000000",
            episode=0,
            environment_index=0,
            phase="train",
            seed=42,
            initial_observation=np.asarray([0.0, 1.0], dtype=np.float32),
        )
    )
    manager.record_transition(
        0,
        Transition(
            observation=np.asarray([0.0, 1.0], dtype=np.float32),
            action=1,
            reward=0.25,
            next_observation=np.asarray([1.0, 1.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            info={"ignored": "value"},
            action_info={"value": 0.5},
        ),
    )
    manager.record_transition(
        0,
        Transition(
            observation=np.asarray([1.0, 1.0], dtype=np.float32),
            action=2,
            reward=1.0,
            next_observation=np.asarray([2.0, 1.0], dtype=np.float32),
            terminated=True,
            truncated=False,
            info={"is_success": True},
            action_info={"value": 0.75},
        ),
    )
    result = EpisodeResult(
        episode=0,
        episode_return=1.25,
        length=2,
        terminated=True,
        truncated=False,
        final_info={"is_success": True},
    )

    record = manager.end_episode(0, result)

    assert record.length == 2
    assert len(record.observations) == 3
    assert record.infos == [{}, {"is_success": True}]
    assert result.metrics["episode/return"] == 1.25
    trajectory_path = Path(result.artifact_paths["trajectory"])
    assert trajectory_path.exists()
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        assert trajectory["observations"].shape == (3, 2)
        assert trajectory["actions"].tolist() == [1, 2]
        metadata = json.loads(str(trajectory["metadata_json"]))
    assert metadata["infos"] == [{}, {"is_success": True}]


def test_standard_episode_manager_summary_mode_does_not_retain_trajectory() -> None:
    """Non-captured phases retain only online scalar statistics."""
    manager = StandardEpisodeManager({"capture_phases": ["evaluation"]})
    manager.begin_episode(
        EpisodeContext(
            episode_id="train-000001",
            episode=1,
            environment_index=0,
            phase="train",
            seed=43,
            initial_observation=0,
        )
    )
    manager.record_transition(
        0,
        Transition(
            observation=0,
            action=1,
            reward=-2.0,
            next_observation=1,
            terminated=False,
            truncated=True,
        ),
    )
    result = EpisodeResult(
        episode=1,
        episode_return=-2.0,
        length=1,
        terminated=False,
        truncated=True,
    )

    record = manager.end_episode(0, result)

    assert record.observations == []
    assert record.actions == []
    assert record.metrics["episode/reward_min"] == -2.0
    assert record.artifact_paths == {}


def test_episode_manager_rejects_overlapping_active_episode() -> None:
    """One environment lane cannot own two active episodes."""
    manager = StandardEpisodeManager()
    context = EpisodeContext(
        episode_id="evaluation-000000",
        episode=0,
        environment_index=2,
        phase="evaluation",
        seed=None,
        initial_observation={"position": np.asarray([0, 0])},
    )
    manager.begin_episode(context)

    try:
        manager.begin_episode(context)
    except RuntimeError as error:
        assert "already has an active episode" in str(error)
    else:
        raise AssertionError("Expected overlapping episode tracking to fail")


def test_episode_manager_snapshots_mutable_trajectory_values() -> None:
    """Captured values should not change when environments reuse array buffers."""
    manager = StandardEpisodeManager(
        {
            "capture_phases": ["train"],
            "info_keys": ["position"],
            "capture_action_info": True,
        }
    )
    initial_observation = np.asarray([0.0], dtype=np.float32)
    next_observation = np.asarray([1.0], dtype=np.float32)
    action = np.asarray([0.5], dtype=np.float32)
    position = np.asarray([1, 0], dtype=np.int64)
    action_value = np.asarray([0.25], dtype=np.float32)
    manager.begin_episode(
        EpisodeContext(
            episode_id="train-000002",
            episode=2,
            environment_index=0,
            phase="train",
            seed=44,
            initial_observation=initial_observation,
        )
    )
    manager.record_transition(
        0,
        Transition(
            observation=initial_observation,
            action=action,
            reward=1.0,
            next_observation=next_observation,
            terminated=True,
            truncated=False,
            info={"position": position},
            action_info={"value": action_value},
        ),
    )

    initial_observation[:] = 9.0
    next_observation[:] = 9.0
    action[:] = 9.0
    position[:] = 9
    action_value[:] = 9.0
    result = EpisodeResult(
        episode=2,
        episode_return=1.0,
        length=1,
        terminated=True,
        truncated=False,
    )
    record = manager.end_episode(0, result)

    assert record.observations[0].tolist() == [0.0]
    assert record.observations[1].tolist() == [1.0]
    assert record.actions[0].tolist() == [0.5]
    assert record.infos[0]["position"].tolist() == [1, 0]
    assert record.action_info[0]["value"].tolist() == [0.25]
    assert result.episode_id == "train-000002"
    assert result.seed == 44


def test_episode_manager_reserves_capture_limit_across_active_lanes() -> None:
    """Concurrent vector lanes should not exceed the configured capture limit."""
    manager = StandardEpisodeManager(
        {
            "capture_phases": ["train"],
            "max_captured_episodes": 1,
        }
    )
    for environment_index in range(2):
        manager.begin_episode(
            EpisodeContext(
                episode_id=f"train-{environment_index:06d}",
                episode=environment_index,
                environment_index=environment_index,
                phase="train",
                seed=environment_index,
                initial_observation=environment_index,
            )
        )
        manager.record_transition(
            environment_index,
            Transition(
                observation=environment_index,
                action=1,
                reward=1.0,
                next_observation=environment_index + 1,
                terminated=True,
                truncated=False,
            ),
        )

    first_record = manager.end_episode(
        0,
        EpisodeResult(0, 1.0, 1, True, False),
    )
    second_record = manager.end_episode(
        1,
        EpisodeResult(1, 1.0, 1, True, False),
    )

    assert len(first_record.observations) == 2
    assert second_record.observations == []


def test_episode_manager_rejects_unsafe_artifact_names() -> None:
    """Episode identity fields must not escape the managed artifact directory."""
    manager = StandardEpisodeManager()
    context = EpisodeContext(
        episode_id="../episode",
        episode=0,
        environment_index=0,
        phase="evaluation",
        seed=None,
        initial_observation=0,
    )

    try:
        manager.begin_episode(context)
    except ValueError as error:
        assert "episode_id" in str(error)
    else:
        raise AssertionError("Expected an unsafe episode ID to fail")
