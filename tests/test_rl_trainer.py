"""Tests for the common reinforcement-learning trainer lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from torch import nn

from dl_core import load_builtin_components
from dl_core.core import Callback, RLTrainer, Transition


class _RecordingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.episodes: list[int] = []
        self.episode_logs: list[dict[str, Any]] = []
        self.updates: list[int] = []
        self.evaluations: list[int] = []

    def on_episode_end(
        self,
        episode: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_episode_end(episode, logs)
        self.episodes.append(episode)
        self.episode_logs.append(logs or {})

    def on_update_end(
        self,
        update: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_update_end(update, logs)
        self.updates.append(update)

    def on_evaluation_end(
        self,
        step: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_evaluation_end(step, logs)
        self.evaluations.append(step)


class _TestRLTrainer(RLTrainer):
    def setup_algorithm(self) -> None:
        self.transition_count = 0
        self.transition_steps: list[int] = []
        self.action_model_modes: list[bool] = []
        self.models = {"policy": nn.Linear(1, 1)}

    def select_action(self, observation: Any, *, deterministic: bool) -> int:
        self.action_model_modes.append(self.models["policy"].training)
        return 1

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float]:
        self.transition_count += 1
        self.transition_steps.append(self.global_step)
        return {"update/count": float(self.transition_count)}

    def algorithm_state_dict(self) -> dict[str, Any]:
        return {"transition_count": self.transition_count}

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        self.transition_count = int(state.get("transition_count", 0))


class _EmptyUpdateRLTrainer(_TestRLTrainer):
    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float]:
        self.transition_count += 1
        return {}


def _config(tmp_path: Path, **trainer_overrides: Any) -> dict[str, Any]:
    trainer_config = {
        "total_timesteps": 4,
        "max_episode_steps": 2,
        "evaluation_frequency": 1,
        "evaluation_episodes": 1,
        "checkpoint_frequency": 1,
        **trainer_overrides,
    }
    return {
        "seed": 5,
        "deterministic": True,
        "environment": {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        },
        "trainer": {"test_rl": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "rl-tests", "run_name": "lifecycle"},
    }


def test_rl_trainer_runs_episode_lifecycle_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(_config(tmp_path))
    callback = _RecordingCallback()
    trainer.setup()
    trainer.callbacks.append(callback)

    trainer.perform_training()

    assert trainer.global_step == 4
    assert trainer.collector_step == 4
    assert trainer.current_episode == 2
    assert trainer.update_step == 4
    assert callback.episodes == [0, 1, 1, 2]
    assert callback.updates == [1, 2, 3, 4]
    assert callback.evaluations == [2, 4]
    assert len(trainer.episode_metrics) == 2
    assert len(trainer.evaluation_metrics) == 2
    assert trainer.artifact_manager.get_final_checkpoint_path("latest.pth").exists()
    trainer.close()


def test_rl_trainer_checkpoint_restores_common_and_algorithm_state(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(_config(tmp_path, evaluation_episodes=0))
    trainer.setup()
    trainer.run_episode(training=True, episode=0)
    trainer.accelerator.get_accelerator_state = lambda: {
        "test_accelerator_state": 13
    }
    checkpoint_path = trainer.save_checkpoint("resume.pth")
    assert checkpoint_path is not None

    restored = _TestRLTrainer(_config(tmp_path, evaluation_episodes=0))
    restored.setup()
    loaded_accelerator_states: list[dict[str, Any]] = []
    restored.accelerator.load_accelerator_state = loaded_accelerator_states.append
    restored.load_checkpoint(str(checkpoint_path))

    assert restored.current_episode == 1
    assert restored.current_epoch == 1
    assert restored.global_step == 2
    assert restored.update_step == 2
    assert restored.transition_count == 2
    assert loaded_accelerator_states[0]["test_accelerator_state"] == 13
    trainer.close()
    restored.close()


def test_rl_trainer_counts_updates_without_metrics(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = _EmptyUpdateRLTrainer(_config(tmp_path, evaluation_episodes=0))
    callback = _RecordingCallback()
    trainer.setup()
    trainer.callbacks.append(callback)

    trainer.run_episode(training=True, episode=0)

    assert trainer.update_step == 2
    assert callback.updates == [1, 2]
    trainer.close()


def test_rl_trainer_collects_vector_environments_and_tracks_each_lane(
    tmp_path: Path,
) -> None:
    """Vector collection should count transitions and episodes per lane."""
    load_builtin_components()
    config = _config(
        tmp_path,
        total_timesteps=8,
        max_episode_steps=2,
        evaluation_episodes=0,
        checkpoint_frequency=0,
    )
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "vectorization_mode": "sync",
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    trainer = _TestRLTrainer(config)
    callback = _RecordingCallback()
    trainer.setup()
    trainer.callbacks.append(callback)

    trainer.perform_training()

    assert trainer.global_step == 8
    assert trainer.collector_step == 4
    assert trainer.current_episode == 4
    assert trainer.update_step == 8
    assert trainer.transition_steps == list(range(1, 9))
    assert callback.episodes == [0, 1, 2, 3]
    assert len(trainer.episode_metrics) == 4
    assert {metric["environment_index"] for metric in trainer.episode_metrics} == {
        0,
        1,
    }
    trainer.close()


def test_rl_trainer_saves_checkpoints_at_step_intervals(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(
        _config(
            tmp_path,
            total_timesteps=4,
            evaluation_episodes=0,
            checkpoint_frequency=0,
            checkpoint_frequency_steps=2,
        )
    )
    trainer.setup()

    trainer.perform_training()

    assert trainer.artifact_manager.get_final_checkpoint_path(
        "step_000000000002.pth"
    ).exists()
    assert trainer.artifact_manager.get_final_checkpoint_path(
        "step_000000000004.pth"
    ).exists()
    trainer.close()


def test_rl_trainer_resumes_step_checkpoint_schedule_after_saved_interval(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(
        _config(
            tmp_path,
            total_timesteps=2,
            evaluation_episodes=0,
            checkpoint_frequency=0,
            checkpoint_frequency_steps=2,
        )
    )
    trainer.setup()
    trainer.perform_training()
    checkpoint_path = trainer.artifact_manager.get_final_checkpoint_path(
        "step_000000000002.pth"
    )

    restored = _TestRLTrainer(
        _config(
            tmp_path,
            total_timesteps=4,
            evaluation_episodes=0,
            checkpoint_frequency=0,
            checkpoint_frequency_steps=2,
        )
    )
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    with patch.object(
        restored,
        "save_checkpoint",
        wraps=restored.save_checkpoint,
    ) as save_checkpoint:
        restored.perform_training()

    assert [call.args[0] for call in save_checkpoint.call_args_list] == [
        "step_000000000004.pth",
        "latest.pth",
    ]
    trainer.close()
    restored.close()


def test_rl_trainer_saves_vector_checkpoints_after_crossed_intervals(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(
        tmp_path,
        total_timesteps=6,
        max_episode_steps=100,
        evaluation_episodes=0,
        checkpoint_frequency=0,
        checkpoint_frequency_steps=3,
    )
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "vectorization_mode": "sync",
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    trainer = _TestRLTrainer(config)
    trainer.setup()

    trainer.perform_training()

    assert trainer.global_step == 6
    assert trainer.artifact_manager.get_final_checkpoint_path(
        "step_000000000004.pth"
    ).exists()
    assert trainer.artifact_manager.get_final_checkpoint_path(
        "step_000000000006.pth"
    ).exists()
    trainer.close()


def test_rl_trainer_optionally_displays_step_progress(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(
        _config(
            tmp_path,
            evaluation_episodes=0,
            checkpoint_frequency=0,
            show_progress=True,
        )
    )
    trainer.setup()

    with patch("dl_core.core.rl_trainer.tqdm") as progress_factory:
        progress = progress_factory.return_value
        progress.n = 0
        progress.total = 4
        progress.update.side_effect = lambda amount: setattr(
            progress,
            "n",
            progress.n + amount,
        )

        trainer.perform_training()

    assert progress.n == 4
    progress.close.assert_called_once_with()
    trainer.close()


def test_rl_trainer_closes_progress_when_training_is_interrupted(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(
        _config(
            tmp_path,
            evaluation_episodes=0,
            checkpoint_frequency=0,
            show_progress=True,
        )
    )
    trainer.setup()

    with (
        patch("dl_core.core.rl_trainer.tqdm") as progress_factory,
        patch.object(trainer, "run_episode", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        trainer.perform_training()

    progress_factory.return_value.close.assert_called_once_with()
    assert trainer._progress_bar is None
    trainer.close()


def test_rl_trainer_evaluates_in_eval_mode_and_restores_model_mode(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(_config(tmp_path))
    callback = _RecordingCallback()
    trainer.setup()
    trainer.callbacks.append(callback)

    trainer.evaluate()

    assert trainer.action_model_modes
    assert not any(trainer.action_model_modes)
    assert trainer.models["policy"].training
    assert callback.episode_logs[-1]["phase"] == "evaluation"
    assert callback.episode_logs[-1]["episode/truncated"] is True
    trainer.close()


def test_rl_trainer_rejects_checkpoint_for_another_trainer(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = _TestRLTrainer(_config(tmp_path, evaluation_episodes=0))
    trainer.setup()
    checkpoint_path = trainer.save_checkpoint("wrong-trainer.pth")
    assert checkpoint_path is not None

    restored = _EmptyUpdateRLTrainer(_config(tmp_path, evaluation_episodes=0))
    restored.setup()

    with pytest.raises(ValueError, match="does not match"):
        restored.load_checkpoint(str(checkpoint_path))
    trainer.close()
    restored.close()


def test_rl_trainer_rejects_distributed_environment_collection(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(tmp_path)
    config["accelerator"] = {"type": "multi_gpu"}
    trainer = _TestRLTrainer(config)

    try:
        trainer.setup_accelerator()
    except NotImplementedError as error:
        assert "distributed environment collection" in str(error)
    else:
        raise AssertionError("multi_gpu RL setup should be rejected")
