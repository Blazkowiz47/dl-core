"""Tests for the common reinforcement-learning trainer lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core import load_builtin_components
from dl_core.core import Callback, RLTrainer, Transition


class _RecordingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.episodes: list[int] = []
        self.updates: list[int] = []
        self.evaluations: list[int] = []

    def on_episode_end(
        self,
        episode: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_episode_end(episode, logs)
        self.episodes.append(episode)

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

    def select_action(self, observation: Any, *, deterministic: bool) -> int:
        return 1

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float]:
        self.transition_count += 1
        return {"update/count": float(self.transition_count)}

    def algorithm_state_dict(self) -> dict[str, Any]:
        return {"transition_count": self.transition_count}

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        self.transition_count = int(state.get("transition_count", 0))


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
    checkpoint_path = trainer.save_checkpoint("resume.pth")
    assert checkpoint_path is not None

    restored = _TestRLTrainer(
        _config(tmp_path, evaluation_episodes=0, continue_model=str(checkpoint_path))
    )
    restored.setup()

    assert restored.current_episode == 1
    assert restored.current_epoch == 1
    assert restored.global_step == 2
    assert restored.update_step == 2
    assert restored.transition_count == 2
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
