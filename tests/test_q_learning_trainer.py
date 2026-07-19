"""Tests for the tabular Q-learning trainer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from dl_core import load_builtin_components
from dl_core.core import TRAINER_REGISTRY, Transition
from dl_core.trainers import QLearningTrainer


def _config(tmp_path: Path, **overrides: float | int) -> dict:
    trainer_config: dict[str, float | int] = {
        "total_timesteps": 20,
        "max_episode_steps": 10,
        "evaluation_episodes": 0,
        "checkpoint_frequency": 0,
        "learning_rate": 0.5,
        "gamma": 0.9,
        "epsilon_start": 1.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 10,
        **overrides,
    }
    return {
        "seed": 11,
        "environment": {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        },
        "trainer": {"q_learning": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "q-learning-tests", "run_name": "q-table"},
    }


def test_q_learning_trainer_is_registered_and_validates_discrete_spaces(
    tmp_path: Path,
) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("q_learning") is QLearningTrainer
    total_timesteps = next(
        field
        for field in QLearningTrainer.CONFIG_FIELDS
        if field["name"] == "total_timesteps"
    )
    assert total_timesteps["default"] == 0
    assert total_timesteps["required"] is False

    config = _config(tmp_path)
    config["environment"] = {"name": "gymnasium", "id": "CartPole-v1"}
    trainer = QLearningTrainer(config)
    trainer.setup_accelerator()
    trainer.setup_environment()

    with pytest.raises(TypeError, match="Discrete observation"):
        trainer.setup_algorithm()
    trainer.close()


def test_q_learning_uses_terminal_aware_td_targets(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = QLearningTrainer(_config(tmp_path))
    trainer.setup()
    trainer.q_table[1, 0] = 2.0

    terminal_metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=True,
            truncated=False,
        )
    )
    truncated_metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=2,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=True,
        )
    )

    assert trainer.q_table[0, 1] == pytest.approx(0.5)
    assert trainer.q_table[0, 2] == pytest.approx(1.4)
    assert terminal_metrics["q_learning/td_error"] == pytest.approx(1.0)
    assert truncated_metrics["q_learning/td_error"] == pytest.approx(2.8)
    trainer.close()


def test_q_learning_epsilon_decay_and_checkpoint_round_trip(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = QLearningTrainer(_config(tmp_path))
    trainer.setup()
    trainer.global_step = 5
    trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )
    trainer.q_table[2, 3] = 7.0
    checkpoint_path = trainer.save_checkpoint("q-learning.pth")
    assert checkpoint_path is not None

    restored = QLearningTrainer(_config(tmp_path))
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert restored.epsilon == pytest.approx(0.5)
    assert np.array_equal(restored.q_table, trainer.q_table)
    assert restored.select_action(2, deterministic=True) == 3
    assert restored.select_action(2, deterministic=False) == trainer.select_action(
        2,
        deterministic=False,
    )
    trainer.close()
    restored.close()


def test_q_learning_supports_nonzero_discrete_starts(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = QLearningTrainer(_config(tmp_path))
    trainer.setup_accelerator()
    trainer.environment = SimpleNamespace(
        observation_space=Discrete(3, start=5),
        action_space=Discrete(2, start=7),
    )
    trainer.evaluation_environment = SimpleNamespace(
        observation_space=Discrete(3, start=5),
        action_space=Discrete(2, start=7),
    )
    trainer.setup_algorithm()
    trainer.q_table[1] = [1.0, 2.0]

    assert trainer.select_action(6, deterministic=True) == 8
    assert all(
        action in (7, 8)
        for action in (
            trainer.select_action(5, deterministic=False) for _ in range(20)
        )
    )


def test_q_learning_rejects_incompatible_checkpoint_spaces(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = QLearningTrainer(_config(tmp_path))
    trainer.setup()
    state = trainer.algorithm_state_dict()
    state["observation_space"] = {"n": 16, "start": 1}

    with pytest.raises(ValueError, match="observation space"):
        trainer.load_algorithm_state_dict(state)

    state = trainer.algorithm_state_dict()
    del state["random_generator_state"]
    with pytest.raises(ValueError, match="random generator state"):
        trainer.load_algorithm_state_dict(state)
    trainer.close()
