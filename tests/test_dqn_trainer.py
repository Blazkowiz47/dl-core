"""Tests for replay storage and the deep Q-network trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dl_core import load_builtin_components
from dl_core.core import ReplayBuffer, TRAINER_REGISTRY, Transition
from dl_core.models import DQNMLP
from dl_core.trainers import DQNTrainer


def _config(tmp_path: Path, **overrides: object) -> dict:
    trainer_config = {
        "total_timesteps": 10,
        "max_episode_steps": 5,
        "evaluation_episodes": 0,
        "checkpoint_frequency": 0,
        "gamma": 0.9,
        "buffer_size": 8,
        "batch_size": 1,
        "learning_starts": 0,
        "train_frequency": 1,
        "gradient_steps": 1,
        "target_update_frequency": 100,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "checkpoint_replay_buffer": True,
        **overrides,
    }
    return {
        "seed": 17,
        "environment": {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        },
        "models": {"q_network": {"name": "dqn_mlp", "hidden_sizes": []}},
        "optimizers": {"name": "sgd", "lr": 0.01},
        "trainer": {"dqn": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "dqn-tests", "run_name": "dqn"},
    }


def test_replay_buffer_round_trip_preserves_ring_and_sampling_state() -> None:
    buffer = ReplayBuffer(3, (2,), (), action_dtype=np.int64, seed=3)
    for index in range(4):
        buffer.add(
            Transition(
                observation=np.asarray([index, index + 1], dtype=np.float32),
                action=index % 2,
                reward=float(index),
                next_observation=np.asarray([index + 1, index + 2], dtype=np.float32),
                terminated=index == 3,
                truncated=False,
            )
        )
    state = buffer.state_dict()
    restored = ReplayBuffer(3, (2,), (), action_dtype=np.int64, seed=99)
    restored.load_state_dict(state)

    assert len(restored) == 3
    assert restored.position == 1
    first_sample = buffer.sample(3, torch.device("cpu"))
    restored_sample = restored.sample(3, torch.device("cpu"))
    assert torch.equal(first_sample.observations, restored_sample.observations)
    assert torch.equal(first_sample.terminated, restored_sample.terminated)


def test_dqn_is_registered_and_builtin_model_has_expected_shape(tmp_path: Path) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("dqn") is DQNTrainer
    model = DQNMLP({"input_dim": 3, "action_dim": 2, "hidden_sizes": [4]})
    assert model(torch.zeros(5, 3)).shape == (5, 2)

    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    assert set(trainer.models) == {"online", "target"}
    assert trainer.models["target"].training is False
    assert all(
        not parameter.requires_grad
        for parameter in trainer.models["target"].parameters()
    )
    trainer.close()


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected_target"),
    [(True, False, 1.0), (False, True, 2.8)],
)
def test_dqn_uses_terminal_aware_targets(
    tmp_path: Path,
    terminated: bool,
    truncated: bool,
    expected_target: float,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    with torch.no_grad():
        trainer.models["online"].network[-1].weight.zero_()
        trainer.models["online"].network[-1].bias.zero_()
        trainer.models["target"].network[-1].weight.zero_()
        trainer.models["target"].network[-1].bias.copy_(
            torch.tensor([2.0, 0.0, 0.0, 0.0])
        )
    trainer.global_step = 1

    metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=terminated,
            truncated=truncated,
        )
    )

    assert metrics is not None
    assert metrics["dqn/target_q_mean"] == pytest.approx(expected_target)
    trainer.close()


def test_dqn_checkpoint_restores_target_network_and_replay(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    trainer.global_step = 1
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
    checkpoint_path = trainer.save_checkpoint("dqn.pth")
    assert checkpoint_path is not None

    restored = DQNTrainer(_config(tmp_path))
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert len(restored.replay_buffer) == 1
    assert restored.epsilon == trainer.epsilon
    for restored_parameter, parameter in zip(
        restored.models["target"].parameters(),
        trainer.models["target"].parameters(),
        strict=True,
    ):
        assert torch.equal(restored_parameter, parameter)
    trainer.close()
    restored.close()
