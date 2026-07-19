"""Tests for soft actor-critic models and trainer behavior."""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dl_core import load_builtin_components
from dl_core.core import TRAINER_REGISTRY, Transition
from dl_core.models import SACGaussianActor, SACTwinQNetwork
from dl_core.trainers import SACTrainer


def _config(tmp_path: Path, **overrides: object) -> dict:
    trainer_config = {
        "total_timesteps": 8,
        "max_episode_steps": 4,
        "evaluation_episodes": 0,
        "checkpoint_frequency": 0,
        "gamma": 0.9,
        "buffer_size": 8,
        "batch_size": 1,
        "learning_starts": 0,
        "train_frequency": 1,
        "gradient_steps": 1,
        "tau": 0.5,
        "initial_alpha": 0.2,
        "automatic_entropy_tuning": False,
        "checkpoint_replay_buffer": True,
        **overrides,
    }
    return {
        "seed": 29,
        "environment": {"name": "gymnasium", "id": "Pendulum-v1"},
        "models": {
            "actor": {"name": "sac_gaussian_actor", "hidden_sizes": [8]},
            "critics": {"name": "sac_twin_q_network", "hidden_sizes": [8]},
        },
        "optimizers": {"name": "sgd", "lr": 0.0},
        "trainer": {"sac": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "sac-tests", "run_name": "sac"},
    }


def _use_zero_policy_statistics(trainer: SACTrainer) -> None:
    def zero_policy_statistics(
        self: SACTrainer,
        observations: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del deterministic
        output = self.models["actor"](observations)
        actions = output["mean"] * 0.0
        return actions, actions.sum(dim=1)

    trainer._sample_action_and_log_probability = MethodType(
        zero_policy_statistics,
        trainer,
    )


def test_sac_is_registered_and_builtin_models_have_expected_shapes(
    tmp_path: Path,
) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("sac") is SACTrainer
    actor = SACGaussianActor({"input_dim": 3, "action_dim": 2, "hidden_sizes": [4]})
    critics = SACTwinQNetwork(
        {"input_dim": 3, "action_dim": 2, "hidden_sizes": [4]}
    )
    actor_output = actor(torch.zeros(5, 3))
    critic_output = critics(torch.zeros(5, 3), torch.zeros(5, 2))

    assert actor_output["mean"].shape == (5, 2)
    assert actor_output["log_std"].shape == (5, 2)
    assert critic_output["q1"].shape == (5,)
    assert critic_output["q2"].shape == (5,)

    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()
    assert set(trainer.models) == {"actor", "critics", "target_critics"}
    assert trainer.models["target_critics"].training is False
    assert all(
        not parameter.requires_grad
        for parameter in trainer.models["target_critics"].parameters()
    )
    trainer.close()


def test_sac_actions_respect_box_bounds_and_restore_actor_mode(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=29)
    trainer.models["actor"].train()

    stochastic = trainer.select_action(observation, deterministic=False)
    deterministic = trainer.select_action(observation, deterministic=True)

    assert trainer.environment.action_space.contains(stochastic)
    assert trainer.environment.action_space.contains(deterministic)
    assert trainer.models["actor"].training is True
    trainer.close()


def test_sac_rejects_non_floating_or_unbounded_action_spaces(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup_accelerator()
    observation_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    trainer.environment = trainer.evaluation_environment = SimpleNamespace(
        observation_space=observation_space,
        action_space=Box(0, 3, shape=(1,), dtype=np.int64),
    )
    with pytest.raises(TypeError, match="floating-point Box"):
        trainer.setup_algorithm()

    trainer.environment = trainer.evaluation_environment = SimpleNamespace(
        observation_space=observation_space,
        action_space=Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finite Box bounds"):
        trainer.setup_algorithm()


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected_target"),
    [(True, False, 1.0), (False, True, 2.8)],
)
def test_sac_uses_twin_minimum_and_terminal_aware_targets(
    tmp_path: Path,
    terminated: bool,
    truncated: bool,
    expected_target: float,
) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()
    _use_zero_policy_statistics(trainer)
    with torch.no_grad():
        for parameter in trainer.models["critics"].parameters():
            parameter.zero_()
        for parameter in trainer.models["target_critics"].parameters():
            parameter.zero_()
        target_critics = trainer.models["target_critics"].critics
        target_critics[0][-1].bias.fill_(2.0)
        target_critics[1][-1].bias.fill_(5.0)
    trainer.global_step = 1

    metrics = trainer.process_transition(
        Transition(
            observation=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=1.0,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
        )
    )

    assert metrics is not None
    assert metrics["sac/target_q_mean"] == pytest.approx(expected_target)
    trainer.close()


def test_sac_soft_updates_target_critics(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path, tau=0.25))
    trainer.setup()
    _use_zero_policy_statistics(trainer)
    with torch.no_grad():
        for parameter in trainer.models["critics"].parameters():
            parameter.zero_()
        for parameter in trainer.models["target_critics"].parameters():
            parameter.fill_(1.0)
    trainer.global_step = 1

    trainer.process_transition(
        Transition(
            observation=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=0.0,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=True,
            truncated=False,
        )
    )

    assert all(
        torch.allclose(parameter, torch.full_like(parameter, 0.75))
        for parameter in trainer.models["target_critics"].parameters()
    )
    trainer.close()


def test_sac_checkpoint_restores_temperature_target_and_replay(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path, automatic_entropy_tuning=True)
    config["optimizers"] = {
        "actor": {"name": "adam", "lr": 1e-3},
        "critics": {"name": "adam", "lr": 1e-3},
        "temperature": {"name": "adam", "lr": 1e-3},
    }
    trainer = SACTrainer(config)
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=29)
    action = trainer.select_action(observation, deterministic=True)
    trainer.replay_buffer.add(
        Transition(
            observation=observation,
            action=action,
            reward=1.0,
            next_observation=observation,
            terminated=False,
            truncated=True,
        )
    )
    with torch.no_grad():
        trainer.models["temperature"].log_alpha.fill_(-0.75)
        for parameter in trainer.models["target_critics"].parameters():
            parameter.fill_(0.25)
    checkpoint_path = trainer.save_checkpoint("sac.pth")
    assert checkpoint_path is not None

    restored = SACTrainer(config)
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert len(restored.replay_buffer) == 1
    assert restored._alpha().item() == pytest.approx(np.exp(-0.75))
    assert all(
        torch.equal(restored_parameter, parameter)
        for restored_parameter, parameter in zip(
            restored.models["target_critics"].parameters(),
            trainer.models["target_critics"].parameters(),
            strict=True,
        )
    )
    trainer.close()
    restored.close()
