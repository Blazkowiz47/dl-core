"""Tests for soft actor-critic models and trainer behavior."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from torch.distributions import Normal
from torch.nn import functional

from dl_core import load_builtin_components
from dl_core.core import (
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    Transition,
    TransitionBatch,
    register_model,
)
from dl_core.trainers import SACTrainer


@register_model("test_sac_actor")
class _TestSACActor(torch.nn.Module):
    """Small project-style Gaussian actor used by trainer tests."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(8, action_dim)
        self.log_std_head = torch.nn.Linear(8, action_dim)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return raw Gaussian policy parameters."""
        features = self.encoder(
            observations.reshape(observations.shape[0], -1)
        )
        return {
            "mean": self.mean_head(features),
            "log_std": self.log_std_head(features),
        }


@register_model("test_sac_critics")
class _TestSACCritics(torch.nn.Module):
    """Small project-style twin critic used by trainer tests."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        self.critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim + action_dim, 8),
                    torch.nn.ReLU(),
                    torch.nn.Linear(8, 1),
                )
                for _ in range(2)
            ]
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the two independent action-value estimates."""
        inputs = torch.cat(
            (
                observations.reshape(observations.shape[0], -1),
                actions.reshape(actions.shape[0], -1),
            ),
            dim=1,
        )
        return {
            "q1": self.critics[0](inputs).squeeze(1),
            "q2": self.critics[1](inputs).squeeze(1),
        }


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
            "actor": {"name": "test_sac_actor"},
            "critics": {"name": "test_sac_critics"},
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


def test_sac_uses_registered_project_models(
    tmp_path: Path,
) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("sac") is SACTrainer
    assert not MODEL_REGISTRY.is_registered("sac_gaussian_actor")
    assert not MODEL_REGISTRY.is_registered("sac_twin_q_network")
    actor = _TestSACActor({"input_dim": 3, "action_dim": 2})
    critics = _TestSACCritics({"input_dim": 3, "action_dim": 2})
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


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        (torch.zeros(2, 1), TypeError, "return a mapping"),
        ({"mean": torch.zeros(2, 1)}, TypeError, "mean.*log_std"),
        (
            {"mean": torch.zeros(1, 1), "log_std": torch.zeros(1, 1)},
            ValueError,
            r"\[batch, action_dimensions\]",
        ),
        (
            {
                "mean": torch.zeros(2, 1, dtype=torch.int64),
                "log_std": torch.zeros(2, 1),
            },
            TypeError,
            "floating-point",
        ),
        (
            {
                "mean": torch.full((2, 1), float("nan")),
                "log_std": torch.zeros(2, 1),
            },
            FloatingPointError,
            "must be finite",
        ),
    ],
)
def test_sac_validates_project_actor_output_contract(
    tmp_path: Path,
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()

    class _ContractActor(torch.nn.Module):
        def forward(self, observations: torch.Tensor) -> Any:
            del observations
            return output

    trainer.models["actor"] = _ContractActor()
    with pytest.raises(error, match=message):
        trainer._sample_action_and_log_probability(
            torch.zeros(2, 3),
            deterministic=True,
        )
    trainer.close()


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        (torch.zeros(2), TypeError, "return a mapping"),
        ({"q1": torch.zeros(2)}, TypeError, "q1.*q2"),
        (
            {"q1": torch.zeros(1), "q2": torch.zeros(1)},
            ValueError,
            r"\[batch\]",
        ),
        (
            {
                "q1": torch.zeros(2, dtype=torch.int64),
                "q2": torch.zeros(2),
            },
            TypeError,
            "floating-point",
        ),
        (
            {
                "q1": torch.full((2,), float("nan")),
                "q2": torch.zeros(2),
            },
            FloatingPointError,
            "must be finite",
        ),
    ],
)
def test_sac_validates_project_critic_output_contract(
    tmp_path: Path,
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()

    class _ContractCritics(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
        ) -> Any:
            del observations, actions
            return output

    with pytest.raises(error, match=message):
        trainer._q_values(
            _ContractCritics(),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
        )
    trainer.close()


def test_sac_accepts_general_mapping_model_outputs(tmp_path: Path) -> None:
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()

    class _MappingActor(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
        ) -> MappingProxyType:
            return MappingProxyType(
                {
                    "mean": torch.zeros(observations.shape[0], 1),
                    "log_std": torch.zeros(observations.shape[0], 1),
                }
            )

    class _MappingCritics(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
        ) -> MappingProxyType:
            del actions
            return MappingProxyType(
                {
                    "q1": torch.zeros(observations.shape[0]),
                    "q2": torch.ones(observations.shape[0]),
                }
            )

    trainer.models["actor"] = _MappingActor()
    actions, log_probabilities = trainer._sample_action_and_log_probability(
        torch.zeros(2, 3),
        deterministic=True,
    )
    q1, q2 = trainer._q_values(
        _MappingCritics(),
        torch.zeros(2, 3),
        actions,
    )

    assert actions.shape == (2, 1)
    assert log_probabilities.shape == (2,)
    assert q1.tolist() == [0.0, 0.0]
    assert q2.tolist() == [1.0, 1.0]
    trainer.close()


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (None, r"requires models\.actor\.name and models\.critics\.name"),
        ({}, r"requires models\.actor\.name"),
        (
            {"actor": {}, "critics": {"name": "test_sac_critics"}},
            r"requires models\.actor\.name",
        ),
        (
            {"actor": {"name": "test_sac_actor"}, "critics": {}},
            r"requires models\.critics\.name",
        ),
    ],
)
def test_sac_requires_explicit_project_models(
    tmp_path: Path,
    models: object,
    message: str,
) -> None:
    config = _config(tmp_path)
    if models is None:
        config.pop("models")
    else:
        config["models"] = models
    trainer = SACTrainer(config)

    with pytest.raises(ValueError, match=message):
        trainer.setup()
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


def test_sac_uses_stable_float32_squashed_gaussian_density(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()

    def saturated_half_precision_actor(
        _actor: torch.nn.Module,
        observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shape = (observations.shape[0], 1)
        return {
            "mean": torch.full(shape, 20.0, device=observations.device).half(),
            "log_std": torch.full(shape, -20.0, device=observations.device).half(),
        }

    trainer.models["actor"].forward = MethodType(
        saturated_half_precision_actor,
        trainer.models["actor"],
    )
    observations = trainer._observations_to_tensor(
        np.zeros((1, 3), dtype=np.float32)
    )

    actions, log_probabilities = trainer._sample_action_and_log_probability(
        observations,
        deterministic=True,
    )

    raw_action = torch.tensor([20.0])
    log_tanh_jacobian = 2.0 * (
        np.log(2.0) - raw_action - functional.softplus(-2.0 * raw_action)
    )
    expected = (
        Normal(raw_action, torch.exp(torch.tensor([-20.0]))).log_prob(raw_action)
        - log_tanh_jacobian
        - torch.log(torch.tensor([2.0]))
    ).sum()
    assert actions.dtype == torch.float32
    assert log_probabilities.dtype == torch.float32
    assert torch.isfinite(log_probabilities).all()
    assert log_probabilities.item() == pytest.approx(expected.item())
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

    trainer.environment = trainer.evaluation_environment = SimpleNamespace(
        observation_space=observation_space,
        action_space=Box(1e100, 2e100, shape=(1,), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="representable in float32"):
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


def test_sac_uses_n_step_rewards_and_bootstrap_discount(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path, n_step=2))
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
    first_metrics = trainer.process_transition(
        Transition(
            observation=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=1.0,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
    )
    trainer.global_step = 2
    metrics = trainer.process_transition(
        Transition(
            observation=np.zeros(3, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=2.0,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
    )

    assert first_metrics is None
    assert metrics is not None
    assert metrics["sac/target_q_mean"] == pytest.approx(4.42)
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


def test_sac_temperature_update_uses_detached_policy_density(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path, automatic_entropy_tuning=True)
    config["optimizers"] = {
        "actor": {"name": "sgd", "lr": 0.0},
        "critics": {"name": "sgd", "lr": 0.0},
        "temperature": {"name": "sgd", "lr": 0.1},
    }
    trainer = SACTrainer(config)
    trainer.setup()
    _use_zero_policy_statistics(trainer)
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

    assert trainer._alpha().item() == pytest.approx(0.2 * np.exp(-0.1))
    assert all(parameter.grad is None for parameter in trainer.models["actor"].parameters())
    assert all(
        parameter.requires_grad for parameter in trainer.models["critics"].parameters()
    )
    trainer.close()


def test_sac_batches_vector_actions_replay_and_update_schedules(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(
        tmp_path,
        total_timesteps=8,
        max_episode_steps=4,
        train_frequency=1,
    )
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "Pendulum-v1",
        "num_envs": 2,
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "Pendulum-v1",
    }
    trainer = SACTrainer(config)
    trainer.setup()
    action_batch_sizes: list[int] = []
    sample_action = trainer._sample_action_and_log_probability

    def recording_sample_action(
        observations: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_grad_enabled():
            action_batch_sizes.append(int(observations.shape[0]))
        return sample_action(observations, deterministic=deterministic)

    trainer._sample_action_and_log_probability = recording_sample_action
    trainer.perform_training()

    assert trainer.global_step == 8
    assert trainer.collector_step == 4
    assert trainer.update_step == 8
    assert len(trainer.replay_buffer) == 8
    assert action_batch_sizes.count(2) == 4
    trainer.close()


def test_sac_preserves_crossed_vector_update_steps_with_n_step_replay(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path, n_step=2, train_frequency=1))
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=29)
    observations = np.repeat(observation[None, :], 2, axis=0)
    actions = np.zeros((2, *trainer.environment.action_space.shape), dtype=np.float32)
    first_transitions = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=np.ones(2, dtype=np.float32),
        next_observations=observations,
        terminated=np.zeros(2, dtype=np.bool_),
        truncated=np.zeros(2, dtype=np.bool_),
    )
    trainer.global_step = 2
    first_logs = trainer.process_transition_batch(first_transitions)
    trainer.global_step = 4
    logs = trainer.process_transition_batch(first_transitions)

    assert first_logs == []
    assert len(logs) == 2
    assert len(trainer.replay_buffer) == 2
    trainer.close()


def test_sac_can_gate_eligible_updates_by_global_step(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path, train_frequency=1))
    trainer.setup()
    checked_steps: list[int] = []

    def update_on_even_steps(global_step: int, transitions: object) -> bool:
        checked_steps.append(global_step)
        return global_step % 2 == 0

    trainer.should_update = update_on_even_steps
    trainer.global_step = 4
    observation, _ = trainer.environment.reset(seed=29)
    action = trainer.environment.action_space.sample()
    logs = trainer.process_transition_batch(
        TransitionBatch(
            observations=np.repeat(observation[None, :], 4, axis=0),
            actions=np.repeat(action[None, :], 4, axis=0),
            rewards=np.ones(4, dtype=np.float32),
            next_observations=np.repeat(observation[None, :], 4, axis=0),
            terminated=np.zeros(4, dtype=np.bool_),
            truncated=np.zeros(4, dtype=np.bool_),
        )
    )

    assert checked_steps == [1, 2, 3, 4]
    assert len(logs) == 2
    trainer.close()


def test_sac_applies_vector_warmup_per_transition(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path, learning_starts=3))
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=29)
    observations = np.stack((observation, observation))
    actor_batch_sizes: list[int] = []

    def fixed_policy_actions(
        observation_batch: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del deterministic
        actor_batch_sizes.append(int(observation_batch.shape[0]))
        actions = torch.full(
            (observation_batch.shape[0], 1),
            0.5,
            device=observation_batch.device,
        )
        return actions, torch.zeros(
            observation_batch.shape[0],
            device=observation_batch.device,
        )

    trainer._sample_action_and_log_probability = fixed_policy_actions
    trainer.global_step = 2
    output = trainer.select_actions(observations, deterministic=False)
    expected_random = np.random.default_rng(29).uniform(
        trainer.environment.action_space.low,
        trainer.environment.action_space.high,
        size=(1, *trainer.environment.action_space.shape),
    ).astype(trainer.environment.action_space.dtype)

    assert actor_batch_sizes == [1]
    assert np.array_equal(output.actions[0], expected_random[0])
    assert np.array_equal(output.actions[1], np.asarray([0.5], dtype=np.float32))
    assert all(
        trainer.environment.action_space.contains(action)
        for action in output.actions
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
    expected_action = trainer.select_action(observation, deterministic=False)
    expected_random_value = trainer.random_generator.random()
    expected_replay_reward = trainer.replay_buffer.sample(
        1,
        torch.device("cpu"),
    ).rewards.item()

    restored = SACTrainer(config)
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert len(restored.replay_buffer) == 1
    assert restored._alpha().item() == pytest.approx(np.exp(-0.75))
    assert np.array_equal(
        restored.select_action(observation, deterministic=False),
        expected_action,
    )
    assert restored.random_generator.random() == expected_random_value
    assert restored.replay_buffer.sample(
        1,
        torch.device("cpu"),
    ).rewards.item() == expected_replay_reward
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


def test_sac_checkpoint_rejects_changed_training_parameters_atomically(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = SACTrainer(_config(tmp_path))
    trainer.setup()
    state = trainer.algorithm_state_dict()
    state["training_parameters"]["tau"] = 0.25

    with pytest.raises(ValueError, match="training parameters"):
        trainer.load_algorithm_state_dict(state)

    state = trainer.algorithm_state_dict()
    state["random_generator_state"] = {"invalid": True}
    replay_buffer = trainer.replay_buffer
    with pytest.raises(ValueError, match="generator state"):
        trainer.load_algorithm_state_dict(state)
    assert trainer.replay_buffer is replay_buffer
    trainer.close()
