"""Tests for rollout estimation and proximal policy optimization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from dl_core import load_builtin_components
from dl_core.core import (
    MODEL_REGISTRY,
    RolloutBuffer,
    TRAINER_REGISTRY,
    Transition,
    register_model,
)
from dl_core.trainers import PPOTrainer


@register_model("test_ppo_policy")
class _TestPPOPolicy(torch.nn.Module):
    """Small project-style policy used only by PPO trainer tests."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        self.continuous_actions = bool(config["continuous_actions"])
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Linear(8, action_dim)
        self.value_head = torch.nn.Linear(8, 1)
        if self.continuous_actions:
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        else:
            self.register_parameter("log_std", None)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return policy parameters and state values."""
        features = self.encoder(
            observations.reshape(observations.shape[0], -1)
        )
        policy_output = self.policy_head(features)
        output = {"value": self.value_head(features).squeeze(1)}
        if self.continuous_actions:
            output["mean"] = policy_output
            output["log_std"] = self.log_std.expand_as(policy_output)
        else:
            output["logits"] = policy_output
        return output


@register_model("test_ppo_non_module")
class _TestPPONonModule:
    """Invalid project registration used to verify setup diagnostics."""

    def __init__(self, config: dict[str, object]):
        del config


def _config(tmp_path: Path, **overrides: object) -> dict:
    trainer_config = {
        "total_timesteps": 8,
        "max_episode_steps": 4,
        "evaluation_episodes": 0,
        "checkpoint_frequency": 0,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "rollout_steps": 2,
        "update_epochs": 2,
        "minibatch_size": 2,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "entropy_coefficient": 0.0,
        **overrides,
    }
    return {
        "seed": 23,
        "environment": {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        },
        "models": {
            "policy": {
                "name": "test_ppo_policy",
            }
        },
        "optimizers": {"name": "adam", "lr": 1e-3},
        "trainer": {"ppo": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "ppo-tests", "run_name": "ppo"},
    }


def test_rollout_buffer_distinguishes_termination_and_truncation() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        observation=0,
        action=0,
        reward=1.0,
        value=0.5,
        log_probability=-0.1,
        next_value=10.0,
        terminated=True,
        truncated=False,
    )
    buffer.add(
        observation=1,
        action=1,
        reward=1.0,
        value=0.5,
        log_probability=-0.2,
        next_value=2.0,
        terminated=False,
        truncated=True,
    )

    batch = buffer.compute_batch(
        gamma=0.9,
        gae_lambda=0.8,
        device=torch.device("cpu"),
    )

    assert batch.advantages.tolist() == pytest.approx([0.5, 2.3])
    assert batch.returns.tolist() == pytest.approx([1.0, 2.8])


def test_rollout_buffer_rejects_invalid_checkpoint_state() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        observation=np.zeros(2, dtype=np.float32),
        action=0,
        reward=1.0,
        value=0.5,
        log_probability=-0.1,
        next_value=0.6,
        terminated=False,
        truncated=False,
    )
    state = buffer.state_dict()
    state["log_probabilities"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        buffer.load_state_dict(state)


def test_rollout_buffer_loads_legacy_single_stream_checkpoint() -> None:
    """The vector buffer should resume pre-vector scalar rollout checkpoints."""
    buffer = RolloutBuffer(capacity=2)
    buffer.load_state_dict(
        {
            "observations": [np.asarray([1.0], dtype=np.float32)],
            "actions": [np.asarray(0)],
            "rewards": [1.0],
            "values": [0.5],
            "log_probabilities": [-0.1],
            "next_values": [0.25],
            "terminated": [False],
            "truncated": [True],
        }
    )

    assert len(buffer) == 1
    assert buffer.rewards[0, 0] == 1.0


def test_rollout_buffer_round_trips_after_clear() -> None:
    """Cleared allocated storage should serialize as an empty rollout."""
    buffer = RolloutBuffer(capacity=2)
    buffer.add(
        observation=np.asarray([1.0], dtype=np.float32),
        action=0,
        reward=1.0,
        value=0.5,
        log_probability=-0.1,
        next_value=0.25,
        terminated=False,
        truncated=True,
    )
    buffer.clear()
    restored = RolloutBuffer(capacity=2)

    restored.load_state_dict(buffer.state_dict())

    assert len(restored) == 0
    assert restored.observations is None


def test_rollout_buffer_computes_gae_per_environment_stream() -> None:
    """Advantages from one environment must not propagate into another lane."""
    buffer = RolloutBuffer(capacity=2, num_envs=2)
    buffer.add_batch(
        observations=np.asarray([[0.0], [10.0]], dtype=np.float32),
        actions=np.asarray([0, 1]),
        rewards=np.asarray([1.0, 2.0]),
        values=np.asarray([0.0, 0.0]),
        log_probabilities=np.asarray([-0.1, -0.2]),
        next_values=np.asarray([0.0, 0.0]),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
    )
    buffer.add_batch(
        observations=np.asarray([[1.0], [11.0]], dtype=np.float32),
        actions=np.asarray([1, 0]),
        rewards=np.asarray([3.0, 4.0]),
        values=np.asarray([0.0, 0.0]),
        log_probabilities=np.asarray([-0.3, -0.4]),
        next_values=np.asarray([0.0, 0.0]),
        terminated=np.asarray([True, True]),
        truncated=np.asarray([False, False]),
    )

    batch = buffer.compute_batch(
        gamma=1.0,
        gae_lambda=1.0,
        device=torch.device("cpu"),
    )

    assert batch.advantages.tolist() == pytest.approx([4.0, 2.0, 3.0, 4.0])
    assert batch.observations[:, 0].tolist() == [0.0, 10.0, 1.0, 11.0]


def test_ppo_uses_a_registered_project_model_for_both_policy_types(
    tmp_path: Path,
) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("ppo") is PPOTrainer
    assert not MODEL_REGISTRY.is_registered("ppo_actor_critic")
    discrete_model = _TestPPOPolicy(
        {
            "input_dim": 3,
            "action_dim": 2,
            "continuous_actions": False,
        }
    )
    continuous_model = _TestPPOPolicy(
        {
            "input_dim": 3,
            "action_dim": 2,
            "continuous_actions": True,
        }
    )
    discrete_output = discrete_model(torch.zeros(5, 3))
    continuous_output = continuous_model(torch.zeros(5, 3))

    assert discrete_output["logits"].shape == (5, 2)
    assert discrete_output["value"].shape == (5,)
    assert continuous_output["mean"].shape == (5, 2)
    assert continuous_output["log_std"].shape == (5, 2)

    trainer = PPOTrainer(_config(tmp_path))
    trainer.setup()
    assert set(trainer.models) == {"policy"}
    trainer.close()


@pytest.mark.parametrize(
    "models",
    [
        None,
        {},
        {"policy": {}},
        {"policy": {"name": ""}},
    ],
)
def test_ppo_requires_an_explicit_project_model(
    tmp_path: Path,
    models: object,
) -> None:
    config = _config(tmp_path)
    if models is None:
        config.pop("models")
    else:
        config["models"] = models
    trainer = PPOTrainer(config)

    with pytest.raises(ValueError, match=r"requires models\.policy\.name"):
        trainer.setup()
    trainer.close()


def test_ppo_requires_a_torch_module_project_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["models"]["policy"]["name"] = "test_ppo_non_module"
    trainer = PPOTrainer(config)

    with pytest.raises(TypeError, match="policy must be a torch module"):
        trainer.setup()
    trainer.close()


def test_ppo_rejects_nonfinite_project_model_output(tmp_path: Path) -> None:
    trainer = PPOTrainer(_config(tmp_path))
    trainer.setup()

    class _NonfinitePolicy(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "logits": torch.full(
                    (observations.shape[0], 4),
                    float("nan"),
                ),
                "value": torch.zeros(observations.shape[0]),
            }

    trainer.models["policy"] = _NonfinitePolicy()
    with pytest.raises(FloatingPointError, match="must be finite"):
        trainer.select_actions(
            np.asarray([0, 1]),
            deterministic=True,
        )
    trainer.close()


def test_ppo_rejects_nonfinite_rewards_before_rollout(
    tmp_path: Path,
) -> None:
    trainer = PPOTrainer(_config(tmp_path))
    trainer.setup()

    with pytest.raises(FloatingPointError, match="rewards must be finite"):
        trainer.process_transition(
            Transition(
                observation=0,
                action=0,
                reward=float("nan"),
                next_observation=1,
                terminated=False,
                truncated=False,
                action_info={
                    "policy_action": 0,
                    "log_probability": 0.0,
                    "value": 0.0,
                },
            )
        )
    assert len(trainer.rollout_buffer) == 0
    trainer.close()


def test_ppo_rejects_nonfinite_update_before_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PPOTrainer(
        _config(
            tmp_path,
            total_timesteps=1,
            rollout_steps=1,
            update_epochs=1,
            minibatch_size=1,
        )
    )
    trainer.setup()
    optimizer_step = Mock(wraps=trainer.accelerator.optimizer_step)
    monkeypatch.setattr(
        trainer.accelerator,
        "optimizer_step",
        optimizer_step,
    )
    trainer.global_step = 1

    with pytest.raises(FloatingPointError, match="must be finite"):
        trainer.process_transition(
            Transition(
                observation=0,
                action=0,
                reward=1.0,
                next_observation=1,
                terminated=True,
                truncated=False,
                action_info={
                    "policy_action": 0,
                    "log_probability": -1000.0,
                    "value": 0.0,
                },
            )
        )
    optimizer_step.assert_not_called()
    trainer.close()


def test_ppo_collects_and_updates_independent_vector_rollouts(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(tmp_path)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    trainer = PPOTrainer(config)
    trainer.setup()
    no_grad_batches: list[int] = []
    distribution_and_value = trainer._distribution_and_value

    def recording_distribution_and_value(
        observations: torch.Tensor,
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        if not torch.is_grad_enabled():
            no_grad_batches.append(int(observations.shape[0]))
        return distribution_and_value(observations)

    trainer._distribution_and_value = recording_distribution_and_value
    trainer.perform_training()

    assert trainer.global_step == 8
    assert trainer.collector_step == 4
    assert trainer.update_step == 2
    assert len(trainer.rollout_buffer) == 0
    assert no_grad_batches.count(2) == 8
    trainer.close()


def test_ppo_updates_continuous_vector_rollouts(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path, total_timesteps=4)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "Pendulum-v1",
        "num_envs": 2,
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "Pendulum-v1",
    }
    trainer = PPOTrainer(config)
    trainer.setup()
    observations, _ = trainer.environment.reset_batch([23, 24])

    action_output = trainer.select_actions(observations, deterministic=False)

    assert len(action_output.actions) == 2
    assert all(
        trainer.environment.action_space.contains(action)
        for action in action_output.actions
    )
    assert all(
        np.asarray(info["policy_action"]).shape == (1,)
        and np.isfinite(info["log_probability"])
        for info in action_output.action_info
    )

    trainer.perform_training()

    assert trainer.global_step == 4
    assert trainer.collector_step == 2
    assert trainer.update_step == 1
    assert len(trainer.rollout_buffer) == 0
    trainer.close()


def test_ppo_builds_continuous_distribution_in_stable_float32(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["environment"] = {"name": "gymnasium", "id": "Pendulum-v1"}
    trainer = PPOTrainer(config)
    trainer.setup()

    class _HalfPrecisionPolicy(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            batch_size = observations.shape[0]
            return {
                "mean": torch.zeros(batch_size, 1).half(),
                "log_std": torch.full((batch_size, 1), -20.0).half(),
                "value": torch.zeros(batch_size).half(),
            }

    trainer.models["policy"] = _HalfPrecisionPolicy()
    distribution, _ = trainer._distribution_and_value(torch.zeros(2, 3))

    assert distribution.mean.dtype == torch.float32
    assert distribution.scale.dtype == torch.float32
    assert torch.all(distribution.scale > 0.0)
    trainer.close()


def test_ppo_updates_from_a_terminal_rollout(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = PPOTrainer(_config(tmp_path, total_timesteps=1, rollout_steps=10))
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=23)
    action_output = trainer.select_action(observation, deterministic=False)
    next_observation, reward, terminated, truncated, info = trainer.environment.step(
        action_output.action
    )
    parameters_before = [
        parameter.detach().clone() for parameter in trainer.models["policy"].parameters()
    ]
    trainer.global_step = 1

    metrics = trainer.process_transition(
        Transition(
            observation=observation,
            action=action_output.action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=True,
            info=info,
            action_info=action_output.info,
        )
    )

    assert metrics is not None
    assert metrics["ppo/rollout_size"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            trainer.models["policy"].parameters(),
            strict=True,
        )
    )
    assert len(trainer.rollout_buffer) == 0
    trainer.close()


def test_ppo_continuous_actions_respect_box_bounds(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path)
    config["environment"] = {"name": "gymnasium", "id": "Pendulum-v1"}
    trainer = PPOTrainer(config)
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=23)

    sampled = trainer.select_action(observation, deterministic=False)
    deterministic = trainer.select_action(observation, deterministic=True)

    assert trainer.environment.action_space.contains(sampled.action)
    assert trainer.environment.action_space.contains(deterministic.action)
    assert np.asarray(sampled.info["policy_action"]).shape == (1,)
    trainer.close()


def test_ppo_supports_nonzero_discrete_starts(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = PPOTrainer(_config(tmp_path))
    trainer.setup()
    trainer.environment.observation_space = Discrete(16, start=3)
    trainer.evaluation_environment.observation_space = Discrete(16, start=3)
    trainer.environment.action_space = Discrete(4, start=7)
    trainer.evaluation_environment.action_space = Discrete(4, start=7)

    action_output = trainer.select_action(3, deterministic=True)

    assert trainer.environment.action_space.contains(action_output.action)
    assert action_output.info["policy_action"] == action_output.action - 7
    trainer.close()


def test_ppo_rejects_integer_box_actions(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = PPOTrainer(_config(tmp_path))
    trainer.setup_accelerator()
    trainer.environment = trainer.evaluation_environment = SimpleNamespace(
        observation_space=Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        action_space=Box(0, 3, shape=(1,), dtype=np.int64),
    )

    with pytest.raises(TypeError, match="floating-point Box"):
        trainer.setup_algorithm()


def test_ppo_checkpoint_restores_partial_rollout(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = PPOTrainer(_config(tmp_path, rollout_steps=10))
    trainer.setup()
    observation, _ = trainer.environment.reset(seed=23)
    action_output = trainer.select_action(observation, deterministic=False)
    next_observation, reward, terminated, truncated, info = trainer.environment.step(
        action_output.action
    )
    trainer.global_step = 1
    metrics = trainer.process_transition(
        Transition(
            observation=observation,
            action=action_output.action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=True,
            info=info,
            action_info=action_output.info,
        )
    )
    assert metrics is None
    checkpoint_path = trainer.save_checkpoint("ppo.pth")
    assert checkpoint_path is not None

    restored = PPOTrainer(_config(tmp_path, rollout_steps=10))
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert len(restored.rollout_buffer) == 1
    assert np.array_equal(
        restored.rollout_buffer.rewards[:1],
        trainer.rollout_buffer.rewards[:1],
    )
    trainer.close()
    restored.close()


def test_ppo_checkpoint_separates_resumed_vector_rollout_fragments(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(tmp_path, total_timesteps=100, rollout_steps=10)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    trainer = PPOTrainer(config)
    trainer.setup()
    trainer.rollout_buffer.add_batch(
        observations=np.asarray([0, 1]),
        actions=np.asarray([0, 1]),
        rewards=np.asarray([1.0, 2.0]),
        values=np.asarray([0.2, 0.3]),
        log_probabilities=np.asarray([-0.1, -0.2]),
        next_values=np.asarray([0.4, 0.5]),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
    )
    checkpoint_path = trainer.save_checkpoint("ppo-vector.pth")
    assert checkpoint_path is not None

    restored = PPOTrainer(config)
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert restored.rollout_buffer.truncated[0].tolist() == [True, False]
    assert restored.rollout_buffer.next_values[0].tolist() == pytest.approx(
        [0.4, 0.5]
    )
    trainer.close()
    restored.close()
