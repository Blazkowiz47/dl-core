"""Tests for rollout estimation and proximal policy optimization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from dl_core import load_builtin_components
from dl_core.core import RolloutBuffer, TRAINER_REGISTRY, Transition
from dl_core.models import PPOActorCritic
from dl_core.trainers import PPOTrainer


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
                "name": "ppo_actor_critic",
                "hidden_sizes": [8],
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


def test_ppo_is_registered_and_builtin_model_supports_both_policy_types(
    tmp_path: Path,
) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("ppo") is PPOTrainer
    discrete_model = PPOActorCritic(
        {
            "input_dim": 3,
            "action_dim": 2,
            "continuous_actions": False,
            "hidden_sizes": [4],
        }
    )
    continuous_model = PPOActorCritic(
        {
            "input_dim": 3,
            "action_dim": 2,
            "continuous_actions": True,
            "hidden_sizes": [4],
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
