"""Tests for reinforcement-learning environment contracts."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.spaces import Box, Discrete

from dl_core import load_builtin_components
from dl_core.core import (
    BatchActionOutput,
    BatchedEnvironment,
    ENVIRONMENT_REGISTRY,
    Environment,
    EpisodeResult,
    Transition,
    register_environment,
)
from dl_core.environments import (
    ActionHistoryWrapper,
    GymnasiumEnvironment,
    GymnasiumVectorEnvironment,
    make_environment,
)


def test_environment_factory_can_be_imported_before_core_components() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dl_core.environments import make_environment; "
            "assert callable(make_environment)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gymnasium_environment_is_registered_and_usable() -> None:
    """The built-in adapter should expose the standard Gymnasium interaction API."""
    load_builtin_components()

    environment = make_environment(
        {"name": "gymnasium", "id": "FrozenLake-v1", "kwargs": {"is_slippery": False}}
    )
    observation, info = environment.reset(seed=7)
    next_observation, reward, terminated, truncated, step_info = environment.step(1)

    assert isinstance(environment, GymnasiumEnvironment)
    assert isinstance(environment, Environment)
    assert isinstance(environment.observation_space, Discrete)
    assert isinstance(environment.action_space, Discrete)
    assert isinstance(observation, int)
    assert isinstance(next_observation, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert isinstance(step_info, dict)
    environment.close()


def test_local_environment_registration_uses_the_shared_factory() -> None:
    """Extension packages should be able to register Gymnasium-compatible classes."""

    @register_environment("test_discrete_environment")
    class TestEnvironment(GymnasiumEnvironment):
        pass

    environment = make_environment(
        {"name": "test_discrete_environment", "id": "FrozenLake-v1"}
    )

    assert isinstance(environment, TestEnvironment)
    assert ENVIRONMENT_REGISTRY.get_class("test_discrete_environment") is TestEnvironment
    environment.close()
    ENVIRONMENT_REGISTRY.unregister(
        "test_discrete_environment",
        expected_class=TestEnvironment,
    )


def test_gymnasium_vector_environment_preserves_terminal_observations() -> None:
    """Same-step autoreset must expose both reset and final observations."""
    load_builtin_components()
    vector_environment = make_environment(
        {
            "name": "gymnasium_vector",
            "id": "CartPole-v1",
            "num_envs": 2,
            "kwargs": {"max_episode_steps": 1},
        }
    )
    assert isinstance(vector_environment, GymnasiumVectorEnvironment)
    assert isinstance(vector_environment.env, AsyncVectorEnv)
    assert vector_environment.supports_async_step is True
    environment = BatchedEnvironment(vector_environment)
    observations, _ = environment.reset_batch([3, 4])

    environment.step_batch_async([0, 1])
    (
        reset_observations,
        rewards,
        terminated,
        truncated,
        _,
        final_observations,
    ) = environment.step_batch_wait()

    assert observations.shape == (2, 4)
    assert reset_observations.shape == (2, 4)
    assert rewards.tolist() == [1.0, 1.0]
    assert not terminated.any()
    assert truncated.all()
    assert all(np.asarray(observation).shape == (4,) for observation in final_observations)
    assert not np.array_equal(final_observations[0], reset_observations[0])
    environment.close()


def test_gymnasium_vector_environment_accepts_explicit_sync_mode() -> None:
    load_builtin_components()
    vector_environment = make_environment(
        {
            "name": "gymnasium_vector",
            "id": "CartPole-v1",
            "num_envs": 2,
            "vectorization_mode": "sync",
        }
    )

    assert isinstance(vector_environment.env, SyncVectorEnv)
    assert vector_environment.supports_async_step is False
    environment = BatchedEnvironment(vector_environment)
    environment.reset_batch([3, 4])
    environment.step_batch_async([0, 1])
    _, rewards, _, _, _, _ = environment.step_batch_wait()

    assert rewards.tolist() == [1.0, 1.0]
    with np.testing.assert_raises_regex(RuntimeError, "No environment step"):
        environment.step_batch_wait()
    vector_environment.close()


def test_batched_scalar_environment_accepts_split_step_api() -> None:
    load_builtin_components()
    scalar_environment = make_environment(
        {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
        }
    )
    environment = BatchedEnvironment(scalar_environment)
    observations, _ = environment.reset_batch([7])
    environment.step_batch_async([1])
    next_observations, rewards, terminated, truncated, _, _ = (
        environment.step_batch_wait()
    )

    assert observations.tolist() == [0]
    assert next_observations.tolist() == [4]
    assert rewards.tolist() == [0.0]
    assert not terminated.any()
    assert not truncated.any()
    environment.close()


def test_batched_environment_rejects_a_second_pending_step() -> None:
    load_builtin_components()
    vector_environment = make_environment(
        {
            "name": "gymnasium_vector",
            "id": "CartPole-v1",
            "num_envs": 2,
        }
    )
    environment = BatchedEnvironment(vector_environment)
    environment.reset_batch([3, 4])
    environment.step_batch_async([0, 1])

    with np.testing.assert_raises_regex(RuntimeError, "already pending"):
        environment.step_batch_async([1, 0])

    environment.step_batch_wait()
    environment.close()


def test_action_history_augments_scalar_discrete_observations() -> None:
    load_builtin_components()
    environment = make_environment(
        {
            "name": "gymnasium",
            "id": "FrozenLake-v1",
            "kwargs": {"is_slippery": False},
            "action_history": {"length": 2},
        }
    )
    observation, _ = environment.reset(seed=7)
    next_observation, _, _, _, _ = environment.step(1)

    assert isinstance(environment, ActionHistoryWrapper)
    assert isinstance(environment.observation_space, Box)
    assert environment.observation_space.shape == (24,)
    assert observation[:16].sum() == 1.0
    assert not observation[16:].any()
    assert not next_observation[16:20].any()
    assert next_observation[20:].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert environment.observation_space.contains(next_observation)
    environment.close()


def test_action_history_preserves_vector_final_history_and_resets_lanes() -> None:
    load_builtin_components()
    vector_environment = make_environment(
        {
            "name": "gymnasium_vector",
            "id": "CartPole-v1",
            "num_envs": 2,
            "kwargs": {"max_episode_steps": 1},
            "action_history": {"length": 2},
        }
    )
    assert isinstance(vector_environment, ActionHistoryWrapper)
    environment = BatchedEnvironment(vector_environment)
    observations, _ = environment.reset_batch([3, 4])
    (
        reset_observations,
        _,
        _,
        truncated,
        _,
        final_observations,
    ) = environment.step_batch([0, 1])

    assert observations.shape == (2, 8)
    assert truncated.all()
    assert not reset_observations[:, 4:].any()
    assert final_observations[0][4:].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert final_observations[1][4:].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert all(
        environment.observation_space.contains(observation)
        for observation in final_observations
    )
    environment.close()


def test_action_history_validates_box_values_without_lossy_casts() -> None:
    class IntegerBoxEnvironment:
        observation_space = Box(0, 10, shape=(1,), dtype=np.int32)
        action_space = Box(0, 1, shape=(1,), dtype=np.int32)

        def __init__(self) -> None:
            self.step_calls = 0
            self.observation = np.asarray([2], dtype=np.int32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return self.observation, {}

        def step(self, action):
            self.step_calls += 1
            return np.asarray([3], dtype=np.int32), 0.0, False, False, {}

        def render(self):
            return None

        def close(self):
            return None

    source_environment = IntegerBoxEnvironment()
    environment = ActionHistoryWrapper(
        source_environment,
        history_length=2,
    )
    observation, _ = environment.reset()
    next_observation, _, _, _, _ = environment.step(
        np.asarray([1], dtype=np.int32)
    )

    assert observation.tolist() == [2.0, 0.0, 0.0]
    assert next_observation.tolist() == [3.0, 0.0, 1.0]
    with np.testing.assert_raises_regex(ValueError, "outside"):
        environment.step(np.asarray([1.5], dtype=np.float64))
    assert source_environment.step_calls == 1
    source_environment.observation = np.asarray([2.5], dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "outside"):
        environment.reset()


def test_action_history_rejects_invalid_length_before_environment_creation() -> None:
    with patch.object(ENVIRONMENT_REGISTRY, "get") as get_environment:
        with np.testing.assert_raises_regex(TypeError, "must be an integer"):
            make_environment(
                {
                    "name": "gymnasium",
                    "id": "CartPole-v1",
                    "action_history": {"length": 1.5},
                }
            )
    get_environment.assert_not_called()


def test_rl_value_objects_preserve_transition_and_episode_state() -> None:
    """Shared RL value objects should retain terminal and metric information."""
    transition = Transition(
        observation=0,
        action=1,
        reward=1.0,
        next_observation=2,
        terminated=False,
        truncated=True,
    )
    result = EpisodeResult(
        episode=3,
        episode_return=2.5,
        length=4,
        terminated=True,
        truncated=False,
        final_info={"is_success": True},
    )

    assert transition.done is True
    assert result.episode_return == 2.5
    assert result.final_info["is_success"] is True


def test_batch_action_output_aligns_optional_metadata() -> None:
    output = BatchActionOutput(actions=[0, 1])

    assert output.action_info == [{}, {}]
    with np.testing.assert_raises_regex(ValueError, "must align"):
        BatchActionOutput(actions=[0, 1], action_info=[{}])
