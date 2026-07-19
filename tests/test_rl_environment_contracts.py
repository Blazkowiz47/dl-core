"""Tests for reinforcement-learning environment contracts."""

from __future__ import annotations

from gymnasium.spaces import Discrete

from dl_core import load_builtin_components
from dl_core.core import (
    ENVIRONMENT_REGISTRY,
    Environment,
    EpisodeResult,
    Transition,
    register_environment,
)
from dl_core.environments import GymnasiumEnvironment, make_environment


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
