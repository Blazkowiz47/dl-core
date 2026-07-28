"""Tests for replay storage and the deep Q-network trainer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from dl_core import load_builtin_components
from dl_core.core import (
    MODEL_REGISTRY,
    ReplayBuffer,
    TRAINER_REGISTRY,
    Transition,
    TransitionBatch,
    register_model,
)
from dl_core.trainers import DQNTrainer


@register_model("test_dqn_q_network")
class _TestDQNQNetwork(torch.nn.Module):
    """Small project-style Q-network used only by trainer tests."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, action_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value per action."""
        return self.network(observations.reshape(observations.shape[0], -1))


@register_model("test_dqn_non_module")
class _TestDQNNonModule:
    """Invalid project registration used to verify setup diagnostics."""

    def __init__(self, config: dict[str, object]):
        del config


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
        "models": {"q_network": {"name": "test_dqn_q_network"}},
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
    assert torch.equal(first_sample.discounts, restored_sample.discounts)


def test_replay_buffer_loads_legacy_one_step_state() -> None:
    buffer = ReplayBuffer(3, (), ())
    buffer.add(
        Transition(
            observation=0.0,
            action=1.0,
            reward=2.0,
            next_observation=1.0,
            terminated=False,
            truncated=False,
        )
    )
    state = buffer.state_dict()
    for key in ("gamma", "n_step", "discounts"):
        state.pop(key)
    restored = ReplayBuffer(3, (), ())
    restored.load_state_dict(state)

    assert restored.discounts[0] == pytest.approx(0.99)


@pytest.mark.parametrize(("missing_key", "message"), [
    ("gamma", "gamma does not match"),
    ("discounts", "discounts shape"),
])
def test_n_step_replay_rejects_incomplete_checkpoint_schema(
    missing_key: str,
    message: str,
) -> None:
    buffer = ReplayBuffer(3, (), (), gamma=0.9, n_step=2)
    buffer.add(
        Transition(
            observation=0.0,
            action=1.0,
            reward=2.0,
            next_observation=1.0,
            terminated=True,
            truncated=False,
        )
    )
    state = buffer.state_dict()
    state.pop(missing_key)

    with pytest.raises(ValueError, match=message):
        ReplayBuffer(3, (), (), gamma=0.9, n_step=2).load_state_dict(state)


def test_replay_buffer_add_batch_wraps_and_retains_latest_transitions() -> None:
    """Batch insertion should preserve ring ordering without scalar loops."""
    buffer = ReplayBuffer(3, (1,), (), action_dtype=np.int64, seed=3)
    buffer.add_batch(
        TransitionBatch(
            observations=np.arange(5, dtype=np.float32).reshape(5, 1),
            actions=np.arange(5, dtype=np.int64),
            rewards=np.arange(5, dtype=np.float32),
            next_observations=np.arange(1, 6, dtype=np.float32).reshape(5, 1),
            terminated=np.asarray([False, False, False, False, True]),
            truncated=np.zeros(5, dtype=np.bool_),
        )
    )

    assert len(buffer) == 3
    assert buffer.position == 2
    assert buffer.observations[:, 0].tolist() == [3.0, 4.0, 2.0]
    assert buffer.actions.tolist() == [3, 4, 2]


def test_replay_buffer_builds_discounted_n_step_returns() -> None:
    buffer = ReplayBuffer(
        8,
        (),
        (),
        action_dtype=np.int64,
        gamma=0.5,
        n_step=3,
    )
    for index, reward in enumerate((1.0, 2.0, 4.0, 8.0)):
        buffer.add(
            Transition(
                observation=index,
                action=index % 2,
                reward=reward,
                next_observation=index + 1,
                terminated=index == 3,
                truncated=False,
            )
        )

    assert len(buffer) == 4
    assert buffer.observations[:4].tolist() == [0, 1, 2, 3]
    assert buffer.rewards[:4].tolist() == pytest.approx([3.0, 6.0, 8.0, 8.0])
    assert buffer.discounts[:4].tolist() == pytest.approx(
        [0.125, 0.125, 0.25, 0.5]
    )
    assert buffer.next_observations[:4].tolist() == [3, 4, 4, 4]
    assert buffer.terminated[:4].tolist() == [False, True, True, True]
    assert buffer.pending_transitions == [[]]


def test_n_step_replay_keeps_vector_lanes_independent() -> None:
    buffer = ReplayBuffer(
        8,
        (),
        (),
        action_dtype=np.int64,
        gamma=1.0,
        n_step=2,
    )
    first_added = buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([0, 10]),
            actions=np.asarray([0, 1]),
            rewards=np.asarray([1.0, 10.0], dtype=np.float32),
            next_observations=np.asarray([1, 11]),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )
    second_added = buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([1, 11]),
            actions=np.asarray([1, 0]),
            rewards=np.asarray([2.0, 20.0], dtype=np.float32),
            next_observations=np.asarray([2, 12]),
            terminated=np.asarray([True, False]),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )

    assert buffer.rewards[:3].tolist() == [3.0, 2.0, 30.0]
    assert buffer.observations[:3].tolist() == [0, 1, 10]
    assert buffer.terminated[:3].tolist() == [True, True, False]
    assert [len(pending) for pending in buffer.pending_transitions] == [0, 1]
    assert first_added.tolist() == [0, 0]
    assert second_added.tolist() == [2, 1]


def test_n_step_replay_discards_incomplete_windows_when_restored() -> None:
    buffer = ReplayBuffer(8, (), (), gamma=0.9, n_step=3)
    for index in range(2):
        buffer.add(
            Transition(
                observation=index,
                action=0.0,
                reward=1.0,
                next_observation=index + 1,
                terminated=False,
                truncated=False,
            )
        )
    state = buffer.state_dict()
    restored = ReplayBuffer(8, (), (), gamma=0.9, n_step=3)
    restored.load_state_dict(state)

    assert len(restored) == 0
    assert restored.pending_transitions == []


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("position", 0, "position is inconsistent"),
        ("actions", np.asarray([0.0], dtype=np.float32), "actions dtype"),
    ],
)
def test_replay_buffer_rejects_inconsistent_state(
    key: str,
    value: object,
    message: str,
) -> None:
    buffer = ReplayBuffer(3, (2,), (), action_dtype=np.int64)
    buffer.add(
        Transition(
            observation=np.asarray([0.0, 1.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_observation=np.asarray([1.0, 2.0], dtype=np.float32),
            terminated=False,
            truncated=False,
        )
    )
    state = buffer.state_dict()
    state[key] = value

    with pytest.raises(ValueError, match=message):
        ReplayBuffer(3, (2,), (), action_dtype=np.int64).load_state_dict(state)


def test_dqn_uses_a_registered_project_model(tmp_path: Path) -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.get_class("dqn") is DQNTrainer
    assert not MODEL_REGISTRY.is_registered("dqn_mlp")
    model = _TestDQNQNetwork({"input_dim": 3, "action_dim": 2})
    assert model(torch.zeros(5, 3)).shape == (5, 2)
    assert model(torch.zeros(5, 1, 3)).shape == (5, 2)

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
    "models",
    [
        None,
        {},
        {"q_network": {}},
        {"q_network": {"name": ""}},
    ],
)
def test_dqn_requires_an_explicit_project_model(
    tmp_path: Path,
    models: object,
) -> None:
    config = _config(tmp_path)
    if models is None:
        config.pop("models")
    else:
        config["models"] = models
    trainer = DQNTrainer(config)

    with pytest.raises(
        ValueError,
        match=r"requires models\.q_network\.name",
    ):
        trainer.setup()
    trainer.close()


def test_dqn_requires_a_torch_module_project_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["models"]["q_network"]["name"] = "test_dqn_non_module"
    trainer = DQNTrainer(config)

    with pytest.raises(TypeError, match="q_network must be a torch module"):
        trainer.setup()
    trainer.close()


def test_compiled_dqn_places_eager_copy_on_accelerator_device(
    tmp_path: Path,
) -> None:
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup_accelerator()
    trainer.accelerator.compile_models = {"online"}
    trainer.accelerator.get_device = lambda: torch.device("meta")
    trainer.setup_environment()
    trainer.setup_algorithm()

    assert trainer.eager_online_model is not None
    assert next(trainer.eager_online_model.parameters()).device.type == "meta"
    trainer.close()


def test_dqn_rejects_nonfinite_rewards_before_replay(
    tmp_path: Path,
) -> None:
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    trainer.global_step = 1

    with pytest.raises(FloatingPointError, match="rewards must be finite"):
        trainer.process_transition(
            Transition(
                observation=0,
                action=0,
                reward=float("nan"),
                next_observation=1,
                terminated=False,
                truncated=False,
            )
        )
    assert len(trainer.replay_buffer) == 0
    trainer.close()


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        (torch.zeros(2), TypeError, r"\[batch, actions\]"),
        (torch.zeros(1, 4), ValueError, "batch dimension"),
        (torch.zeros(2, 3), ValueError, "action dimension"),
        (torch.zeros(2, 4, dtype=torch.int64), TypeError, "floating-point"),
        ({"other": torch.zeros(2, 4)}, TypeError, r"\[batch, actions\]"),
    ],
)
def test_dqn_validates_project_model_output_contract(
    tmp_path: Path,
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()

    class _ContractQNetwork(torch.nn.Module):
        def forward(self, observations: torch.Tensor) -> Any:
            del observations
            return output

    with pytest.raises(error, match=message):
        trainer._q_values(
            _ContractQNetwork(),
            torch.zeros(2, 16),
        )
    trainer.close()


def test_dqn_accepts_mapping_project_model_output(tmp_path: Path) -> None:
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()

    class _MappingQNetwork(torch.nn.Module):
        def forward(
            self,
            observations: torch.Tensor,
        ) -> Mapping[str, torch.Tensor]:
            return {"q_values": torch.zeros(observations.shape[0], 4)}

    q_values = trainer._q_values(
        _MappingQNetwork(),
        torch.zeros(2, 16),
    )

    assert q_values.shape == (2, 4)
    trainer.close()


def test_dqn_rejects_nonfinite_loss_before_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    optimizer_step = Mock(wraps=trainer.accelerator.optimizer_step)
    monkeypatch.setattr(
        trainer.accelerator,
        "optimizer_step",
        optimizer_step,
    )

    def nonfinite_forward(
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return (
            trainer.models["online"].network(observations.float())
            * float("nan")
        )

    trainer.models["online"].forward = nonfinite_forward
    trainer.global_step = 1
    with pytest.raises(FloatingPointError, match="DQN loss must be finite"):
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
    optimizer_step.assert_not_called()
    trainer.close()


def test_dqn_compiles_only_gradient_enabled_online_forwards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_builtin_components()
    compiled_grad_modes: list[bool] = []
    forward_hook_grad_modes: list[bool] = []
    eager_hook_grad_modes: list[bool] = []

    def compile_with_grad_guard(
        model: torch.nn.Module,
        *,
        mode: str,
    ) -> None:
        del mode
        eager_call = model._call_impl

        def compiled_call(*args: object, **kwargs: object) -> object:
            compiled_grad_modes.append(torch.is_grad_enabled())
            if not torch.is_grad_enabled():
                raise RuntimeError("Inference must bypass the compiled call")
            return eager_call(*args, **kwargs)

        model._compiled_call_impl = compiled_call

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.nn.Module, "compile", compile_with_grad_guard)
    config = _config(
        tmp_path,
        total_timesteps=4,
        actor_model_copies=2,
    )
    config["accelerator"] = {
        "type": "single_gpu",
        "compile_models": ["online"],
    }
    trainer = DQNTrainer(config)

    trainer.setup()
    trainer.select_actions(np.asarray([0]), deterministic=False)
    hook = trainer.models["online"].register_forward_hook(
        lambda _model, _inputs, _output: forward_hook_grad_modes.append(
            torch.is_grad_enabled()
        )
    )
    assert trainer.eager_online_model is not None
    eager_hook = trainer.eager_online_model.register_forward_hook(
        lambda _model, _inputs, _output: eager_hook_grad_modes.append(
            torch.is_grad_enabled()
        )
    )
    actor_hook_grad_modes: list[bool] = []
    actor_hooks = [
        actor_model.register_forward_hook(
            lambda _model, _inputs, _output: (
                actor_hook_grad_modes.append(torch.is_grad_enabled())
            )
        )
        for actor_model in trainer.actor_models
    ]
    trainer.perform_training()

    assert compiled_grad_modes
    assert all(compiled_grad_modes)
    assert any(forward_hook_grad_modes)
    assert all(forward_hook_grad_modes)
    assert eager_hook_grad_modes
    assert all(not grad_enabled for grad_enabled in eager_hook_grad_modes)
    assert actor_hook_grad_modes
    assert all(not grad_enabled for grad_enabled in actor_hook_grad_modes)
    assert all(
        actor_model._compiled_call_impl is None
        for actor_model in trainer.actor_models
    )
    hook.remove()
    eager_hook.remove()
    for actor_hook in actor_hooks:
        actor_hook.remove()
    trainer.close()


def test_dqn_trains_with_action_history_observations(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path, total_timesteps=4)
    config["environment"]["action_history"] = {"length": 2}
    trainer = DQNTrainer(config)
    trainer.setup()
    trainer.perform_training()

    assert isinstance(trainer.environment.observation_space, Box)
    assert trainer.environment.observation_space.shape == (24,)
    assert trainer.global_step == 4
    assert trainer.update_step == 4
    trainer.close()


def test_dqn_requires_identical_box_observation_spaces(tmp_path: Path) -> None:
    load_builtin_components()
    config = _config(tmp_path)
    config["environment"] = {"name": "gymnasium", "id": "CartPole-v1"}
    trainer = DQNTrainer(config)
    trainer.setup_accelerator()
    trainer.setup_environment()
    trainer.evaluation_environment.observation_space = Box(
        low=-2.0,
        high=2.0,
        shape=(4,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="observation spaces must match"):
        trainer.setup_algorithm()
    trainer.close()


def test_dqn_preserves_box_observation_dtype_in_replay(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup_accelerator()
    trainer.environment = trainer.evaluation_environment = SimpleNamespace(
        observation_space=Box(0, 255, shape=(2, 2), dtype=np.uint8),
        action_space=Discrete(2),
    )

    trainer.setup_algorithm()

    assert trainer.replay_buffer.observation_dtype == np.dtype(np.uint8)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor_model_copies": -1}, "actor_model_copies cannot be negative"),
        (
            {"actor_model_sync_frequency": 0},
            "actor_model_sync_frequency must be positive",
        ),
    ],
)
def test_dqn_validates_actor_model_copy_configuration(
    tmp_path: Path,
    overrides: dict[str, int],
    message: str,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, **overrides))

    with pytest.raises(ValueError, match=message):
        trainer.setup()
    trainer.close()


def test_dqn_supports_nonzero_discrete_starts(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path))
    trainer.setup()
    trainer.environment.observation_space = Discrete(16, start=3)
    trainer.evaluation_environment.observation_space = Discrete(16, start=3)
    trainer.environment.action_space = Discrete(4, start=7)
    trainer.evaluation_environment.action_space = Discrete(4, start=7)

    output = trainer.select_actions(
        np.asarray([3, 4], dtype=np.int64),
        deterministic=True,
    )
    trainer.global_step = 2
    logs = trainer.process_transition_batch(
        TransitionBatch(
            observations=np.asarray([3, 4], dtype=np.int64),
            actions=np.asarray(output.actions, dtype=np.int64),
            rewards=np.ones(2, dtype=np.float32),
            next_observations=np.asarray([4, 5], dtype=np.int64),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )

    assert all(
        trainer.environment.action_space.contains(action)
        for action in output.actions
    )
    assert len(logs) == 2
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


def test_dqn_uses_n_step_rewards_and_bootstrap_discount(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, n_step=2))
    trainer.setup()
    with torch.no_grad():
        trainer.models["online"].network[-1].weight.zero_()
        trainer.models["online"].network[-1].bias.zero_()
        trainer.models["target"].network[-1].weight.zero_()
        trainer.models["target"].network[-1].bias.copy_(
            torch.tensor([2.0, 0.0, 0.0, 0.0])
        )
    trainer.global_step = 1
    first_metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )
    trainer.global_step = 2
    metrics = trainer.process_transition(
        Transition(
            observation=1,
            action=1,
            reward=2.0,
            next_observation=2,
            terminated=False,
            truncated=False,
        )
    )

    assert first_metrics is None
    assert metrics is not None
    assert metrics["dqn/target_q_mean"] == pytest.approx(4.42)
    trainer.close()


@pytest.mark.parametrize(
    ("double_dqn", "expected_target"),
    [(True, 1.9), (False, 5.5)],
)
def test_dqn_selects_double_or_standard_bootstrap_target(
    tmp_path: Path,
    double_dqn: bool,
    expected_target: float,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, double_dqn=double_dqn))
    trainer.setup()
    with torch.no_grad():
        trainer.models["online"].network[-1].weight.zero_()
        trainer.models["online"].network[-1].bias.copy_(
            torch.tensor([0.0, 3.0, 0.0, 0.0])
        )
        trainer.models["target"].network[-1].weight.zero_()
        trainer.models["target"].network[-1].bias.copy_(
            torch.tensor([5.0, 1.0, 0.0, 0.0])
        )
    trainer.global_step = 1

    metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )

    assert metrics is not None
    assert metrics["dqn/target_q_mean"] == pytest.approx(expected_target)
    trainer.close()


def test_dqn_honors_autocast_and_target_schedule_between_updates(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(
        _config(tmp_path, train_frequency=2, target_update_frequency=3)
    )
    trainer.setup()
    autocast_calls = 0

    @contextmanager
    def recording_autocast() -> Iterator[None]:
        nonlocal autocast_calls
        autocast_calls += 1
        yield

    trainer.accelerator.autocast_context = recording_autocast
    trainer.select_action(0, deterministic=True)
    trainer.global_step = 2
    update_metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )
    with torch.no_grad():
        for parameter in trainer.models["online"].parameters():
            parameter.fill_(1.0)
        for parameter in trainer.models["target"].parameters():
            parameter.zero_()
    trainer.global_step = 3

    metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )

    assert update_metrics is not None
    assert metrics is None
    assert autocast_calls == 2
    for target_parameter, online_parameter in zip(
        trainer.models["target"].parameters(),
        trainer.models["online"].parameters(),
        strict=True,
    ):
        assert torch.equal(target_parameter, online_parameter)
    trainer.close()


def test_dqn_batches_vector_inference_replay_and_update_schedules(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(
        tmp_path,
        total_timesteps=8,
        max_episode_steps=2,
        train_frequency=2,
    )
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
    trainer = DQNTrainer(config)
    trainer.setup()
    inference_batches: list[int] = []
    events: list[str] = []
    update_metrics: list[dict[str, float]] = []
    callback_global_steps: list[int] = []
    original_q_values = trainer._q_values
    step_batch_async = trainer.environment.step_batch_async
    step_batch_wait = trainer.environment.step_batch_wait
    process_transition_batch = trainer.process_transition_batch
    on_update_end = trainer.callbacks.on_update_end

    def recording_q_values(
        model: torch.nn.Module,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if not model.training:
            inference_batches.append(int(observations.shape[0]))
        return original_q_values(model, observations)

    def recording_step_batch_async(actions: list[int]) -> None:
        events.append("dispatch")
        step_batch_async(actions)

    def recording_step_batch_wait():
        events.append("wait")
        return step_batch_wait()

    def recording_process_transition_batch(
        transitions: TransitionBatch,
    ) -> list[dict[str, float]]:
        events.append("update")
        return process_transition_batch(transitions)

    def recording_update_end(
        update: int,
        logs: dict[str, float] | None = None,
    ) -> None:
        update_metrics.append(logs or {})
        callback_global_steps.append(trainer.global_step)
        on_update_end(update, logs)

    trainer._q_values = recording_q_values
    trainer.environment.step_batch_async = recording_step_batch_async
    trainer.environment.step_batch_wait = recording_step_batch_wait
    trainer.process_transition_batch = recording_process_transition_batch
    trainer.callbacks.on_update_end = recording_update_end
    trainer.perform_training()

    assert trainer.global_step == 8
    assert trainer.collector_step == 4
    assert trainer.update_step == 4
    assert len(trainer.replay_buffer) == 8
    assert inference_batches.count(2) == 4
    assert events == [
        "dispatch",
        "wait",
        "dispatch",
        "update",
        "wait",
        "dispatch",
        "update",
        "wait",
        "dispatch",
        "update",
        "wait",
        "update",
    ]
    assert [metrics["global_step"] for metrics in update_metrics] == [
        2.0,
        4.0,
        6.0,
        8.0,
    ]
    assert callback_global_steps == [2, 4, 6, 8]
    assert all(
        metrics["rl/collection_overlap_enabled"] == 1.0
        and metrics["rl/timing/learner_update_ms"] >= 0.0
        and metrics["rl/timing/environment_wait_ms"] >= 0.0
        and metrics["dqn/timing/transition_validation_ms"] >= 0.0
        and metrics["dqn/timing/replay_add_ms"] >= 0.0
        and metrics["dqn/timing/replay_sample_ms"] >= 0.0
        and metrics["dqn/timing/model_update_ms"] >= 0.0
        and metrics["dqn/timing/actor_sync_ms"] >= 0.0
        for metrics in update_metrics
    )
    trainer.close()


def test_dqn_actor_models_balance_environment_shards(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, actor_model_copies=2))
    trainer.setup()

    trainer.select_actions(
        np.asarray([0], dtype=np.int64),
        deterministic=False,
    )
    with torch.no_grad():
        for action_index, actor_model in enumerate(trainer.actor_models):
            actor_model.network[-1].weight.zero_()
            actor_model.network[-1].bias.zero_()
            actor_model.network[-1].bias[action_index] = 1.0
    inference_batches: list[int] = []
    handles = [
        actor_model.register_forward_hook(
            lambda _model, inputs, _output: inference_batches.append(
                int(inputs[0].shape[0])
            )
        )
        for actor_model in trainer.actor_models
    ]
    output = trainer.select_actions(
        np.asarray([0, 1, 2, 3, 0], dtype=np.int64),
        deterministic=False,
    )
    with torch.no_grad():
        trainer.models["online"].network[-1].weight.zero_()
        trainer.models["online"].network[-1].bias.zero_()
        trainer.models["online"].network[-1].bias[3] = 1.0
    evaluation_output = trainer.select_actions(
        np.asarray([0, 1], dtype=np.int64),
        deterministic=True,
    )

    assert output.actions == [0, 0, 0, 1, 1]
    assert evaluation_output.actions == [3, 3]
    assert inference_batches == [3, 2]
    assert trainer.actor_policy_version == 1
    assert len(trainer.actor_models) == 2
    assert all(not actor_model.training for actor_model in trainer.actor_models)
    assert all(
        not parameter.requires_grad
        for actor_model in trainer.actor_models
        for parameter in actor_model.parameters()
    )
    online_parameter = next(trainer.models["online"].parameters())
    assert all(
        next(actor_model.parameters()).data_ptr() != online_parameter.data_ptr()
        for actor_model in trainer.actor_models
    )
    assert all(
        next(actor_model.parameters()).device == online_parameter.device
        for actor_model in trainer.actor_models
    )
    for handle in handles:
        handle.remove()
    trainer.close()


def test_dqn_synchronizes_actor_models_after_optimizer_steps(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(
        _config(
            tmp_path,
            actor_model_copies=2,
            actor_model_sync_frequency=2,
        )
    )
    trainer.setup()
    trainer.select_action(0, deterministic=False)
    with torch.no_grad():
        next(trainer.models["online"].parameters()).add_(1.0)

    trainer.global_step = 1
    first_metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )

    assert first_metrics is not None
    assert first_metrics["dqn/actor_model_copies"] == 2.0
    assert first_metrics["dqn/actor_policy_version"] == 1.0
    assert first_metrics["dqn/actor_policy_lag"] == 1.0
    assert not torch.equal(
        next(trainer.actor_models[0].parameters()),
        next(trainer.models["online"].parameters()),
    )

    trainer.global_step = 2
    second_metrics = trainer.process_transition(
        Transition(
            observation=1,
            action=1,
            reward=1.0,
            next_observation=2,
            terminated=False,
            truncated=False,
        )
    )

    assert second_metrics is not None
    assert second_metrics["dqn/actor_policy_version"] == 2.0
    assert second_metrics["dqn/actor_policy_lag"] == 0.0
    for actor_model in trainer.actor_models:
        for actor_parameter, online_parameter in zip(
            actor_model.parameters(),
            trainer.models["online"].parameters(),
            strict=True,
        ):
            assert torch.equal(actor_parameter, online_parameter)
    trainer.close()


def test_dqn_timing_sums_gradient_steps_without_double_counting_actor_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(
        _config(
            tmp_path,
            gradient_steps=2,
            actor_model_copies=1,
            actor_model_sync_frequency=1,
        )
    )
    trainer.setup()
    trainer.select_action(0, deterministic=False)
    clock = iter(float(value) for value in range(1, 17))
    monkeypatch.setattr(
        "dl_core.trainers.dqn_trainer.perf_counter",
        lambda: next(clock),
    )
    trainer.global_step = 1

    metrics = trainer.process_transition(
        Transition(
            observation=0,
            action=1,
            reward=1.0,
            next_observation=1,
            terminated=False,
            truncated=False,
        )
    )

    assert metrics is not None
    assert metrics["dqn/timing/transition_validation_ms"] == 1000.0
    assert metrics["dqn/timing/replay_add_ms"] == 1000.0
    assert metrics["dqn/timing/replay_sample_ms"] == 2000.0
    assert metrics["dqn/timing/model_update_ms"] == 2000.0
    assert metrics["dqn/timing/actor_sync_ms"] == 2000.0
    trainer.close()


def test_dqn_preserves_crossed_vector_update_steps_with_n_step_replay(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, n_step=2, train_frequency=1))
    trainer.setup()
    first_transitions = TransitionBatch(
        observations=np.asarray([0, 1], dtype=np.int64),
        actions=np.asarray([1, 1], dtype=np.int64),
        rewards=np.ones(2, dtype=np.float32),
        next_observations=np.asarray([1, 2], dtype=np.int64),
        terminated=np.zeros(2, dtype=np.bool_),
        truncated=np.zeros(2, dtype=np.bool_),
    )
    trainer.global_step = 2
    first_logs = trainer.process_transition_batch(first_transitions)
    trainer.global_step = 4
    logs = trainer.process_transition_batch(
        TransitionBatch(
            observations=first_transitions.next_observations,
            actions=np.asarray([1, 1], dtype=np.int64),
            rewards=np.ones(2, dtype=np.float32),
            next_observations=np.asarray([2, 3], dtype=np.int64),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
        )
    )

    assert first_logs == []
    assert len(logs) == 2
    assert len(trainer.replay_buffer) == 2
    trainer.close()


def test_dqn_can_gate_eligible_updates_by_global_step(tmp_path: Path) -> None:
    load_builtin_components()
    trainer = DQNTrainer(_config(tmp_path, train_frequency=1))
    trainer.setup()
    checked_steps: list[int] = []

    def update_every_four_steps(
        global_step: int,
        transitions: TransitionBatch,
    ) -> bool:
        checked_steps.append(global_step)
        return global_step % 4 == 0 and transitions.size == 6

    trainer.should_update = update_every_four_steps
    trainer.global_step = 6
    logs = trainer.process_transition_batch(
        TransitionBatch(
            observations=np.arange(6, dtype=np.int64) % 4,
            actions=np.ones(6, dtype=np.int64),
            rewards=np.ones(6, dtype=np.float32),
            next_observations=(np.arange(6, dtype=np.int64) + 1) % 4,
            terminated=np.zeros(6, dtype=np.bool_),
            truncated=np.zeros(6, dtype=np.bool_),
        )
    )

    assert checked_steps == [1, 2, 3, 4, 5, 6]
    assert len(logs) == 1
    trainer.close()


def test_dqn_orders_target_syncs_between_crossed_vector_updates(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    trainer = DQNTrainer(
        _config(
            tmp_path,
            train_frequency=2,
            target_update_frequency=3,
        )
    )
    trainer.setup()
    events: list[str] = []
    optimizer_step = trainer.accelerator.optimizer_step
    synchronize_target = trainer._synchronize_target_network

    def recording_optimizer_step(
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
    ) -> bool:
        events.append("update")
        return optimizer_step(optimizer, model)

    def recording_target_sync() -> None:
        events.append("target")
        synchronize_target()

    trainer.accelerator.optimizer_step = recording_optimizer_step
    trainer._synchronize_target_network = recording_target_sync
    trainer.global_step = 6
    trainer.process_transition_batch(
        TransitionBatch(
            observations=np.arange(6, dtype=np.int64) % 4,
            actions=np.ones(6, dtype=np.int64),
            rewards=np.ones(6, dtype=np.float32),
            next_observations=(np.arange(6, dtype=np.int64) + 1) % 4,
            terminated=np.zeros(6, dtype=np.bool_),
            truncated=np.zeros(6, dtype=np.bool_),
        )
    )

    assert events == ["update", "target", "update", "update", "target"]
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


def test_dqn_checkpoint_recreates_actor_models_from_online_policy(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config = _config(tmp_path, actor_model_copies=2)
    trainer = DQNTrainer(config)
    trainer.setup()
    trainer.select_action(0, deterministic=False)
    trainer.actor_policy_version = 7
    checkpoint_path = trainer.save_checkpoint("dqn-actors.pth")
    assert checkpoint_path is not None

    restored = DQNTrainer(config)
    restored.setup()
    restored.load_checkpoint(str(checkpoint_path))

    assert restored.actor_policy_version == 7
    assert restored.actor_models == []
    restored.select_action(0, deterministic=False)
    assert restored.actor_policy_version == 8
    assert len(restored.actor_models) == 2
    for actor_model in restored.actor_models:
        for actor_parameter, online_parameter in zip(
            actor_model.parameters(),
            restored.models["online"].parameters(),
            strict=True,
        ):
            assert torch.equal(actor_parameter, online_parameter)

    legacy_state = restored.algorithm_state_dict()
    legacy_state.pop("actor_policy_version")
    restored.load_algorithm_state_dict(legacy_state)
    assert restored.actor_policy_version == 0
    assert restored.actor_models == []
    trainer.close()
    restored.close()
