"""Tests for the discrete-action Dreamer trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dl_core import load_builtin_components
from dl_core.core import MODEL_REGISTRY, TRAINER_REGISTRY, TransitionBatch
from dl_core.trainers import (
    DreamerPolicyState,
    DreamerTrainer,
    ImaginedTrajectory,
)
from dreamer_test_models import (  # noqa: F401
    ProjectDreamerActor,
    ProjectDreamerCritic,
    ProjectDreamerWorldModel,
)


def _config(tmp_path: Path, **overrides: object) -> dict:
    trainer_config = {
        "total_timesteps": 10,
        "max_episode_steps": 10,
        "evaluation_episodes": 0,
        "checkpoint_frequency": 0,
        "buffer_size": 8,
        "batch_size": 2,
        "sequence_length": 2,
        "burn_in": 0,
        "learning_starts": 0,
        "train_frequency": 1,
        "gradient_steps": 1,
        "imagination_horizon": 3,
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
        "models": {
            "world_model": {
                "name": "test_dreamer_world_model",
                "embedding_size": 8,
                "deterministic_size": 8,
                "stochastic_size": 2,
                "classes": 3,
                "hidden_size": 16,
            },
            "actor": {
                "name": "test_dreamer_actor",
                "hidden_size": 16,
            },
            "critic": {
                "name": "test_dreamer_critic",
                "hidden_size": 16,
            },
        },
        "optimizers": {
            "world_model": {"name": "adam", "lr": 1e-3},
            "actor": {"name": "adam", "lr": 1e-3},
            "critic": {"name": "adam", "lr": 1e-3},
        },
        "trainer": {"dreamer": trainer_config},
        "accelerator": {"type": "cpu"},
        "runtime": {"output_dir": str(tmp_path / "artifacts")},
        "experiment": {"name": "dreamer-tests", "run_name": "dreamer"},
    }


def test_dreamer_trainer_is_registered_with_public_types() -> None:
    load_builtin_components()

    assert TRAINER_REGISTRY.is_registered("dreamer")
    assert DreamerTrainer is not None
    assert DreamerPolicyState is not None
    assert ImaginedTrajectory is not None


def test_dreamer_accepts_a_structural_project_world_model(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()

    assert isinstance(
        trainer.models["world_model"],
        ProjectDreamerWorldModel,
    )
    assert not MODEL_REGISTRY.is_registered("dreamer_world_model")
    assert not MODEL_REGISTRY.is_registered("dreamer_actor")
    assert not MODEL_REGISTRY.is_registered("dreamer_critic")
    trainer.close()


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            None,
            r"requires models\.world_model\.name, models\.actor\.name, "
            r"and models\.critic\.name",
        ),
        ({}, r"requires models\.world_model\.name"),
        (
            {
                "world_model": {},
                "actor": {"name": "test_dreamer_actor"},
                "critic": {"name": "test_dreamer_critic"},
            },
            r"requires models\.world_model\.name",
        ),
        (
            {
                "world_model": {"name": "test_dreamer_world_model"},
                "actor": {},
                "critic": {"name": "test_dreamer_critic"},
            },
            r"requires models\.actor\.name",
        ),
        (
            {
                "world_model": {"name": "test_dreamer_world_model"},
                "actor": {"name": "test_dreamer_actor"},
                "critic": {},
            },
            r"requires models\.critic\.name",
        ),
    ],
)
def test_dreamer_requires_explicit_project_models(
    tmp_path: Path,
    models: object,
    message: str,
) -> None:
    config = _config(tmp_path)
    if models is None:
        config.pop("models")
    else:
        config["models"] = models
    trainer = DreamerTrainer(config)

    with pytest.raises(ValueError, match=message):
        trainer.setup()
    trainer.close()


def test_dreamer_policy_state_tracks_and_resets_individual_lanes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "vectorization_mode": "sync",
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    config["trainer"]["dreamer"]["buffer_size"] = 16
    trainer = DreamerTrainer(config)
    trainer.setup()
    try:
        policy_state = trainer.initialize_policy_state(
            2,
            evaluation=False,
        )
        output = trainer.select_actions_with_state(
            np.asarray([0, 1]),
            policy_state,
            deterministic=True,
        )

        assert len(output.actions) == 2
        assert isinstance(output.policy_state, DreamerPolicyState)
        assert not output.policy_state.is_first.any()
        reset_state = trainer.reset_policy_state(
            output.policy_state,
            np.asarray([True, False]),
            evaluation=False,
        )
        assert reset_state.is_first.tolist() == [True, False]
        assert torch.count_nonzero(
            reset_state.world_state.deterministic[0]
        ) == 0
        assert torch.equal(
            reset_state.world_state.deterministic[1],
            output.policy_state.world_state.deterministic[1],
        )
        assert torch.count_nonzero(reset_state.previous_actions[0]) == 0
        assert torch.equal(
            reset_state.previous_actions[1],
            output.policy_state.previous_actions[1],
        )
    finally:
        trainer.close()


def test_dreamer_action_distribution_applies_stable_unimix(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path, actor_unimix=0.2))
    trainer.setup()
    try:
        logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])
        distribution = trainer.build_action_distribution(logits)

        assert torch.isfinite(distribution.logits).all()
        assert distribution.probs[0, 1:].tolist() == pytest.approx(
            [0.05, 0.05, 0.05]
        )
        assert distribution.probs[0, 0].item() == pytest.approx(0.85)
    finally:
        trainer.close()


def test_dreamer_public_research_hooks_drive_acting_and_learning(
    tmp_path: Path,
) -> None:
    class HookedDreamerTrainer(DreamerTrainer):
        def __init__(self, config: dict) -> None:
            self.observation_hook_calls = 0
            self.distribution_hook_calls = 0
            super().__init__(config)

        def transform_observations(
            self,
            observations: object,
        ) -> torch.Tensor:
            self.observation_hook_calls += 1
            return super().transform_observations(observations)

        def build_action_distribution(
            self,
            logits: torch.Tensor,
        ) -> torch.distributions.Categorical:
            self.distribution_hook_calls += 1
            return super().build_action_distribution(logits)

    trainer = HookedDreamerTrainer(_config(tmp_path))
    trainer.setup()
    try:
        state = trainer.initialize_policy_state(1, evaluation=False)
        trainer.select_actions_with_state(
            np.asarray([0]),
            state,
            deterministic=False,
        )
        for step in range(2):
            trainer.global_step = step + 1
            trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([step]),
                    actions=np.asarray([1]),
                    rewards=np.asarray([1.0], dtype=np.float32),
                    next_observations=np.asarray([step + 1]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                )
            )

        assert trainer.observation_hook_calls == 2
        assert trainer.distribution_hook_calls == (
            1 + trainer.imagination_horizon
        )
    finally:
        trainer.close()


def test_dreamer_box_observation_transform_preserves_sequence_axes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["environment"] = {
        "name": "gymnasium",
        "id": "CartPole-v1",
    }
    trainer = DreamerTrainer(config)
    trainer.setup()
    try:
        observations = np.zeros((2, 3, 4), dtype=np.float32)

        transformed = trainer.transform_observations(observations)

        assert transformed.shape == (2, 3, 4)
        assert transformed.dtype == torch.float32
    finally:
        trainer.close()


def test_dreamer_lambda_returns_match_manual_recursion(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path, lambda_=0.5))
    trainer.setup()
    try:
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        discounts = torch.tensor([[0.9, 0.9, 0.0]])
        next_values = torch.tensor([[10.0, 20.0, 30.0]])

        returns = trainer.compute_lambda_returns(
            rewards,
            discounts,
            next_values,
        )

        assert torch.allclose(
            returns,
            torch.tensor([[11.0575, 12.35, 3.0]]),
        )
    finally:
        trainer.close()


def test_dreamer_update_changes_all_trainable_models(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()
    try:
        for step in range(2):
            trainer.global_step = step + 1
            logs = trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([step]),
                    actions=np.asarray([1]),
                    rewards=np.asarray([1.0], dtype=np.float32),
                    next_observations=np.asarray([step + 1]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                )
            )
            if step == 0:
                initial_parameters = {
                    name: [
                        parameter.detach().clone()
                        for parameter in trainer.models[name].parameters()
                    ]
                    for name in ("world_model", "actor", "critic")
                }

        expected_metrics = {
            "dreamer/world_model_loss",
            "dreamer/actor_loss",
            "dreamer/critic_loss",
            "dreamer/dynamics_kl",
            "dreamer/imagined_return",
            "dreamer/replay_sequences",
        }
        assert expected_metrics <= logs[0].keys()
        assert all(np.isfinite(value) for value in logs[0].values())
        for name in ("world_model", "actor", "critic"):
            assert any(
                not torch.equal(before, after)
                for before, after in zip(
                    initial_parameters[name],
                    trainer.models[name].parameters(),
                    strict=True,
                )
            )
        assert all(
            parameter.grad is None
            for parameter in trainer.models["world_model"].parameters()
        )
        assert all(
            parameter.grad is None
            for parameter in trainer.models[
                "target_critic"
            ].parameters()
        )
    finally:
        trainer.close()


def test_dreamer_rejects_nonfinite_rewards_before_replay(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()
    try:
        trainer.global_step = 1
        with pytest.raises(ValueError, match="reward must be finite"):
            trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([0]),
                    actions=np.asarray([1]),
                    rewards=np.asarray([np.nan], dtype=np.float32),
                    next_observations=np.asarray([1]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                )
            )

        assert len(trainer.replay_buffer) == 0
    finally:
        trainer.close()


def test_dreamer_rejects_nonfinite_world_loss_before_update(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()
    try:
        for step in range(2):
            trainer.replay_buffer.add_batch(
                TransitionBatch(
                    observations=np.asarray([step]),
                    actions=np.asarray([1]),
                    rewards=np.asarray([1.0], dtype=np.float32),
                    next_observations=np.asarray([step + 1]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                )
            )
        sequences = trainer.replay_buffer.sample(
            1,
            torch.device("cpu"),
        )
        initial_parameters = [
            parameter.detach().clone()
            for parameter in trainer.models["world_model"].parameters()
        ]
        world_model = trainer.models["world_model"]
        with (
            patch.object(
                world_model,
                "predict_rewards",
                side_effect=lambda features: torch.full(
                    features.shape[:-1],
                    float("nan"),
                    device=features.device,
                ),
            ),
            pytest.raises(
                FloatingPointError,
                match="world-model loss must be finite",
            ),
        ):
            trainer.update_model(sequences)

        assert all(
            torch.equal(before, after)
            for before, after in zip(
                initial_parameters,
                trainer.models["world_model"].parameters(),
                strict=True,
            )
        )
    finally:
        trainer.close()


def test_dreamer_aggregates_gradient_steps_per_scheduled_update(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path, gradient_steps=3))
    trainer.setup()
    try:
        trainer.global_step = 1
        trainer.process_transition_batch(
            TransitionBatch(
                observations=np.asarray([0]),
                actions=np.asarray([1]),
                rewards=np.asarray([1.0], dtype=np.float32),
                next_observations=np.asarray([1]),
                terminated=np.asarray([False]),
                truncated=np.asarray([False]),
            )
        )
        trainer.global_step = 2
        with patch.object(
            trainer,
            "update_model",
            return_value={"dreamer/test_loss": 2.0},
        ) as update_model:
            logs = trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([1]),
                    actions=np.asarray([1]),
                    rewards=np.asarray([1.0], dtype=np.float32),
                    next_observations=np.asarray([2]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                )
            )

        assert update_model.call_count == 3
        assert len(logs) == 1
        assert logs[0]["dreamer/test_loss"] == pytest.approx(2.0)
    finally:
        trainer.close()


def test_dreamer_first_vector_sequences_keep_all_ready_boundaries(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 4,
        "vectorization_mode": "sync",
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    config["trainer"]["dreamer"]["buffer_size"] = 32
    trainer = DreamerTrainer(config)
    trainer.setup()
    try:
        trainer.global_step = 4
        trainer.process_transition_batch(
            TransitionBatch(
                observations=np.asarray([0, 0, 0, 0]),
                actions=np.asarray([1, 1, 1, 1]),
                rewards=np.ones(4, dtype=np.float32),
                next_observations=np.asarray([1, 1, 1, 1]),
                terminated=np.zeros(4, dtype=np.bool_),
                truncated=np.zeros(4, dtype=np.bool_),
            )
        )
        trainer.global_step = 8
        with patch.object(
            trainer,
            "update_model",
            return_value={"dreamer/test_loss": 1.0},
        ) as update_model:
            logs = trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([1, 1, 1, 1]),
                    actions=np.asarray([1, 1, 1, 1]),
                    rewards=np.ones(4, dtype=np.float32),
                    next_observations=np.asarray([2, 2, 2, 2]),
                    terminated=np.zeros(4, dtype=np.bool_),
                    truncated=np.zeros(4, dtype=np.bool_),
                )
            )

        assert update_model.call_count == 4
        assert len(logs) == 4
    finally:
        trainer.close()


def test_dreamer_vector_schedule_waits_for_replacement_window(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["environment"] = {
        "name": "gymnasium_vector",
        "id": "FrozenLake-v1",
        "num_envs": 2,
        "vectorization_mode": "sync",
        "kwargs": {"is_slippery": False},
    }
    config["evaluation_environment"] = {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
    config["trainer"]["dreamer"]["buffer_size"] = 4
    trainer = DreamerTrainer(config)
    trainer.setup()
    try:
        trainer.global_step = 2
        trainer.process_transition_batch(
            TransitionBatch(
                observations=np.asarray([0, 0]),
                actions=np.asarray([1, 1]),
                rewards=np.zeros(2, dtype=np.float32),
                next_observations=np.asarray([1, 0]),
                terminated=np.asarray([False, True]),
                truncated=np.zeros(2, dtype=np.bool_),
            )
        )
        trainer.global_step = 4
        with patch.object(
            trainer,
            "update_model",
            return_value={"dreamer/test_loss": 1.0},
        ):
            trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([1, 0]),
                    actions=np.asarray([1, 1]),
                    rewards=np.zeros(2, dtype=np.float32),
                    next_observations=np.asarray([2, 1]),
                    terminated=np.asarray([True, False]),
                    truncated=np.zeros(2, dtype=np.bool_),
                )
            )

        trainer.global_step = 6
        with patch.object(
            trainer,
            "update_model",
            return_value={"dreamer/test_loss": 1.0},
        ) as update_model:
            logs = trainer.process_transition_batch(
                TransitionBatch(
                    observations=np.asarray([0, 1]),
                    actions=np.asarray([1, 1]),
                    rewards=np.zeros(2, dtype=np.float32),
                    next_observations=np.asarray([1, 2]),
                    terminated=np.zeros(2, dtype=np.bool_),
                    truncated=np.zeros(2, dtype=np.bool_),
                )
            )

        assert update_model.call_count == 1
        assert len(logs) == 1
    finally:
        trainer.close()


def test_dreamer_checkpoint_restores_sequence_replay(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()
    trainer.global_step = 2
    trainer.replay_buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([0]),
            actions=np.asarray([1]),
            rewards=np.asarray([1.0], dtype=np.float32),
            next_observations=np.asarray([1]),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
        )
    )
    trainer.replay_buffer.add_batch(
        TransitionBatch(
            observations=np.asarray([1]),
            actions=np.asarray([2]),
            rewards=np.asarray([2.0], dtype=np.float32),
            next_observations=np.asarray([2]),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
        )
    )
    checkpoint_path = trainer.save_checkpoint("dreamer.pth")
    trainer.close()

    restored = DreamerTrainer(_config(tmp_path))
    restored.setup()
    try:
        assert checkpoint_path is not None
        restored.load_checkpoint(str(checkpoint_path))

        assert restored.global_step == 2
        assert len(restored.replay_buffer) == 2
        assert restored.replay_buffer.num_sequences == 1
        original_sample = trainer.replay_buffer.sample(
            1,
            torch.device("cpu"),
        )
        restored_sample = restored.replay_buffer.sample(
            1,
            torch.device("cpu"),
        )
        assert torch.equal(
            original_sample.observations,
            restored_sample.observations,
        )
        assert torch.equal(
            original_sample.actions,
            restored_sample.actions,
        )
    finally:
        restored.close()


def test_dreamer_checkpoint_rejects_changed_training_parameters(
    tmp_path: Path,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path))
    trainer.setup()
    checkpoint_path = trainer.save_checkpoint("dreamer-contract.pth")
    trainer.close()

    restored = DreamerTrainer(_config(tmp_path, gamma=0.5))
    restored.setup()
    try:
        assert checkpoint_path is not None
        with pytest.raises(
            ValueError,
            match="training parameters do not match",
        ):
            restored.load_checkpoint(str(checkpoint_path))
    finally:
        restored.close()


@pytest.mark.parametrize(
    "parameter",
    [
        "free_nats",
        "dynamics_kl_weight",
        "representation_kl_weight",
        "reconstruction_weight",
        "reward_weight",
        "continuation_weight",
        "actor_entropy_weight",
    ],
)
def test_dreamer_rejects_nonfinite_loss_parameters(
    tmp_path: Path,
    parameter: str,
) -> None:
    trainer = DreamerTrainer(_config(tmp_path, **{parameter: np.nan}))

    with pytest.raises(
        ValueError,
        match="finite and nonnegative",
    ):
        trainer.setup()
    trainer.close()
