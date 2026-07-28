"""Tests for compact DreamerV3-inspired model components."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from dl_core import load_builtin_components
from dl_core.core import MODEL_REGISTRY
from dl_core.models import (
    DreamerActor,
    DreamerCritic,
    DreamerWorldModel,
    WorldModelState,
)


def _world_model() -> DreamerWorldModel:
    return DreamerWorldModel(
        {
            "input_dim": 4,
            "action_dim": 3,
            "embedding_size": 8,
            "deterministic_size": 12,
            "stochastic_size": 3,
            "classes": 4,
            "hidden_size": 16,
            "unimix": 0.01,
        }
    )


def test_dreamer_world_model_observes_vector_sequences() -> None:
    torch.manual_seed(3)
    model = _world_model()
    observations = torch.randn(2, 6, 2, 2)
    actions = torch.randint(0, 3, (2, 5))
    is_first = torch.zeros(2, 6, dtype=torch.bool)
    is_first[:, 0] = True

    output = model(
        observations,
        actions,
        is_first,
        deterministic=False,
    )

    assert output.states.deterministic.shape == (2, 6, 12)
    assert output.states.stochastic.shape == (2, 6, 3, 4)
    assert output.states.logits.shape == (2, 6, 3, 4)
    assert output.prior_logits.shape == (2, 6, 3, 4)
    assert output.observation_targets.shape == (2, 6, 4)
    assert output.reconstructions.shape == (2, 6, 4)
    assert output.reward_predictions.shape == (2, 5)
    assert output.continue_logits.shape == (2, 5)
    assert torch.allclose(
        output.states.stochastic.sum(dim=-1),
        torch.ones(2, 6, 3),
    )
    assert torch.allclose(
        output.states.logits.exp().sum(dim=-1),
        torch.ones(2, 6, 3),
    )
    assert torch.allclose(
        output.prior_logits.exp().sum(dim=-1),
        torch.ones(2, 6, 3),
    )
    assert torch.equal(
        output.observation_targets,
        model.transform_observations(observations.reshape(2, 6, 4)),
    )


def test_dreamer_world_model_flattens_singleton_observation_axes() -> None:
    model = _world_model()
    output = model(
        torch.randn(2, 5, 1, 4),
        torch.randint(0, 3, (2, 4)),
        torch.zeros(2, 5, dtype=torch.bool),
        deterministic=True,
    )

    assert output.observation_targets.shape == (2, 5, 4)
    assert output.reconstructions.shape == (2, 5, 4)


def test_dreamer_world_model_aligns_actions_with_next_observations() -> None:
    model = _world_model()
    observations = torch.randn(2, 4, 4)
    actions = torch.tensor([[0, 1, 2], [2, 1, 0]])
    is_first = torch.zeros(2, 4, dtype=torch.bool)
    is_first[:, 0] = True
    expected_actions = torch.nn.functional.one_hot(actions, 3).float()

    with patch.object(
        model,
        "observe_step",
        wraps=model.observe_step,
    ) as observe_step:
        output = model(
            observations,
            actions,
            is_first,
            deterministic=True,
        )

    assert len(observe_step.call_args_list) == 4
    assert torch.equal(
        observe_step.call_args_list[0].args[1],
        torch.zeros(2, 3),
    )
    for step in range(1, 4):
        assert torch.equal(
            observe_step.call_args_list[step].args[1],
            expected_actions[:, step - 1],
        )
    assert output.reward_predictions.shape[1] == actions.shape[1]
    assert output.continue_logits.shape[1] == actions.shape[1]


def test_dreamer_world_model_resets_only_selected_lanes() -> None:
    torch.manual_seed(5)
    model = _world_model()
    first_state = model.initial_state(2, device=torch.device("cpu"))
    second_state = WorldModelState(
        deterministic=first_state.deterministic.clone(),
        stochastic=first_state.stochastic.clone(),
        logits=first_state.logits.clone(),
    )
    first_state.deterministic[0] = 100.0
    first_state.stochastic[0, :, -1] = 1.0
    second_state.deterministic[0] = -100.0
    second_state.stochastic[0, :, 1] = 1.0
    first_actions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    second_actions = first_actions.clone()
    second_actions[0] = torch.tensor([0.0, 0.0, 1.0])
    embeddings = torch.randn(2, 8)
    is_first = torch.tensor([True, False])

    first_output = model.observe_step(
        first_state,
        first_actions,
        embeddings,
        is_first,
        deterministic=True,
    )
    second_output = model.observe_step(
        second_state,
        second_actions,
        embeddings,
        is_first,
        deterministic=True,
    )

    assert torch.allclose(
        first_output.state.deterministic[0],
        second_output.state.deterministic[0],
    )
    assert torch.allclose(
        first_output.state.stochastic[0],
        second_output.state.stochastic[0],
    )


def test_dreamer_world_model_rejects_broadcast_reset_masks() -> None:
    model = _world_model()
    state = model.initial_state(2, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="mask shape"):
        model.observe_step(
            state,
            torch.zeros(2, 3),
            torch.zeros(2, 8),
            torch.tensor(True),
        )


def test_dreamer_zero_unimix_keeps_extreme_logits_finite() -> None:
    model = DreamerWorldModel(
        {
            "input_dim": 4,
            "action_dim": 3,
            "embedding_size": 8,
            "deterministic_size": 12,
            "stochastic_size": 1,
            "classes": 4,
            "hidden_size": 16,
            "unimix": 0.0,
        }
    )
    with torch.no_grad():
        model.prior[-1].weight.zero_()
        model.prior[-1].bias.copy_(
            torch.tensor([1000.0, -1000.0, 500.0, -500.0])
        )
    state = model.initial_state(2, device=torch.device("cpu"))
    imagined = model.imagine_step(
        state,
        torch.zeros(2, 3),
        deterministic=True,
    )

    assert torch.isfinite(imagined.logits).all()
    assert torch.allclose(
        imagined.logits.exp().sum(dim=-1),
        torch.ones(2, 1),
    )


def test_dreamer_world_model_losses_reach_every_component() -> None:
    torch.manual_seed(7)
    model = _world_model()
    observations = torch.randn(3, 5, 4)
    actions = torch.randint(0, 3, (3, 4))
    is_first = torch.zeros(3, 5, dtype=torch.bool)
    is_first[:, 0] = True
    output = model(observations, actions, is_first)
    posterior = torch.distributions.Categorical(
        logits=output.states.logits
    )
    prior = torch.distributions.Categorical(logits=output.prior_logits)
    loss = (
        output.reconstructions.square().mean()
        + output.reward_predictions.square().mean()
        + output.continue_logits.square().mean()
        + torch.distributions.kl_divergence(posterior, prior).mean()
    )

    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_dreamer_imagination_actor_and_critic_shapes() -> None:
    torch.manual_seed(11)
    world_model = _world_model()
    state = world_model.initial_state(5, device=torch.device("cpu"))
    actions = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 1, 0]),
        3,
    ).float()
    imagined_state = world_model.imagine_step(
        state,
        actions,
        deterministic=True,
    )
    features = world_model.features(imagined_state)
    actor = DreamerActor(
        {
            "feature_dim": world_model.feature_size,
            "action_dim": 3,
            "hidden_size": 16,
        }
    )
    critic = DreamerCritic(
        {
            "feature_dim": world_model.feature_size,
            "hidden_size": 16,
        }
    )

    assert features.shape == (5, world_model.feature_size)
    assert actor(features).shape == (5, 3)
    assert critic(features).shape == (5,)
    assert torch.allclose(
        imagined_state.stochastic.sum(dim=-1),
        torch.ones(5, 3),
    )
    (actor(features).sum() + critic(features).sum()).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())
    assert all(parameter.grad is not None for parameter in critic.parameters())


def test_dreamer_models_are_registered() -> None:
    load_builtin_components()

    world_model = MODEL_REGISTRY.get(
        "dreamer_world_model",
        {
            "input_dim": 4,
            "action_dim": 3,
            "embedding_size": 8,
            "deterministic_size": 12,
            "stochastic_size": 3,
            "classes": 4,
            "hidden_size": 16,
        },
    )
    actor = MODEL_REGISTRY.get(
        "dreamer_actor",
        {
            "feature_dim": world_model.feature_size,
            "action_dim": 3,
            "hidden_size": 16,
        },
    )
    critic = MODEL_REGISTRY.get(
        "dreamer_critic",
        {
            "feature_dim": world_model.feature_size,
            "hidden_size": 16,
        },
    )

    assert isinstance(world_model, DreamerWorldModel)
    assert isinstance(actor, DreamerActor)
    assert isinstance(critic, DreamerCritic)
