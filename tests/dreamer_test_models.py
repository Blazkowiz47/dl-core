"""Test-owned models implementing the public Dreamer trainer contracts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from dl_core.core import (
    WorldModelOutput,
    WorldModelState,
    WorldModelStep,
    register_model,
)


@register_model("test_dreamer_world_model")
class ProjectDreamerWorldModel(nn.Module):
    """Compact categorical state-space model used only by trainer tests."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.input_dim = int(config["input_dim"])
        self.action_dim = int(config["action_dim"])
        self.deterministic_size = int(config.get("deterministic_size", 8))
        self.classes = int(config.get("classes", 3))
        self.feature_size = self.deterministic_size + self.classes
        self.encoder = nn.Linear(self.input_dim, self.deterministic_size)
        self.recurrent = nn.Linear(
            self.feature_size + self.action_dim,
            self.deterministic_size,
        )
        self.prior = nn.Linear(self.deterministic_size, self.classes)
        self.posterior = nn.Linear(
            self.deterministic_size * 2,
            self.classes,
        )
        self.decoder = nn.Linear(self.feature_size, self.input_dim)
        self.reward_head = nn.Linear(self.feature_size, 1)
        self.continue_head = nn.Linear(self.feature_size, 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> WorldModelState:
        """Create a zero latent state."""
        deterministic = torch.zeros(
            batch_size,
            self.deterministic_size,
            device=device,
            dtype=dtype,
        )
        logits = torch.zeros(
            batch_size,
            1,
            self.classes,
            device=device,
            dtype=dtype,
        )
        return WorldModelState(
            deterministic=deterministic,
            stochastic=torch.zeros_like(logits),
            logits=logits,
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode flattened observations."""
        return self.encoder(
            observations.reshape(observations.shape[0], -1).float()
        )

    def features(self, state: WorldModelState) -> torch.Tensor:
        """Concatenate deterministic and categorical state."""
        return torch.cat(
            (state.deterministic, state.stochastic.flatten(start_dim=-2)),
            dim=-1,
        )

    def observe_step(
        self,
        previous_state: WorldModelState,
        previous_actions: torch.Tensor,
        embeddings: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> WorldModelStep:
        """Infer one posterior state."""
        keep = (~is_first.bool()).to(embeddings.dtype).unsqueeze(-1)
        previous_state = WorldModelState(
            deterministic=previous_state.deterministic * keep,
            stochastic=previous_state.stochastic * keep.unsqueeze(-1),
            logits=previous_state.logits * keep.unsqueeze(-1),
        )
        previous_actions = previous_actions * keep
        deterministic_state = torch.tanh(
            self.recurrent(
                torch.cat(
                    (self.features(previous_state), previous_actions),
                    dim=-1,
                )
            )
        )
        prior_logits = self.prior(deterministic_state).unsqueeze(-2)
        posterior_logits = self.posterior(
            torch.cat((deterministic_state, embeddings), dim=-1)
        ).unsqueeze(-2)
        probabilities = functional.softmax(posterior_logits, dim=-1)
        stochastic_state = (
            functional.one_hot(
                probabilities.argmax(dim=-1),
                self.classes,
            ).to(probabilities.dtype)
            if deterministic
            else probabilities
        )
        return WorldModelStep(
            state=WorldModelState(
                deterministic=deterministic_state,
                stochastic=stochastic_state,
                logits=posterior_logits,
            ),
            prior_logits=prior_logits,
        )

    def imagine_step(
        self,
        previous_state: WorldModelState,
        actions: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> WorldModelState:
        """Advance one action-conditioned prior state."""
        deterministic_state = torch.tanh(
            self.recurrent(
                torch.cat((self.features(previous_state), actions), dim=-1)
            )
        )
        logits = self.prior(deterministic_state).unsqueeze(-2)
        probabilities = functional.softmax(logits, dim=-1)
        stochastic_state = (
            functional.one_hot(
                probabilities.argmax(dim=-1),
                self.classes,
            ).to(probabilities.dtype)
            if deterministic
            else probabilities
        )
        return WorldModelState(
            deterministic=deterministic_state,
            stochastic=stochastic_state,
            logits=logits,
        )

    def predict_rewards(self, features: torch.Tensor) -> torch.Tensor:
        """Predict symlog rewards."""
        return self.reward_head(features).squeeze(-1)

    def predict_continue_logits(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict continuation logits."""
        return self.continue_head(features).squeeze(-1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> WorldModelOutput:
        """Observe a full sequence and return trainer-owned loss inputs."""
        batch_size, observation_steps = observations.shape[:2]
        observation_targets = observations.reshape(
            batch_size,
            observation_steps,
            -1,
        ).float()
        embeddings = self.encoder(observation_targets)
        if actions.ndim == 2:
            actions = functional.one_hot(
                actions.long(),
                self.action_dim,
            ).float()
        state = self.initial_state(
            batch_size,
            device=observations.device,
            dtype=embeddings.dtype,
        )
        zero_actions = torch.zeros(
            batch_size,
            self.action_dim,
            device=observations.device,
            dtype=embeddings.dtype,
        )
        states: list[WorldModelState] = []
        prior_logits: list[torch.Tensor] = []
        for step in range(observation_steps):
            world_step = self.observe_step(
                state,
                zero_actions if step == 0 else actions[:, step - 1],
                embeddings[:, step],
                is_first[:, step],
                deterministic=deterministic,
            )
            state = world_step.state
            states.append(state)
            prior_logits.append(world_step.prior_logits)
        stacked_state = WorldModelState(
            deterministic=torch.stack(
                [state.deterministic for state in states],
                dim=1,
            ),
            stochastic=torch.stack(
                [state.stochastic for state in states],
                dim=1,
            ),
            logits=torch.stack(
                [state.logits for state in states],
                dim=1,
            ),
        )
        features = self.features(stacked_state)
        transition_features = features[:, 1:]
        return WorldModelOutput(
            states=stacked_state,
            prior_logits=torch.stack(prior_logits, dim=1),
            observation_targets=observation_targets,
            reconstructions=self.decoder(features),
            reward_predictions=self.predict_rewards(transition_features),
            continue_logits=self.predict_continue_logits(
                transition_features
            ),
        )


@register_model("test_incomplete_dreamer_world_model")
class IncompleteDreamerWorldModel(nn.Module):
    """Registered module intentionally missing the world-model protocol."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        del config
        self.feature_size = 1
        self.parameter = nn.Parameter(torch.zeros(()))


@register_model("test_invalid_dreamer_feature_size")
class InvalidFeatureSizeDreamerWorldModel(ProjectDreamerWorldModel):
    """Structurally valid world model with invalid feature metadata."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.feature_size = 0


@register_model("test_dreamer_actor")
class ProjectDreamerActor(nn.Module):
    """Project-style categorical actor used by trainer tests."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.network = nn.Linear(
            int(config["feature_dim"]),
            int(config["action_dim"]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return categorical action logits."""
        return self.network(features)


@register_model("test_dreamer_critic")
class ProjectDreamerCritic(nn.Module):
    """Project-style scalar critic used by trainer tests."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.network = nn.Linear(int(config["feature_dim"]), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return symlog state values."""
        return self.network(features).squeeze(-1)


__all__ = [
    "IncompleteDreamerWorldModel",
    "InvalidFeatureSizeDreamerWorldModel",
    "ProjectDreamerActor",
    "ProjectDreamerCritic",
    "ProjectDreamerWorldModel",
]
