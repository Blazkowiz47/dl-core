"""Compact DreamerV3-inspired models for vector observations."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from dl_core.core import (
    WorldModelOutput,
    WorldModelState,
    WorldModelStep,
    config_field,
    register_model,
)


@register_model("dreamer_world_model")
class DreamerWorldModel(nn.Module):
    """Categorical RSSM with compact MLP prediction heads."""

    CONFIG_FIELDS = [
        config_field(
            "input_dim",
            "int",
            "Flattened vector-observation dimension.",
            required=True,
        ),
        config_field(
            "action_dim",
            "int",
            "Number of discrete actions.",
            required=True,
        ),
        config_field(
            "embedding_size",
            "int",
            "Observation encoder output width.",
            default=128,
        ),
        config_field(
            "deterministic_size",
            "int",
            "RSSM recurrent deterministic-state width.",
            default=128,
        ),
        config_field(
            "stochastic_size",
            "int",
            "Number of categorical latent variables.",
            default=16,
        ),
        config_field(
            "classes",
            "int",
            "Classes in each categorical latent variable.",
            default=16,
        ),
        config_field(
            "hidden_size",
            "int",
            "MLP hidden-layer width.",
            default=256,
        ),
        config_field(
            "unimix",
            "float",
            "Uniform mixture added to categorical latent probabilities.",
            default=0.01,
        ),
        config_field(
            "symlog_observations",
            "bool",
            "Encode and reconstruct observations in symlog space.",
            default=True,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.input_dim = int(config["input_dim"])
        self.action_dim = int(config["action_dim"])
        self.embedding_size = int(config.get("embedding_size", 128))
        self.deterministic_size = int(
            config.get("deterministic_size", 128)
        )
        self.stochastic_size = int(config.get("stochastic_size", 16))
        self.classes = int(config.get("classes", 16))
        hidden_size = int(config.get("hidden_size", 256))
        self.unimix = float(config.get("unimix", 0.01))
        self.symlog_observations = bool(
            config.get("symlog_observations", True)
        )
        if min(
            self.input_dim,
            self.action_dim,
            self.embedding_size,
            self.deterministic_size,
            self.stochastic_size,
            self.classes,
            hidden_size,
        ) <= 0:
            raise ValueError("Dreamer model dimensions must be positive")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("Dreamer unimix must be in [0, 1)")

        stochastic_width = self.stochastic_size * self.classes
        self.feature_size = self.deterministic_size + stochastic_width
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.embedding_size),
        )
        self.recurrent = nn.GRUCell(
            stochastic_width + self.action_dim,
            self.deterministic_size,
        )
        self.prior = nn.Sequential(
            nn.Linear(self.deterministic_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, stochastic_width),
        )
        self.posterior = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.embedding_size,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(hidden_size, stochastic_width),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.input_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> WorldModelState:
        """Create zero deterministic and stochastic recurrent state."""
        return self._initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def _initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WorldModelState:
        if batch_size <= 0:
            raise ValueError("Dreamer state batch size must be positive")
        deterministic = torch.zeros(
            batch_size,
            self.deterministic_size,
            device=device,
            dtype=dtype,
        )
        logits = torch.zeros(
            batch_size,
            self.stochastic_size,
            self.classes,
            device=device,
            dtype=dtype,
        )
        stochastic = torch.zeros_like(logits)
        return WorldModelState(
            deterministic=deterministic,
            stochastic=stochastic,
            logits=logits,
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode flattened vector observations."""
        return self._encode(observations)

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.transform_observations(observations))

    def transform_observations(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder-target transform to flattened observations."""
        return self._transform_observations(observations)

    def _transform_observations(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                "Dreamer observation dimension does not match input_dim"
            )
        flattened = observations.float()
        if self.symlog_observations:
            flattened = torch.sign(flattened) * torch.log1p(
                flattened.abs()
            )
        return flattened

    def features(self, state: WorldModelState) -> torch.Tensor:
        """Concatenate deterministic and flattened stochastic state."""
        return self._features(state)

    def _features(self, state: WorldModelState) -> torch.Tensor:
        return torch.cat(
            (
                state.deterministic,
                state.stochastic.flatten(start_dim=-2),
            ),
            dim=-1,
        )

    def predict_rewards(self, features: torch.Tensor) -> torch.Tensor:
        """Predict symlog rewards from RSSM features."""
        return self._predict_rewards(features)

    def _predict_rewards(self, features: torch.Tensor) -> torch.Tensor:
        return self.reward_head(features).squeeze(-1)

    def predict_continue_logits(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict continuation logits from RSSM features."""
        return self._predict_continue_logits(features)

    def _predict_continue_logits(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.continue_head(features).squeeze(-1)

    def _categorical_logits(
        self,
        raw_logits: torch.Tensor,
    ) -> torch.Tensor:
        log_probabilities = functional.log_softmax(raw_logits, dim=-1)
        if self.unimix == 0.0:
            return log_probabilities
        return torch.logaddexp(
            log_probabilities + math.log1p(-self.unimix),
            torch.full_like(
                log_probabilities,
                math.log(self.unimix / self.classes),
            ),
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
        """Infer one posterior state from an observation embedding."""
        return self._observe_step(
            previous_state,
            previous_actions,
            embeddings,
            is_first,
            deterministic=deterministic,
        )

    def _observe_step(
        self,
        previous_state: WorldModelState,
        previous_actions: torch.Tensor,
        embeddings: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool,
    ) -> WorldModelStep:
        if previous_actions.shape != (
            previous_state.deterministic.shape[0],
            self.action_dim,
        ):
            raise ValueError("Dreamer previous-action shape is invalid")
        if embeddings.shape != (
            previous_state.deterministic.shape[0],
            self.embedding_size,
        ):
            raise ValueError("Dreamer embedding shape is invalid")
        if is_first.shape != (
            previous_state.deterministic.shape[0],
        ):
            raise ValueError("Dreamer first-observation mask shape is invalid")
        reset_mask = (~is_first.bool()).to(
            previous_state.deterministic.dtype
        ).unsqueeze(-1)
        deterministic_state = previous_state.deterministic * reset_mask
        stochastic_state = previous_state.stochastic * reset_mask.unsqueeze(-1)
        previous_actions = previous_actions * reset_mask
        deterministic_state = self.recurrent(
            torch.cat(
                (
                    stochastic_state.flatten(start_dim=-2),
                    previous_actions,
                ),
                dim=-1,
            ),
            deterministic_state,
        )
        prior_logits = self._categorical_logits(
            self.prior(deterministic_state).reshape(
                -1,
                self.stochastic_size,
                self.classes,
            )
        )
        posterior_logits = self._categorical_logits(
            self.posterior(
                torch.cat((deterministic_state, embeddings), dim=-1)
            ).reshape(-1, self.stochastic_size, self.classes)
        )
        probabilities = posterior_logits.exp()
        if deterministic:
            sample = functional.one_hot(
                probabilities.argmax(dim=-1),
                self.classes,
            ).to(probabilities.dtype)
        else:
            sample = torch.distributions.OneHotCategorical(
                probs=probabilities
            ).sample()
        stochastic_state = sample + probabilities - probabilities.detach()
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
        """Predict one prior state without an environment observation."""
        return self._imagine_step(
            previous_state,
            actions,
            deterministic=deterministic,
        )

    def _imagine_step(
        self,
        previous_state: WorldModelState,
        actions: torch.Tensor,
        *,
        deterministic: bool,
    ) -> WorldModelState:
        if actions.shape != (
            previous_state.deterministic.shape[0],
            self.action_dim,
        ):
            raise ValueError("Dreamer imagined-action shape is invalid")
        deterministic_state = self.recurrent(
            torch.cat(
                (
                    previous_state.stochastic.flatten(start_dim=-2),
                    actions,
                ),
                dim=-1,
            ),
            previous_state.deterministic,
        )
        logits = self._categorical_logits(
            self.prior(deterministic_state).reshape(
                -1,
                self.stochastic_size,
                self.classes,
            )
        )
        probabilities = logits.exp()
        if deterministic:
            sample = functional.one_hot(
                probabilities.argmax(dim=-1),
                self.classes,
            ).to(probabilities.dtype)
        else:
            sample = torch.distributions.OneHotCategorical(
                probs=probabilities
            ).sample()
        stochastic_state = sample + probabilities - probabilities.detach()
        return WorldModelState(
            deterministic=deterministic_state,
            stochastic=stochastic_state,
            logits=logits,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> WorldModelOutput:
        """Observe a sequence and predict observations, rewards, and continuation."""
        return self._forward(
            observations,
            actions,
            is_first,
            deterministic=deterministic,
        )

    def _forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool,
    ) -> WorldModelOutput:
        if observations.ndim < 3:
            raise ValueError(
                "Dreamer observations must have [batch, time, ...] dimensions"
            )
        batch_size, observation_steps = observations.shape[:2]
        if actions.shape == (batch_size, observation_steps - 1):
            actions = functional.one_hot(
                actions.long(),
                self.action_dim,
            ).float()
        if actions.shape != (
            batch_size,
            observation_steps - 1,
            self.action_dim,
        ):
            raise ValueError("Dreamer action sequence shape is invalid")
        if is_first.shape != (batch_size, observation_steps):
            raise ValueError("Dreamer first-observation mask shape is invalid")

        observation_targets = self.transform_observations(
            observations.reshape(batch_size, observation_steps, -1)
        )
        embeddings = self.encoder(observation_targets)
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
        deterministic_states: list[torch.Tensor] = []
        stochastic_states: list[torch.Tensor] = []
        posterior_logits: list[torch.Tensor] = []
        prior_logits: list[torch.Tensor] = []
        for step in range(observation_steps):
            output = self.observe_step(
                state,
                zero_actions if step == 0 else actions[:, step - 1],
                embeddings[:, step],
                is_first[:, step],
                deterministic=deterministic,
            )
            state = output.state
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
            posterior_logits.append(state.logits)
            prior_logits.append(output.prior_logits)

        states = WorldModelState(
            deterministic=torch.stack(deterministic_states, dim=1),
            stochastic=torch.stack(stochastic_states, dim=1),
            logits=torch.stack(posterior_logits, dim=1),
        )
        features = self.features(states)
        reconstructions = self.decoder(features)
        reward_predictions = self.predict_rewards(features[:, 1:])
        continue_logits = self.predict_continue_logits(features[:, 1:])
        return WorldModelOutput(
            states=states,
            prior_logits=torch.stack(prior_logits, dim=1),
            observation_targets=observation_targets,
            reconstructions=reconstructions,
            reward_predictions=reward_predictions,
            continue_logits=continue_logits,
        )


@register_model("dreamer_actor")
class DreamerActor(nn.Module):
    """Categorical policy over discrete actions from RSSM features."""

    CONFIG_FIELDS = [
        config_field(
            "feature_dim",
            "int",
            "Flattened RSSM feature dimension.",
            required=True,
        ),
        config_field(
            "action_dim",
            "int",
            "Number of discrete actions.",
            required=True,
        ),
        config_field(
            "hidden_size",
            "int",
            "Policy hidden-layer width.",
            default=256,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        feature_dim = int(config["feature_dim"])
        action_dim = int(config["action_dim"])
        hidden_size = int(config.get("hidden_size", 256))
        if min(feature_dim, action_dim, hidden_size) <= 0:
            raise ValueError("Dreamer actor dimensions must be positive")
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return categorical action logits."""
        return self._forward(features)

    def _forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        if logits.shape[-1] != self.action_dim:
            raise ValueError("Dreamer actor action dimension is invalid")
        return logits


@register_model("dreamer_critic")
class DreamerCritic(nn.Module):
    """Scalar value predictor over RSSM features."""

    CONFIG_FIELDS = [
        config_field(
            "feature_dim",
            "int",
            "Flattened RSSM feature dimension.",
            required=True,
        ),
        config_field(
            "hidden_size",
            "int",
            "Value hidden-layer width.",
            default=256,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        feature_dim = int(config["feature_dim"])
        hidden_size = int(config.get("hidden_size", 256))
        if min(feature_dim, hidden_size) <= 0:
            raise ValueError("Dreamer critic dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar value per RSSM feature."""
        return self._forward(features)

    def _forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
