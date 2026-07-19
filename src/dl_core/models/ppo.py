"""Built-in actor-critic network for proximal policy optimization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dl_core.core import config_field, register_model


@register_model("ppo_actor_critic")
class PPOActorCritic(nn.Module):
    """Shared MLP encoder with policy and scalar-value heads."""

    CONFIG_FIELDS = [
        config_field("input_dim", "int", "Flattened observation dimension.", required=True),
        config_field("action_dim", "int", "Number of action dimensions.", required=True),
        config_field(
            "continuous_actions",
            "bool",
            "Use a diagonal Gaussian policy instead of categorical logits.",
            required=True,
        ),
        config_field(
            "hidden_sizes",
            "list[int]",
            "Shared encoder widths.",
            default=[64, 64],
        ),
        config_field(
            "initial_log_std",
            "float",
            "Initial continuous-policy log standard deviation.",
            default=0.0,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        hidden_sizes = config.get("hidden_sizes", [64, 64])
        self.continuous_actions = bool(config["continuous_actions"])
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        if not isinstance(hidden_sizes, list) or not all(
            isinstance(size, int) and size > 0 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a list of positive integers")

        layers: list[nn.Module] = []
        previous_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.Tanh()))
            previous_size = hidden_size
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(previous_size, action_dim)
        self.value_head = nn.Linear(previous_size, 1)
        if self.continuous_actions:
            self.log_std = nn.Parameter(
                torch.full(
                    (action_dim,),
                    float(config.get("initial_log_std", 0.0)),
                )
            )
        else:
            self.register_parameter("log_std", None)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return policy parameters and scalar state values."""
        return self._forward(observations)

    def _forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(observations.reshape(observations.shape[0], -1))
        policy_output = self.policy_head(features)
        output = {
            "value": self.value_head(features).squeeze(1),
        }
        if self.continuous_actions:
            output["mean"] = policy_output
            output["log_std"] = self.log_std.expand_as(policy_output)
        else:
            output["logits"] = policy_output
        return output
