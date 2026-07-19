"""Built-in multilayer perceptron for discrete-action value learning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dl_core.core import config_field, register_model


@register_model("dqn_mlp")
class DQNMLP(nn.Module):
    """MLP mapping flattened observations to one Q-value per action."""

    CONFIG_FIELDS = [
        config_field("input_dim", "int", "Flattened observation dimension.", required=True),
        config_field("action_dim", "int", "Number of discrete actions.", required=True),
        config_field(
            "hidden_sizes",
            "list[int]",
            "Hidden-layer widths; an empty list creates a linear Q-network.",
            default=[128, 128],
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        hidden_sizes = config.get("hidden_sizes", [128, 128])
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        if not isinstance(hidden_sizes, list) or not all(
            isinstance(size, int) and size > 0 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a list of positive integers")

        layers: list[nn.Module] = []
        previous_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value vector per flattened observation."""
        return self._forward(observations)

    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
