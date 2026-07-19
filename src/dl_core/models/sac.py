"""Built-in policy and twin-Q networks for soft actor-critic."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dl_core.core import config_field, register_model


@register_model("sac_gaussian_actor")
class SACGaussianActor(nn.Module):
    """MLP producing state-dependent Gaussian policy parameters."""

    CONFIG_FIELDS = [
        config_field("input_dim", "int", "Flattened observation dimension.", required=True),
        config_field("action_dim", "int", "Number of action dimensions.", required=True),
        config_field(
            "hidden_sizes",
            "list[int]",
            "Policy hidden-layer widths.",
            default=[256, 256],
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        hidden_sizes = config.get("hidden_sizes", [256, 256])
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
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(previous_size, action_dim)
        self.log_std_head = nn.Linear(previous_size, action_dim)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the mean and log standard deviation of the raw Gaussian."""
        return self._forward(observations)

    def _forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(observations.reshape(observations.shape[0], -1))
        return {
            "mean": self.mean_head(features),
            "log_std": self.log_std_head(features),
        }


@register_model("sac_twin_q_network")
class SACTwinQNetwork(nn.Module):
    """Two independent MLP critics over observation-action pairs."""

    CONFIG_FIELDS = [
        config_field("input_dim", "int", "Flattened observation dimension.", required=True),
        config_field("action_dim", "int", "Number of action dimensions.", required=True),
        config_field(
            "hidden_sizes",
            "list[int]",
            "Critic hidden-layer widths.",
            default=[256, 256],
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        hidden_sizes = config.get("hidden_sizes", [256, 256])
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        if not isinstance(hidden_sizes, list) or not all(
            isinstance(size, int) and size > 0 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a list of positive integers")

        critics: list[nn.Module] = []
        for _ in range(2):
            layers: list[nn.Module] = []
            previous_size = input_dim + action_dim
            for hidden_size in hidden_sizes:
                layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, 1))
            critics.append(nn.Sequential(*layers))
        self.critics = nn.ModuleList(critics)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return one scalar estimate from each independent critic."""
        return self._forward(observations, actions)

    def _forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = torch.cat(
            (
                observations.reshape(observations.shape[0], -1),
                actions.reshape(actions.shape[0], -1),
            ),
            dim=1,
        )
        return {
            "q1": self.critics[0](inputs).squeeze(1),
            "q2": self.critics[1](inputs).squeeze(1),
        }
