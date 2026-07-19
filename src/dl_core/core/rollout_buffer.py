"""On-policy rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    """Tensor rollout with returns and generalized advantages."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    """Sequential on-policy rollout storage for a single environment stream."""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.actions: list[Any] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probabilities: list[float] = []
        self.next_values: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        value: float,
        log_probability: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one policy transition."""
        self._add(
            observation=observation,
            action=action,
            reward=reward,
            value=value,
            log_probability=log_probability,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

    def _add(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float,
        value: float,
        log_probability: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.observations.append(np.asarray(observation).copy())
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probabilities.append(float(log_probability))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def compute_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        """Compute returns and generalized advantages for the current rollout."""
        return self._compute_batch(gamma=gamma, gae_lambda=gae_lambda, device=device)

    def _compute_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        if not self.rewards:
            raise RuntimeError("Cannot compute an empty rollout")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=np.bool_)
        truncated = np.asarray(self.truncated, dtype=np.bool_)
        advantages = np.empty_like(rewards)
        running_advantage = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            bootstrap_mask = 0.0 if terminated[index] else 1.0
            continuation_mask = 0.0 if terminated[index] or truncated[index] else 1.0
            delta = (
                rewards[index]
                + (gamma * bootstrap_mask * next_values[index])
                - values[index]
            )
            running_advantage = delta + (
                gamma * gae_lambda * continuation_mask * running_advantage
            )
            advantages[index] = running_advantage
        returns = advantages + values
        return RolloutBatch(
            observations=torch.as_tensor(np.asarray(self.observations), device=device),
            actions=torch.as_tensor(np.asarray(self.actions), device=device),
            old_log_probabilities=torch.as_tensor(
                self.log_probabilities,
                dtype=torch.float32,
                device=device,
            ),
            old_values=torch.as_tensor(values, device=device),
            returns=torch.as_tensor(returns, device=device),
            advantages=torch.as_tensor(advantages, device=device),
        )

    def clear(self) -> None:
        """Remove all collected transitions."""
        self._clear()

    def _clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probabilities.clear()
        self.next_values.clear()
        self.terminated.clear()
        self.truncated.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return a copy of a potentially partial rollout."""
        return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        return {
            "observations": [value.copy() for value in self.observations],
            "actions": [value.copy() for value in self.actions],
            "rewards": list(self.rewards),
            "values": list(self.values),
            "log_probabilities": list(self.log_probabilities),
            "next_values": list(self.next_values),
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a partial rollout after validating field lengths."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        field_names = (
            "observations",
            "actions",
            "rewards",
            "values",
            "log_probabilities",
            "next_values",
            "terminated",
            "truncated",
        )
        if not isinstance(state, dict) or any(name not in state for name in field_names):
            raise ValueError("Rollout checkpoint state is incomplete")
        if any(not isinstance(state[name], (list, tuple)) for name in field_names):
            raise ValueError("Rollout checkpoint fields must be sequences")
        lengths = {len(state[name]) for name in field_names}
        if len(lengths) != 1:
            raise ValueError("Rollout checkpoint fields have inconsistent lengths")
        observations = [np.asarray(value).copy() for value in state["observations"]]
        actions = [np.asarray(value).copy() for value in state["actions"]]
        if len({value.shape for value in observations}) > 1:
            raise ValueError("Rollout checkpoint observation shapes are inconsistent")
        if len({value.shape for value in actions}) > 1:
            raise ValueError("Rollout checkpoint action shapes are inconsistent")
        try:
            rewards = [float(value) for value in state["rewards"]]
            values = [float(value) for value in state["values"]]
            log_probabilities = [
                float(value) for value in state["log_probabilities"]
            ]
            next_values = [float(value) for value in state["next_values"]]
        except (TypeError, ValueError) as error:
            raise ValueError("Rollout checkpoint scalar fields are invalid") from error
        if not np.isfinite(
            np.asarray([rewards, values, log_probabilities, next_values])
        ).all():
            raise ValueError("Rollout checkpoint scalar fields must be finite")
        if any(
            not isinstance(value, (bool, np.bool_))
            for name in ("terminated", "truncated")
            for value in state[name]
        ):
            raise ValueError("Rollout checkpoint boundary fields must be booleans")

        self._clear()
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.values.extend(values)
        self.log_probabilities.extend(log_probabilities)
        self.next_values.extend(next_values)
        self.terminated.extend(bool(value) for value in state["terminated"])
        self.truncated.extend(bool(value) for value in state["truncated"])
