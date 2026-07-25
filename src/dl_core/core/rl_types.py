"""Shared reinforcement-learning contracts and value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from gymnasium import Space


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@runtime_checkable
class Environment(Protocol[ObservationT, ActionT]):
    """Structural contract accepted by reinforcement-learning trainers."""

    observation_space: Space[ObservationT]
    action_space: Space[ActionT]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationT, dict[str, Any]]:
        """Reset the environment and return its initial observation."""
        ...

    def step(
        self,
        action: ActionT,
    ) -> tuple[ObservationT, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one action."""
        ...

    def render(self) -> Any:
        """Render one environment frame when supported."""
        ...

    def close(self) -> None:
        """Release resources held by the environment."""
        ...


@dataclass(slots=True)
class Transition(Generic[ObservationT, ActionT]):
    """One environment transition shared by off-policy algorithms."""

    observation: ObservationT
    action: ActionT
    reward: float
    next_observation: ObservationT
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    action_info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Return whether this transition ends the current episode."""
        return self.terminated or self.truncated


@dataclass(slots=True)
class TransitionBatch(Generic[ObservationT, ActionT]):
    """One batched environment transition shared by vector-aware algorithms."""

    observations: ObservationT
    actions: ActionT
    rewards: np.ndarray
    next_observations: ObservationT
    terminated: np.ndarray
    truncated: np.ndarray
    infos: dict[str, Any] = field(default_factory=dict)
    action_info: dict[str, Any] = field(default_factory=dict)
    final_observations: ObservationT | None = None

    @property
    def done(self) -> np.ndarray:
        """Return the per-environment episode-completion mask."""
        return np.logical_or(self.terminated, self.truncated)

    @property
    def size(self) -> int:
        """Return the number of environment transitions in the batch."""
        return int(self.rewards.shape[0])


@dataclass(slots=True)
class EpisodeContext:
    """Identity and reset data for one environment episode."""

    episode_id: str
    episode: int
    environment_index: int
    phase: str
    seed: int | None
    initial_observation: Any
    reset_info: dict[str, Any] = field(default_factory=dict)
    start_global_step: int = 0
    environment_name: str | None = None
    scenario_fingerprint: str | None = None


@dataclass(slots=True)
class EpisodeResult:
    """Summary produced after one training or evaluation episode."""

    episode: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    final_info: dict[str, Any] = field(default_factory=dict)
    episode_id: str | None = None
    environment_index: int = 0
    seed: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    completion_reason: str | None = None


@dataclass(slots=True)
class EpisodeRecord:
    """Complete environment-boundary trajectory for one episode."""

    context: EpisodeContext
    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    action_info: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the number of transitions in the episode."""
        return len(self.actions)

    @property
    def episode_return(self) -> float:
        """Return the undiscounted episode return."""
        return float(sum(self.rewards))


@dataclass(slots=True)
class ActionOutput(Generic[ActionT]):
    """Action selected by a policy with optional algorithm-specific metadata."""

    action: ActionT
    info: dict[str, Any] = field(default_factory=dict)
