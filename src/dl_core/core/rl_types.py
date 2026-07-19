"""Shared reinforcement-learning contracts and value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

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

    @property
    def done(self) -> bool:
        """Return whether this transition ends the current episode."""
        return self.terminated or self.truncated


@dataclass(slots=True)
class EpisodeResult:
    """Summary produced after one training or evaluation episode."""

    episode: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    final_info: dict[str, Any] = field(default_factory=dict)
