"""Episode accumulation and artifact contracts for reinforcement learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from dl_core.utils import ArtifactManager

from .config_metadata import config_field
from .rl_types import EpisodeContext, EpisodeRecord, EpisodeResult, Transition


@dataclass(slots=True)
class _EpisodeAccumulator:
    context: EpisodeContext
    capture: bool
    episode_return: float = 0.0
    reward_square_sum: float = 0.0
    reward_min: float = float("inf")
    reward_max: float = float("-inf")
    length: int = 0
    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    action_info: list[dict[str, Any]] = field(default_factory=list)


class BaseEpisodeManager(ABC):
    """Accumulate episode summaries and selected complete trajectories."""

    CONFIG_FIELDS = [
        config_field(
            "capture_phases",
            "list[str]",
            "Phases whose complete trajectories should be persisted.",
            default=["evaluation"],
        ),
        config_field(
            "capture_every_n_episodes",
            "int",
            "Capture every Nth eligible episode.",
            default=1,
        ),
        config_field(
            "max_captured_episodes",
            "int",
            "Maximum captured trajectories; zero is unlimited.",
            default=0,
        ),
        config_field(
            "info_keys",
            "list[str]",
            "Info keys retained in complete trajectory artifacts.",
            default=[],
        ),
        config_field(
            "capture_action_info",
            "bool",
            "Retain algorithm metadata such as values or log probabilities.",
            default=False,
        ),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        artifact_manager: ArtifactManager | None = None,
        trainer: Any = None,
    ):
        self.config = dict(config or {})
        self.artifact_manager = artifact_manager
        self.trainer = trainer
        self.name = self.__class__.__name__
        phases = self.config.get("capture_phases", ["evaluation"])
        if not isinstance(phases, list) or not all(
            isinstance(phase, str) for phase in phases
        ):
            raise TypeError("capture_phases must be a list of strings")
        self.capture_phases = set(phases)
        self.capture_every_n_episodes = int(
            self.config.get("capture_every_n_episodes", 1)
        )
        self.max_captured_episodes = int(
            self.config.get("max_captured_episodes", 0)
        )
        info_keys = self.config.get("info_keys", [])
        if not isinstance(info_keys, list) or not all(
            isinstance(key, str) for key in info_keys
        ):
            raise TypeError("info_keys must be a list of strings")
        self.info_keys = set(info_keys)
        self.capture_action_info = bool(
            self.config.get("capture_action_info", False)
        )
        if self.capture_every_n_episodes <= 0:
            raise ValueError("capture_every_n_episodes must be positive")
        if self.max_captured_episodes < 0:
            raise ValueError("max_captured_episodes cannot be negative")
        self._active: dict[tuple[str, int], _EpisodeAccumulator] = {}
        self._captured_episodes = 0
        self._capture_reservations = 0

    def set_name(self, name: str) -> None:
        """Set the configured component name used for artifact streams."""
        self._set_name(name)

    def _set_name(self, name: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError(
                "Episode manager name must contain only letters, numbers, "
                "'.', '_', or '-'"
            )
        self.name = name

    def begin_episode(self, context: EpisodeContext) -> None:
        """Begin tracking one environment lane."""
        self._begin_episode(context)

    def _begin_episode(self, context: EpisodeContext) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", context.episode_id) is None:
            raise ValueError(
                "episode_id must contain only letters, numbers, '.', '_', or '-'"
            )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", context.phase) is None:
            raise ValueError(
                "phase must contain only letters, numbers, '.', '_', or '-'"
            )
        key = (context.phase, context.environment_index)
        if key in self._active:
            raise RuntimeError(
                f"Environment {context.environment_index} already has an active "
                f"episode for phase {context.phase}"
            )
        capture = (
            context.phase in self.capture_phases
            and context.episode % self.capture_every_n_episodes == 0
            and (
                self.max_captured_episodes == 0
                or self._captured_episodes + self._capture_reservations
                < self.max_captured_episodes
            )
        )
        accumulator = _EpisodeAccumulator(context=context, capture=capture)
        if capture:
            accumulator.observations.append(deepcopy(context.initial_observation))
            self._capture_reservations += 1
        self._active[key] = accumulator

    def record_transition(
        self,
        environment_index: int,
        transition: Transition[Any, Any],
        *,
        phase: str | None = None,
    ) -> None:
        """Record one transition for an active environment lane."""
        self._record_transition(environment_index, transition, phase=phase)

    def _record_transition(
        self,
        environment_index: int,
        transition: Transition[Any, Any],
        *,
        phase: str | None = None,
    ) -> None:
        key = self._resolve_active_key(environment_index, phase)
        accumulator = self._active[key]
        reward = float(transition.reward)
        accumulator.episode_return += reward
        accumulator.reward_square_sum += reward * reward
        accumulator.reward_min = min(accumulator.reward_min, reward)
        accumulator.reward_max = max(accumulator.reward_max, reward)
        accumulator.length += 1
        if not accumulator.capture:
            return
        accumulator.observations.append(deepcopy(transition.next_observation))
        accumulator.actions.append(deepcopy(transition.action))
        accumulator.rewards.append(reward)
        accumulator.terminated.append(bool(transition.terminated))
        accumulator.truncated.append(bool(transition.truncated))
        accumulator.infos.append(
            {
                key: deepcopy(transition.info[key])
                for key in self.info_keys
                if key in transition.info
            }
        )
        if self.capture_action_info:
            accumulator.action_info.append(deepcopy(transition.action_info))

    def end_episode(
        self,
        environment_index: int,
        result: EpisodeResult,
        *,
        phase: str | None = None,
    ) -> EpisodeRecord:
        """Finalize an active episode and persist configured artifacts."""
        return self._end_episode(environment_index, result, phase=phase)

    def _end_episode(
        self,
        environment_index: int,
        result: EpisodeResult,
        *,
        phase: str | None = None,
    ) -> EpisodeRecord:
        key = self._resolve_active_key(environment_index, phase)
        accumulator = self._active.pop(key)
        result.episode_id = accumulator.context.episode_id
        result.environment_index = accumulator.context.environment_index
        if result.seed is None:
            result.seed = accumulator.context.seed
        record = EpisodeRecord(
            context=accumulator.context,
            observations=accumulator.observations,
            actions=accumulator.actions,
            rewards=accumulator.rewards,
            terminated=accumulator.terminated,
            truncated=accumulator.truncated,
            infos=accumulator.infos,
            action_info=accumulator.action_info,
        )
        record.metrics = self.summarize_episode(
            record,
            result,
            episode_return=accumulator.episode_return,
            reward_square_sum=accumulator.reward_square_sum,
            reward_min=accumulator.reward_min,
            reward_max=accumulator.reward_max,
            length=accumulator.length,
        )
        if self.artifact_manager is not None:
            summary_filename = (
                "episodes.jsonl"
                if self.name == "standard"
                else f"episodes_{self.name}.jsonl"
            )
            summary_path = self.artifact_manager.append_final_jsonl(
                f"metrics/{summary_filename}",
                {
                    "episode_id": accumulator.context.episode_id,
                    "episode": accumulator.context.episode,
                    "environment_index": environment_index,
                    "phase": accumulator.context.phase,
                    "seed": accumulator.context.seed,
                    **record.metrics,
                },
            )
            record.artifact_paths["summary_stream"] = str(summary_path)
            if accumulator.capture:
                episode_dir = (
                    self.artifact_manager.get_final_dir()
                    / "episodes"
                    / record.context.phase
                )
                episode_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = (
                    episode_dir / f"{record.context.episode_id}.npz"
                )
                arrays: dict[str, np.ndarray] = {
                    "rewards": np.asarray(
                        record.rewards,
                        dtype=np.float32,
                    ),
                    "terminated": np.asarray(
                        record.terminated,
                        dtype=np.bool_,
                    ),
                    "truncated": np.asarray(
                        record.truncated,
                        dtype=np.bool_,
                    ),
                }
                self._flatten_sequence(
                    "observations",
                    record.observations,
                    arrays,
                )
                self._flatten_sequence("actions", record.actions, arrays)
                metadata = {
                    "episode_id": record.context.episode_id,
                    "episode": record.context.episode,
                    "environment_index": record.context.environment_index,
                    "phase": record.context.phase,
                    "seed": record.context.seed,
                    "start_global_step": record.context.start_global_step,
                    "environment_name": record.context.environment_name,
                    "scenario_fingerprint": (
                        record.context.scenario_fingerprint
                    ),
                    "reset_info": self._json_safe(
                        record.context.reset_info
                    ),
                    "infos": self._json_safe(record.infos),
                    "action_info": self._json_safe(record.action_info),
                    "final_info": self._json_safe(result.final_info),
                    "metrics": record.metrics,
                }
                arrays["metadata_json"] = np.asarray(
                    json.dumps(metadata, sort_keys=True)
                )
                np.savez_compressed(trajectory_path, **arrays)
                index_path = self.artifact_manager.append_final_jsonl(
                    "episodes/index.jsonl",
                    {
                        "episode_id": record.context.episode_id,
                        "phase": record.context.phase,
                        "trajectory": str(trajectory_path),
                        "length": record.length,
                        **record.metrics,
                    },
                )
                record.artifact_paths.update(
                    {
                        "trajectory": str(trajectory_path),
                        "trajectory_index": str(index_path),
                    }
                )
        if accumulator.capture:
            self._capture_reservations -= 1
            self._captured_episodes += 1
        result.metrics.update(record.metrics)
        result.artifact_paths.update(record.artifact_paths)
        return record

    @abstractmethod
    def summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float | int,
    ) -> dict[str, float]:
        """Compute the manager's scalar summary for one completed episode."""

    def abort_episode(
        self,
        environment_index: int,
        *,
        phase: str | None = None,
    ) -> None:
        """Discard one incomplete active episode."""
        self._abort_episode(environment_index, phase=phase)

    def _abort_episode(
        self,
        environment_index: int,
        *,
        phase: str | None = None,
    ) -> None:
        try:
            key = self._resolve_active_key(environment_index, phase)
        except RuntimeError:
            return
        accumulator = self._active.pop(key)
        if accumulator is not None and accumulator.capture:
            self._capture_reservations -= 1

    def _resolve_active_key(
        self,
        environment_index: int,
        phase: str | None,
    ) -> tuple[str, int]:
        if phase is not None:
            key = (phase, environment_index)
            if key not in self._active:
                raise RuntimeError(
                    f"Environment {environment_index} has no active {phase} episode"
                )
            return key
        matches = [
            key for key in self._active if key[1] == environment_index
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Environment {environment_index} does not identify exactly one "
                "active episode; provide phase"
            )
        return matches[0]

    def state_dict(self) -> dict[str, Any]:
        """Return persistent manager state."""
        return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        return {"captured_episodes": self._captured_episodes}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore persistent manager state."""
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        captured_episodes = int(state.get("captured_episodes", 0))
        if captured_episodes < 0:
            raise ValueError("captured_episodes cannot be negative")
        self._captured_episodes = captured_episodes

    def _flatten_sequence(
        self,
        prefix: str,
        values: list[Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        if not values:
            arrays[prefix] = np.asarray([])
            return
        first = values[0]
        if isinstance(first, dict):
            for key in sorted(first):
                self._flatten_sequence(
                    f"{prefix}.{key}",
                    [value[key] for value in values],
                    arrays,
                )
            return
        if isinstance(first, tuple):
            for index in range(len(first)):
                self._flatten_sequence(
                    f"{prefix}.{index}",
                    [value[index] for value in values],
                    arrays,
                )
            return
        try:
            array = np.stack([np.asarray(value) for value in values])
        except ValueError as error:
            raise ValueError(
                f"Cannot persist heterogeneous trajectory field '{prefix}'"
            ) from error
        if array.dtype.hasobject:
            raise ValueError(
                f"Cannot persist object-valued trajectory field '{prefix}' "
                "without pickle"
            )
        arrays[prefix] = array

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)
