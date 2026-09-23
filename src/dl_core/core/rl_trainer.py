"""Episode-driven trainer foundation for reinforcement-learning algorithms."""

from __future__ import annotations

import logging
import random
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from dl_core.utils import ArtifactManager, set_seeds
from dl_core.utils.artifact_manager import get_legacy_run_artifact_dir
from dl_core.utils.checkpoint_utils import (
    atomic_torch_save,
    find_latest_checkpoint_local,
)
from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)

from .base_accelerator import BaseAccelerator
from .base_callback import CallbackList
from .base_episode_manager import BaseEpisodeManager
from .batched_environment import BatchedEnvironment
from .config_metadata import config_field
from .registry import (
    ACCELERATOR_REGISTRY,
    CALLBACK_REGISTRY,
    EPISODE_MANAGER_REGISTRY,
)
from .rl_types import (
    ActionOutput,
    BatchActionOutput,
    EpisodeContext,
    EpisodeResult,
    Transition,
    TransitionBatch,
)


class RLTrainer(ABC):
    """Base class for episode-driven reinforcement-learning trainers."""

    REQUIRED_CONFIG_SECTIONS = ("environment",)

    CONFIG_FIELDS = [
        config_field(
            "total_timesteps",
            "int",
            "Maximum training transitions; zero uses the max_episodes budget.",
            default=0,
        ),
        config_field(
            "max_episodes",
            "int | None",
            "Optional maximum number of completed training episodes.",
            default=None,
        ),
        config_field(
            "max_episode_steps",
            "int",
            "Maximum transitions allowed in one episode.",
            default=1000,
        ),
        config_field(
            "evaluation_frequency",
            "int",
            "Evaluate every N completed episodes; zero disables periodic evaluation.",
            default=10,
        ),
        config_field(
            "evaluation_episodes",
            "int",
            "Number of episodes in each evaluation group.",
            default=5,
        ),
        config_field(
            "checkpoint_frequency",
            "int",
            "Save a numbered checkpoint every N completed episodes.",
            default=100,
        ),
        config_field(
            "checkpoint_frequency_steps",
            "int",
            "Save a numbered checkpoint every N training transitions.",
            default=0,
        ),
        config_field(
            "show_progress",
            "bool",
            "Display training progress on the main process.",
            default=False,
        ),
        config_field(
            "continue_model",
            "str | None",
            "RL checkpoint path used to resume a run.",
            default=None,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.seed = int(config.get("seed", 42))
        self.deterministic = bool(config.get("deterministic", True))

        trainer_section = config.get("trainer")
        if not isinstance(trainer_section, dict) or len(trainer_section) != 1:
            raise ValueError("trainer must define exactly one named RL trainer")
        self.trainer_name, trainer_config = next(iter(trainer_section.items()))
        if not isinstance(trainer_config, dict):
            raise TypeError(f"trainer.{self.trainer_name} must be a mapping")
        self.trainer_config = trainer_config

        self.total_timesteps = int(trainer_config.get("total_timesteps", 0))
        max_episodes = trainer_config.get("max_episodes")
        self.max_episodes = int(max_episodes) if max_episodes is not None else None
        self.max_episode_steps = int(trainer_config.get("max_episode_steps", 1000))
        self.evaluation_frequency = int(
            trainer_config.get("evaluation_frequency", 10)
        )
        self.evaluation_episodes = int(trainer_config.get("evaluation_episodes", 5))
        self.checkpoint_frequency = int(
            trainer_config.get("checkpoint_frequency", 100)
        )
        self.checkpoint_frequency_steps = int(
            trainer_config.get("checkpoint_frequency_steps", 0)
        )
        self.show_progress = bool(trainer_config.get("show_progress", False))
        self.continue_model = trainer_config.get("continue_model")
        if self.total_timesteps < 0:
            raise ValueError("total_timesteps cannot be negative")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided")
        if self.total_timesteps <= 0 and self.max_episodes is None:
            raise ValueError("RL training requires total_timesteps or max_episodes")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.evaluation_frequency < 0:
            raise ValueError("evaluation_frequency cannot be negative")
        if self.evaluation_episodes < 0:
            raise ValueError("evaluation_episodes cannot be negative")
        if self.checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency cannot be negative")
        if self.checkpoint_frequency_steps < 0:
            raise ValueError("checkpoint_frequency_steps cannot be negative")

        self.accelerator: BaseAccelerator
        self.environment: BatchedEnvironment
        self.evaluation_environment: BatchedEnvironment
        self.artifact_manager: ArtifactManager
        self.callbacks: CallbackList
        self.episode_managers: dict[str, BaseEpisodeManager] = {}
        self.models: dict[str, nn.Module] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LRScheduler] = {}

        self.current_episode = 0
        self.current_epoch = 0
        self.global_step = 0
        self.collector_step = 0
        self.update_step = 0
        self.stop_training = False
        self._progress_bar: tqdm | None = None
        self.episode_metrics: list[dict[str, Any]] = []
        self.evaluation_metrics: list[dict[str, Any]] = []

        self.setup_artifact_manager()

    def run(self) -> None:
        """Set up, execute, and finalize reinforcement-learning training."""
        self._run()

    def _run(self) -> None:
        status = "completed"
        error_message: str | None = None
        pending_error: BaseException | None = None
        try:
            self.setup()
            self.callbacks.on_training_start(
                {"trainer": self.trainer_name, "total_timesteps": self.total_timesteps}
            )
            self.perform_training()
        except KeyboardInterrupt as error:
            status = "interrupted"
            error_message = str(error) or "Training interrupted by user"
            pending_error = error
            self.logger.warning(error_message)
        except Exception as error:
            status = "failed"
            error_message = str(error)
            pending_error = error
            self.logger.error(f"RL training failed: {error}")
            traceback.print_exc()
        finally:
            final_logs: dict[str, Any] = {
                "status": status,
                "final_episode": self.current_episode,
                "global_step": self.global_step,
                "collector_step": self.collector_step,
                "updates": self.update_step,
            }
            if error_message is not None:
                final_logs["error_message"] = error_message
            try:
                self.artifact_manager.save_final_json(
                    "metrics/history.json",
                    {
                        "episodes": self.episode_metrics,
                        "evaluations": self.evaluation_metrics,
                    },
                )
                self.artifact_manager.save_run_info(final_logs)
            except Exception as artifact_error:
                self.logger.warning(f"Failed to persist RL run summary: {artifact_error}")

            callbacks = getattr(self, "callbacks", None)
            if callbacks is not None:
                try:
                    callbacks.on_training_end(final_logs, synchronize=False)
                except Exception as callback_error:
                    self.logger.warning(
                        f"Failed to finalize RL callbacks: {callback_error}"
                    )
            try:
                self.close()
            except Exception as close_error:
                self.logger.warning(f"Failed to close RL environments: {close_error}")
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                try:
                    accelerator.cleanup()
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to clean up RL accelerator: {cleanup_error}"
                    )
            if callbacks is not None:
                try:
                    callbacks.on_training_finalized(final_logs)
                except Exception as callback_error:
                    self.logger.warning(
                        f"Failed to run finalized RL callbacks: {callback_error}"
                    )

        if pending_error is not None:
            raise pending_error

    def setup(self) -> None:
        """Set up runtime, environments, algorithm state, and callbacks."""
        self._setup()

    def _setup(self) -> None:
        set_seeds(self.seed, self.deterministic)
        self.setup_accelerator()
        self.setup_environment()
        self.setup_algorithm()
        (
            self.models,
            self.optimizers,
            _,
            self.schedulers,
            _,
        ) = self.accelerator.prepare(
            models=self.models,
            optimizers=self.optimizers,
            criterions={},
            schedulers=self.schedulers,
            dataloaders={},
        )
        self.setup_episode_managers()
        self.setup_callbacks()
        if not self.continue_model and self.config.get("auto_resume_local", False):
            checkpoint_dir = getattr(self, "checkpoint_dir", None)
            if checkpoint_dir is not None:
                checkpoint_dirs = [checkpoint_dir]
                artifact_manager = getattr(self, "artifact_manager", None)
                if artifact_manager is not None:
                    legacy_dir = Path(
                        get_legacy_run_artifact_dir(
                            run_name=artifact_manager.run_name,
                            output_dir=str(artifact_manager.output_dir),
                            experiment_name=artifact_manager.experiment_name,
                            sweep_name=artifact_manager.sweep_name,
                        )
                    ) / "final" / "checkpoints"
                    if str(legacy_dir) != checkpoint_dir:
                        checkpoint_dirs.append(str(legacy_dir))

                unreadable_dirs: list[str] = []
                for candidate_dir in checkpoint_dirs:
                    try:
                        self.continue_model = find_latest_checkpoint_local(
                            candidate_dir
                        )
                    except RuntimeError:
                        unreadable_dirs.append(candidate_dir)
                    if self.continue_model:
                        break
                if not self.continue_model and unreadable_dirs:
                    raise RuntimeError(
                        "Checkpoint artifacts exist but none can be loaded from "
                        f"{', '.join(unreadable_dirs)}"
                    )
                if self.continue_model:
                    self.trainer_config["continue_model"] = self.continue_model
                    try:
                        self.artifact_manager.save_config(self.config)
                    except Exception as error:
                        self.logger.warning(
                            f"Failed to persist auto-resume checkpoint path: {error}"
                        )
                    self.logger.info(
                        f"Auto-resuming from local checkpoint: {self.continue_model}"
                    )
        if self.continue_model:
            self.load_checkpoint(str(self.continue_model))

    def setup_accelerator(self) -> None:
        """Create the configured CPU or single-GPU accelerator."""
        self._setup_accelerator()

    def _setup_accelerator(self) -> None:
        accelerator_config = self.config.get("accelerator", {"type": "cpu"})
        if isinstance(accelerator_config, str):
            accelerator_config = {"type": accelerator_config}
        if not isinstance(accelerator_config, dict):
            raise TypeError("accelerator must be a string or mapping")
        accelerator_type = str(accelerator_config.get("type", "cpu"))
        if accelerator_type == "multi_gpu":
            raise NotImplementedError(
                "RLTrainer does not yet support distributed environment collection"
            )
        self.accelerator = ACCELERATOR_REGISTRY.get(
            accelerator_type,
            dict(accelerator_config),
        )

    def setup_environment(self) -> None:
        """Create independent training and evaluation environments."""
        self._setup_environment()

    def _setup_environment(self) -> None:
        from dl_core.environments import make_environment

        environment_config = self.config.get("environment")
        if not isinstance(environment_config, dict):
            raise TypeError("environment must be a mapping")
        evaluation_config = self.config.get("evaluation_environment", environment_config)
        if not isinstance(evaluation_config, dict):
            raise TypeError("evaluation_environment must be a mapping")
        self.environment = BatchedEnvironment(
            make_environment(dict(environment_config))
        )
        self.evaluation_environment = BatchedEnvironment(
            make_environment(dict(evaluation_config))
        )
        if self.evaluation_environment.num_envs != 1:
            raise ValueError(
                "evaluation_environment must contain exactly one environment; "
                "use a scalar evaluation configuration for deterministic episodes"
            )

    def setup_episode_managers(self) -> None:
        """Create configured episode managers and attach this trainer."""
        self._setup_episode_managers()

    def _setup_episode_managers(self) -> None:
        manager_configs = self.config.get(
            "episode_managers",
            {"standard": {}},
        )
        if not isinstance(manager_configs, dict):
            raise TypeError("episode_managers must be a mapping")
        self.episode_managers = {}
        for manager_name, manager_config in manager_configs.items():
            if manager_config is None:
                manager_config = {}
            if not isinstance(manager_config, dict):
                raise TypeError(
                    f"episode_managers.{manager_name} must be a mapping"
                )
            manager = EPISODE_MANAGER_REGISTRY.get(
                manager_name,
                dict(manager_config),
                artifact_manager=self.artifact_manager,
                trainer=self,
            )
            manager.set_name(manager_name)
            self.episode_managers[manager_name] = manager

    def setup_callbacks(self) -> None:
        """Create configured callbacks and attach this trainer."""
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        callback_instances = []
        callbacks_config = self.config.get("callbacks", {})
        if not isinstance(callbacks_config, dict):
            raise TypeError("callbacks must be a mapping")
        for callback_name, callback_config in callbacks_config.items():
            if callback_config is None:
                callback_config = {}
            if not isinstance(callback_config, dict):
                raise TypeError(f"callbacks.{callback_name} must be a mapping")
            callback_instances.append(
                CALLBACK_REGISTRY.get(callback_name, **callback_config)
            )
        self.callbacks = CallbackList(callback_instances)
        self.callbacks.set_trainer(self)

    def setup_artifact_manager(self) -> None:
        """Create the standard artifact layout before worker logging starts."""
        self._setup_artifact_manager()

    def _setup_artifact_manager(self) -> None:
        runtime_config = self.config.get("runtime", {})
        if not isinstance(runtime_config, dict):
            raise TypeError("runtime must be a mapping")
        config_path = self.config.get("_config_path")
        experiment_name = resolve_config_experiment_name(
            self.config,
            config_path=config_path,
        )
        run_name = resolve_config_run_name(
            self.config,
            config_path=config_path,
            fallback=self.__class__.__name__,
        )
        sweep_file = self.config.get("sweep_file")
        sweep_name = Path(sweep_file).stem if sweep_file else None
        self.artifact_manager = ArtifactManager(
            run_name=run_name,
            output_dir=str(runtime_config.get("output_dir", "artifacts")),
            experiment_name=experiment_name,
            sweep_name=sweep_name,
        )
        self.checkpoint_dir = str(self.artifact_manager.get_checkpoints_dir())
        self.artifact_manager.save_config(self.config)

    def perform_training(self) -> None:
        """Run episodes until the configured training budget is exhausted."""
        try:
            self._perform_training()
        finally:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None

    def _perform_training(self) -> None:
        next_checkpoint_step = (
            (self.global_step // self.checkpoint_frequency_steps + 1)
            * self.checkpoint_frequency_steps
            if self.checkpoint_frequency_steps > 0
            else None
        )
        if self.show_progress and self.accelerator.is_main_process():
            progress_uses_steps = self.total_timesteps > 0
            progress_total = (
                self.total_timesteps
                if progress_uses_steps
                else int(self.max_episodes or 0)
            )
            progress_value = (
                self.global_step if progress_uses_steps else self.current_episode
            )
            self._progress_bar = tqdm(
                total=progress_total,
                initial=min(progress_value, progress_total),
                desc="RL training",
                unit="step" if progress_uses_steps else "episode",
                dynamic_ncols=True,
            )

        if self.environment.num_envs == 1:
            last_evaluation_episode = -1
            while not self.stop_training:
                if (
                    self.total_timesteps > 0
                    and self.global_step >= self.total_timesteps
                ):
                    break
                if (
                    self.max_episodes is not None
                    and self.current_episode >= self.max_episodes
                ):
                    break

                result = self.run_episode(
                    training=True,
                    episode=self.current_episode,
                )
                episode_logs = {
                    "episode": result.episode,
                    "global_step": self.global_step,
                    **result.metrics,
                    "episode/terminated": result.terminated,
                    "episode/truncated": result.truncated,
                }
                self.episode_metrics.append(episode_logs)
                if self._progress_bar is not None:
                    progress_value = (
                        self.global_step
                        if self.total_timesteps > 0
                        else self.current_episode
                    )
                    self._progress_bar.update(
                        max(
                            0,
                            min(progress_value, self._progress_bar.total)
                            - self._progress_bar.n,
                        )
                    )
                    self._progress_bar.set_postfix(
                        episodes=self.current_episode,
                        updates=self.update_step,
                        refresh=False,
                    )

                if (
                    self.evaluation_frequency > 0
                    and self.evaluation_episodes > 0
                    and self.current_episode % self.evaluation_frequency == 0
                ):
                    self.evaluate()
                    last_evaluation_episode = self.current_episode

                if (
                    self.checkpoint_frequency > 0
                    and self.current_episode % self.checkpoint_frequency == 0
                ):
                    self.save_checkpoint(
                        f"episode_{self.current_episode:08d}.pth"
                    )
                if (
                    next_checkpoint_step is not None
                    and self.global_step >= next_checkpoint_step
                ):
                    self.save_checkpoint(
                        f"step_{self.global_step:012d}.pth"
                    )
                    while next_checkpoint_step <= self.global_step:
                        next_checkpoint_step += self.checkpoint_frequency_steps

            if (
                self.evaluation_episodes > 0
                and last_evaluation_episode != self.current_episode
            ):
                self.evaluate()
            self.save_checkpoint("latest.pth")
            return

        num_envs = self.environment.num_envs
        episode_numbers = np.arange(
            self.current_episode,
            self.current_episode + num_envs,
            dtype=np.int64,
        )
        next_episode_number = int(episode_numbers[-1]) + 1
        seeds = [self.seed + int(episode) for episode in episode_numbers]
        observations, reset_infos = self.environment.reset_batch(seeds)
        policy_state = self.initialize_policy_state(
            num_envs,
            evaluation=False,
        )
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        for environment_index in range(num_envs):
            observation = self.environment.batch_item(
                observations,
                environment_index,
            )
            episode = int(episode_numbers[environment_index])
            episode_id = f"train-{episode:08d}-env-{environment_index:04d}"
            context = EpisodeContext(
                episode_id=episode_id,
                episode=episode,
                environment_index=environment_index,
                phase="train",
                seed=seeds[environment_index],
                initial_observation=observation,
                reset_info=reset_infos[environment_index],
                start_global_step=self.global_step,
            )
            for manager in self.episode_managers.values():
                manager.begin_episode(context)
            self.callbacks.on_episode_start(
                episode,
                {
                    **reset_infos[environment_index],
                    "episode_id": episode_id,
                    "environment_index": environment_index,
                    "phase": "train",
                    "global_step": self.global_step,
                },
            )

        next_evaluation_episode = (
            self.current_episode + self.evaluation_frequency
            if self.evaluation_frequency > 0
            else None
        )
        next_checkpoint_episode = (
            self.current_episode + self.checkpoint_frequency
            if self.checkpoint_frequency > 0
            else None
        )
        last_evaluation_episode = -1
        overlap_environment_steps = bool(
            getattr(self, "overlap_environment_steps", False)
            and self.environment.supports_async_step
        )
        pending_transition_batch: TransitionBatch[Any, Any] | None = None
        pending_phase_timings: dict[str, float] | None = None
        while not self.stop_training:
            if self.total_timesteps > 0 and self.global_step >= self.total_timesteps:
                break
            if self.max_episodes is not None and self.current_episode >= self.max_episodes:
                break

            action_selection_start = perf_counter()
            action_output = self.select_actions_with_state(
                observations,
                policy_state,
                deterministic=False,
            )
            policy_state = action_output.policy_state
            action_selection_ms = (
                perf_counter() - action_selection_start
            ) * 1000.0
            actions = action_output.actions
            action_infos = action_output.action_info
            if len(actions) != num_envs:
                raise ValueError(
                    "Batched action selection must return one action per environment"
                )
            if len(action_infos) != num_envs:
                raise ValueError(
                    "Batched action metadata must contain one mapping per environment"
                )

            environment_dispatch_start = perf_counter()
            self.environment.step_batch_async(actions)
            environment_dispatch_ms = (
                perf_counter() - environment_dispatch_start
            ) * 1000.0
            learner_update_ms = 0.0
            update_logs: list[dict[str, float]] = []
            if overlap_environment_steps and pending_transition_batch is not None:
                assert pending_phase_timings is not None
                transition_batch_to_process = pending_transition_batch
                phase_timings = pending_phase_timings
                pending_transition_batch = None
                pending_phase_timings = None
                learner_update_start = perf_counter()
                update_logs = self.process_transition_batch(
                    transition_batch_to_process
                )
                learner_update_ms = (
                    perf_counter() - learner_update_start
                ) * 1000.0
                self._emit_vector_update_logs(
                    update_logs,
                    global_step=self.global_step,
                    phase_timings={
                        **phase_timings,
                        "rl/timing/learner_update_ms": learner_update_ms,
                    },
                )

            environment_wait_start = perf_counter()
            (
                next_observations,
                rewards,
                environment_terminated,
                environment_truncated,
                lane_infos,
                final_observations,
            ) = self.environment.step_batch_wait()
            environment_wait_ms = (
                perf_counter() - environment_wait_start
            ) * 1000.0
            self._validate_rewards(rewards, source="Environment")
            self.collector_step += 1
            transition_processing_start = perf_counter()
            episode_lengths += 1
            episode_returns += rewards
            trainer_truncated = episode_lengths >= self.max_episode_steps
            if (
                self.total_timesteps > 0
                and self.global_step + num_envs >= self.total_timesteps
            ):
                trainer_truncated = np.ones(num_envs, dtype=np.bool_)
            terminated = environment_terminated.copy()
            truncated = np.logical_or(
                environment_truncated,
                np.logical_and(trainer_truncated, ~terminated),
            )
            done = np.logical_or(terminated, truncated)

            final_infos: list[dict[str, Any]] = []
            for environment_index in range(num_envs):
                final_info = lane_infos[environment_index]
                if environment_terminated[environment_index] or (
                    environment_truncated[environment_index]
                ):
                    final_info = dict(
                        final_info.get("final_info", final_info)
                    )
                transition = Transition(
                    observation=self.environment.batch_item(
                        observations,
                        environment_index,
                    ),
                    action=actions[environment_index],
                    reward=float(rewards[environment_index]),
                    next_observation=final_observations[environment_index],
                    terminated=bool(terminated[environment_index]),
                    truncated=bool(truncated[environment_index]),
                    info=final_info,
                    action_info=action_infos[environment_index],
                )
                final_infos.append(final_info)
                for manager in self.episode_managers.values():
                    manager.record_transition(
                        environment_index,
                        transition,
                        phase="train",
                    )
            self.global_step += num_envs
            batched_final_observations = self.environment.stack_values(
                final_observations
            )
            transition_batch = TransitionBatch(
                observations=observations,
                actions=self.environment.stack_values(actions),
                rewards=rewards,
                next_observations=batched_final_observations,
                terminated=terminated,
                truncated=truncated,
                infos=final_infos,
                action_info=action_infos,
                final_observations=batched_final_observations,
            )
            transition_batch = self.prepare_transition_batch(
                transition_batch
            )
            if overlap_environment_steps:
                pending_transition_batch = transition_batch
            else:
                learner_update_start = perf_counter()
                update_logs = self.process_transition_batch(transition_batch)
                learner_update_ms = (
                    perf_counter() - learner_update_start
                ) * 1000.0
            transition_processing_ms = (
                perf_counter() - transition_processing_start
            ) * 1000.0
            if not overlap_environment_steps:
                transition_processing_ms -= learner_update_ms
            transition_processing_ms = max(transition_processing_ms, 0.0)
            phase_timings = {
                "rl/timing/action_selection_ms": action_selection_ms,
                "rl/timing/environment_dispatch_ms": environment_dispatch_ms,
                "rl/timing/environment_wait_ms": environment_wait_ms,
                "rl/timing/learner_update_ms": learner_update_ms,
                "rl/timing/transition_processing_ms": transition_processing_ms,
                "rl/timing/collector_cycle_ms": (
                    action_selection_ms
                    + environment_dispatch_ms
                    + environment_wait_ms
                    + transition_processing_ms
                ),
                "rl/collection_overlap_enabled": float(
                    overlap_environment_steps
                ),
            }
            if overlap_environment_steps:
                pending_phase_timings = phase_timings
            else:
                self._emit_vector_update_logs(
                    update_logs,
                    global_step=self.global_step,
                    phase_timings=phase_timings,
                )

            for environment_index in range(num_envs):
                final_info = final_infos[environment_index]
                if not done[environment_index]:
                    continue

                episode = int(episode_numbers[environment_index])
                result = EpisodeResult(
                    episode=episode,
                    episode_return=float(episode_returns[environment_index]),
                    length=int(episode_lengths[environment_index]),
                    terminated=bool(terminated[environment_index]),
                    truncated=bool(truncated[environment_index]),
                    final_info=final_info,
                    environment_index=environment_index,
                    completion_reason=(
                        "terminated"
                        if terminated[environment_index]
                        else "truncated"
                    ),
                )
                for manager in self.episode_managers.values():
                    manager.end_episode(
                        environment_index,
                        result,
                        phase="train",
                    )
                episode_logs = {
                    "episode": episode,
                    "environment_index": environment_index,
                    "global_step": self.global_step,
                    **result.metrics,
                    "episode/terminated": result.terminated,
                    "episode/truncated": result.truncated,
                }
                self.episode_metrics.append(episode_logs)
                self.callbacks.on_episode_end(episode, episode_logs)
                self.current_episode += 1
                self.current_epoch = self.current_episode

            step_checkpoint_due = (
                next_checkpoint_step is not None
                and self.global_step >= next_checkpoint_step
            )
            evaluation_due = (
                next_evaluation_episode is not None
                and self.evaluation_episodes > 0
                and self.current_episode >= next_evaluation_episode
            )
            episode_checkpoint_due = (
                next_checkpoint_episode is not None
                and self.current_episode >= next_checkpoint_episode
            )
            budget_reached = (
                self.total_timesteps > 0
                and self.global_step >= self.total_timesteps
            )
            episode_budget_reached = (
                self.max_episodes is not None
                and self.current_episode >= self.max_episodes
            )
            if (
                overlap_environment_steps
                and pending_transition_batch is not None
                and (
                    step_checkpoint_due
                    or evaluation_due
                    or episode_checkpoint_due
                    or budget_reached
                    or episode_budget_reached
                    or self.stop_training
                )
            ):
                assert pending_phase_timings is not None
                transition_batch_to_process = pending_transition_batch
                phase_timings = pending_phase_timings
                pending_transition_batch = None
                pending_phase_timings = None
                learner_update_start = perf_counter()
                flushed_update_logs = self.process_transition_batch(
                    transition_batch_to_process
                )
                flushed_learner_update_ms = (
                    perf_counter() - learner_update_start
                ) * 1000.0
                self._emit_vector_update_logs(
                    flushed_update_logs,
                    global_step=self.global_step,
                    phase_timings={
                        **phase_timings,
                        "rl/timing/learner_update_ms": flushed_learner_update_ms,
                    },
                )

            if self._progress_bar is not None:
                progress_value = (
                    self.global_step
                    if self.total_timesteps > 0
                    else self.current_episode
                )
                self._progress_bar.update(
                    max(
                        0,
                        min(progress_value, self._progress_bar.total)
                        - self._progress_bar.n,
                    )
                )
                self._progress_bar.set_postfix(
                    episodes=self.current_episode,
                    updates=self.update_step,
                    refresh=False,
                )
            if step_checkpoint_due:
                self.save_checkpoint(
                    f"step_{self.global_step:012d}.pth"
                )
                while next_checkpoint_step <= self.global_step:
                    next_checkpoint_step += self.checkpoint_frequency_steps

            if budget_reached or episode_budget_reached or self.stop_training:
                break

            forced_reset = np.logical_and(
                done,
                ~np.logical_or(
                    environment_terminated,
                    environment_truncated,
                ),
            )
            if forced_reset.any():
                reset_observations, forced_reset_infos = (
                    self.environment.reset_lanes(forced_reset)
                )
                next_observations = self.environment.replace_batch_items(
                    next_observations,
                    reset_observations,
                    forced_reset,
                )
            else:
                forced_reset_infos = [{} for _ in range(num_envs)]

            for environment_index in np.flatnonzero(done):
                index = int(environment_index)
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
                episode_numbers[index] = next_episode_number
                next_episode_number += 1
                observation = self.environment.batch_item(
                    next_observations,
                    index,
                )
                episode = int(episode_numbers[index])
                episode_id = f"train-{episode:08d}-env-{index:04d}"
                reset_info = {
                    key: value
                    for key, value in lane_infos[index].items()
                    if key not in {"final_obs", "final_info"}
                }
                if forced_reset[index]:
                    reset_info = forced_reset_infos[index]
                context = EpisodeContext(
                    episode_id=episode_id,
                    episode=episode,
                    environment_index=index,
                    phase="train",
                    seed=None,
                    initial_observation=observation,
                    reset_info=reset_info,
                    start_global_step=self.global_step,
                )
                for manager in self.episode_managers.values():
                    manager.begin_episode(context)
                self.callbacks.on_episode_start(
                    episode,
                    {
                        **reset_info,
                        "episode_id": episode_id,
                        "environment_index": index,
                        "phase": "train",
                        "global_step": self.global_step,
                    },
                )
            if done.any():
                policy_state = self.reset_policy_state(
                    policy_state,
                    done,
                    evaluation=False,
                )
            observations = next_observations

            if evaluation_due:
                self.evaluate()
                last_evaluation_episode = self.current_episode
                while next_evaluation_episode <= self.current_episode:
                    next_evaluation_episode += self.evaluation_frequency
            if episode_checkpoint_due:
                self.save_checkpoint(
                    f"episode_{self.current_episode:08d}.pth"
                )
                while next_checkpoint_episode <= self.current_episode:
                    next_checkpoint_episode += self.checkpoint_frequency

        if pending_transition_batch is not None:
            assert pending_phase_timings is not None
            learner_update_start = perf_counter()
            final_update_logs = self.process_transition_batch(
                pending_transition_batch
            )
            final_learner_update_ms = (
                perf_counter() - learner_update_start
            ) * 1000.0
            self._emit_vector_update_logs(
                final_update_logs,
                global_step=self.global_step,
                phase_timings={
                    **pending_phase_timings,
                    "rl/timing/learner_update_ms": final_learner_update_ms,
                },
            )

        for environment_index in range(num_envs):
            for manager in self.episode_managers.values():
                manager.abort_episode(environment_index, phase="train")
        if (
            self.evaluation_episodes > 0
            and last_evaluation_episode != self.current_episode
        ):
            self.evaluate()
        self.save_checkpoint("latest.pth")

    def _emit_vector_update_logs(
        self,
        update_logs: list[dict[str, float]],
        *,
        global_step: int,
        phase_timings: dict[str, float],
    ) -> None:
        for logs in update_logs:
            self.update_step += 1
            self.callbacks.on_update_end(
                self.update_step,
                {
                    **logs,
                    **phase_timings,
                    "update": float(self.update_step),
                    "global_step": float(global_step),
                },
            )

    def run_episode(self, *, training: bool, episode: int) -> EpisodeResult:
        """Run one training or evaluation episode."""
        return self._run_episode(training=training, episode=episode)

    def _run_episode(self, *, training: bool, episode: int) -> EpisodeResult:
        environment = self.environment if training else self.evaluation_environment
        if environment.num_envs != 1:
            raise RuntimeError("run_episode requires a scalar environment")
        phase = "train" if training else "evaluation"
        episode_seed = self.seed + episode + (0 if training else 1_000_000)
        observations, reset_infos = environment.reset_batch([episode_seed])
        observation = environment.batch_item(observations, 0)
        reset_info = reset_infos[0]
        episode_id = f"{phase}-{episode:08d}-env-0000"
        context = EpisodeContext(
            episode_id=episode_id,
            episode=episode,
            environment_index=0,
            phase=phase,
            seed=episode_seed,
            initial_observation=observation,
            reset_info=reset_info,
            start_global_step=self.global_step,
        )
        for manager in self.episode_managers.values():
            manager.begin_episode(context)
        self.callbacks.on_episode_start(
            episode,
            {
                **reset_info,
                "episode_id": episode_id,
                "phase": phase,
                "global_step": self.global_step,
            },
        )

        episode_return = 0.0
        length = 0
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}
        policy_state = self.initialize_policy_state(
            1,
            evaluation=not training,
        )
        while not (terminated or truncated):
            action_output = self.select_action_with_state(
                observation,
                policy_state,
                deterministic=not training,
            )
            action = action_output.action
            action_info = action_output.info
            policy_state = action_output.policy_state
            (
                next_observations,
                rewards,
                terminated_batch,
                truncated_batch,
                lane_infos,
                final_observations,
            ) = environment.step_batch([action])
            reward = float(rewards[0])
            self._validate_rewards(reward, source="Environment")
            if training:
                self.collector_step += 1
            next_observation = final_observations[0]
            terminated = bool(terminated_batch[0])
            truncated = bool(truncated_batch[0])
            final_info = lane_infos[0]
            if terminated or truncated:
                final_info = dict(final_info.get("final_info", final_info))
            returned_observation = environment.batch_item(
                next_observations,
                0,
            )
            length += 1
            episode_return += reward

            if training:
                self.global_step += 1
            if length >= self.max_episode_steps and not (terminated or truncated):
                truncated = True
            if (
                training
                and self.total_timesteps > 0
                and self.global_step >= self.total_timesteps
                and not (terminated or truncated)
            ):
                truncated = True

            transition = Transition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=final_info,
                action_info=action_info,
            )
            for manager in self.episode_managers.values():
                manager.record_transition(0, transition, phase=phase)
            if training:
                update_logs = self.process_transition(
                    self.prepare_transition(transition)
                )
                if update_logs is not None:
                    self.update_step += 1
                    update_logs = {
                        **update_logs,
                        "update": float(self.update_step),
                        "global_step": float(self.global_step),
                    }
                    self.callbacks.on_update_end(self.update_step, update_logs)
                if self._progress_bar is not None and self.total_timesteps > 0:
                    self._progress_bar.update(
                        min(self.global_step, self._progress_bar.total)
                        - self._progress_bar.n
                    )
                    self._progress_bar.set_postfix(
                        episodes=self.current_episode,
                        updates=self.update_step,
                        refresh=False,
                    )
            observation = returned_observation

        result = EpisodeResult(
            episode=episode,
            episode_return=episode_return,
            length=length,
            terminated=terminated,
            truncated=truncated,
            final_info=final_info,
            episode_id=episode_id,
            environment_index=0,
            seed=episode_seed,
            completion_reason=(
                "terminated" if terminated else "truncated"
            ),
        )
        for manager in self.episode_managers.values():
            manager.end_episode(0, result, phase=phase)
        episode_logs: dict[str, Any] = {
            "phase": phase,
            "global_step": self.global_step,
            **result.metrics,
            "episode/terminated": result.terminated,
            "episode/truncated": result.truncated,
        }
        self.callbacks.on_episode_end(episode, episode_logs)
        if training:
            self.current_episode += 1
            self.current_epoch = self.current_episode
        return result

    def evaluate(self) -> dict[str, float]:
        """Evaluate the deterministic policy in the independent evaluation environment."""
        return self._evaluate()

    def _evaluate(self) -> dict[str, float]:
        if self.evaluation_episodes == 0:
            raise RuntimeError("evaluation_episodes must be positive to evaluate")

        model_modes = {name: model.training for name, model in self.models.items()}
        try:
            for model in self.models.values():
                model.eval()
            with torch.no_grad():
                results = [
                    self.run_episode(
                        training=False,
                        episode=(
                            self.current_episode * self.evaluation_episodes
                        )
                        + index,
                    )
                    for index in range(self.evaluation_episodes)
                ]
        finally:
            for name, model in self.models.items():
                model.train(model_modes[name])
        returns = np.asarray([result.episode_return for result in results], dtype=float)
        lengths = np.asarray([result.length for result in results], dtype=float)
        metrics = {
            "evaluation/mean_return": float(returns.mean()),
            "evaluation/std_return": float(returns.std()),
            "evaluation/mean_length": float(lengths.mean()),
            "global_step": float(self.global_step),
        }
        successes = [
            float(result.final_info["is_success"])
            for result in results
            if isinstance(result.final_info.get("is_success"), (bool, int, float))
        ]
        if successes:
            metrics["evaluation/success_rate"] = float(np.mean(successes))
        self.evaluation_metrics.append(metrics)
        self.artifact_manager.append_final_jsonl("metrics/evaluations.jsonl", metrics)
        self.callbacks.on_evaluation_end(self.global_step, metrics)
        return metrics

    def save_checkpoint(self, filename: str = "latest.pth") -> Path | None:
        """Persist common and algorithm-specific RL state."""
        return self._save_checkpoint(filename)

    def _save_checkpoint(self, filename: str) -> Path | None:
        if not self.accelerator.is_main_process():
            return None
        callback_states = {}
        for index, callback in enumerate(self.callbacks.callbacks):
            state = callback.get_state()
            if state:
                callback_states[f"callback_{index}_{callback.__class__.__name__}"] = state
        checkpoint = {
            "trainer_type": "reinforcement_learning",
            "trainer_name": self.trainer_name,
            "trainer_class": (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            ),
            "config": self.config,
            "current_episode": self.current_episode,
            "global_step": self.global_step,
            "collector_step": self.collector_step,
            "update_step": self.update_step,
            "episode_metrics": self.episode_metrics,
            "evaluation_metrics": self.evaluation_metrics,
            "models_state_dict": {
                name: self.accelerator.unwrap_model(model).state_dict()
                for name, model in self.models.items()
            },
            "optimizers_state_dict": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            "schedulers_state_dict": {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            },
            "callback_states": callback_states,
            "episode_manager_states": {
                name: manager.state_dict()
                for name, manager in self.episode_managers.items()
            },
            "algorithm_state": self.algorithm_state_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        checkpoint.update(self.accelerator.get_accelerator_state())
        if torch.cuda.is_available():
            checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()
        checkpoint_path = self.artifact_manager.get_final_checkpoint_path(filename)
        return atomic_torch_save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restore common and algorithm-specific RL state."""
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.accelerator.get_device(),
            weights_only=False,
        )
        if checkpoint.get("trainer_type") != "reinforcement_learning":
            raise ValueError("Checkpoint is not an RLTrainer checkpoint")
        trainer_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        checkpoint_trainer_class = checkpoint.get("trainer_class")
        if (
            checkpoint_trainer_class is not None
            and checkpoint_trainer_class != trainer_class
        ):
            raise ValueError(
                f"Checkpoint trainer '{checkpoint_trainer_class}' does not match "
                f"'{trainer_class}'"
            )

        model_states = checkpoint.get("models_state_dict", {})
        if set(model_states) != set(self.models):
            raise ValueError("Checkpoint model names do not match the configured models")
        for name, state in model_states.items():
            self.accelerator.unwrap_model(self.models[name]).load_state_dict(state)

        optimizer_states = checkpoint.get("optimizers_state_dict", {})
        if set(optimizer_states) != set(self.optimizers):
            raise ValueError(
                "Checkpoint optimizer names do not match the configured optimizers"
            )
        for name, state in optimizer_states.items():
            self.optimizers[name].load_state_dict(state)

        scheduler_states = checkpoint.get("schedulers_state_dict", {})
        if set(scheduler_states) != set(self.schedulers):
            raise ValueError(
                "Checkpoint scheduler names do not match the configured schedulers"
            )
        for name, state in scheduler_states.items():
            self.schedulers[name].load_state_dict(state)
        self.accelerator.load_accelerator_state(checkpoint)

        self.current_episode = int(checkpoint.get("current_episode", 0))
        self.current_epoch = self.current_episode
        self.global_step = int(checkpoint.get("global_step", 0))
        self.collector_step = int(checkpoint.get("collector_step", 0))
        self.update_step = int(checkpoint.get("update_step", 0))
        self.episode_metrics = list(checkpoint.get("episode_metrics", []))
        self.evaluation_metrics = list(checkpoint.get("evaluation_metrics", []))
        self.load_algorithm_state_dict(checkpoint.get("algorithm_state", {}))

        callback_states = checkpoint.get("callback_states", {})
        for index, callback in enumerate(self.callbacks.callbacks):
            key = f"callback_{index}_{callback.__class__.__name__}"
            if key in callback_states:
                callback.set_state(callback_states[key])
        manager_states = checkpoint.get("episode_manager_states")
        if manager_states is not None:
            if set(manager_states) != set(self.episode_managers):
                raise ValueError(
                    "Checkpoint episode manager names do not match configuration"
                )
            for name, state in manager_states.items():
                self.episode_managers[name].load_state_dict(state)
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )

    def close(self) -> None:
        """Close environments owned by this trainer."""
        self._close()

    def _close(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        close_error: Exception | None = None
        for attribute in ("environment", "evaluation_environment"):
            environment = getattr(self, attribute, None)
            if environment is not None:
                try:
                    environment.close()
                except Exception as error:
                    close_error = close_error or error
        if close_error is not None:
            raise close_error

    @abstractmethod
    def setup_algorithm(self) -> None:
        """Create algorithm-specific models, optimizers, buffers, and state."""

    @abstractmethod
    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
    ) -> Any | ActionOutput[Any]:
        """Select an action for one observation."""

    @abstractmethod
    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float] | None:
        """Consume a training transition and optionally report update metrics."""

    def select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[Any]:
        """Select one action per environment lane."""
        return self._select_actions(
            observations,
            deterministic=deterministic,
        )

    def _select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[Any]:
        actions: list[Any] = []
        action_info: list[dict[str, Any]] = []
        for environment_index in range(self.environment.num_envs):
            output = self.select_action(
                self.environment.batch_item(observations, environment_index),
                deterministic=deterministic,
            )
            if isinstance(output, ActionOutput):
                actions.append(output.action)
                action_info.append(output.info)
            else:
                actions.append(output)
                action_info.append({})
        return BatchActionOutput(actions=actions, action_info=action_info)

    def initialize_policy_state(
        self,
        batch_size: int,
        *,
        evaluation: bool,
    ) -> Any:
        """Create recurrent policy state for newly reset environment lanes."""
        return self._initialize_policy_state(
            batch_size,
            evaluation=evaluation,
        )

    def _initialize_policy_state(
        self,
        batch_size: int,
        *,
        evaluation: bool,
    ) -> Any:
        if batch_size <= 0:
            raise ValueError("Policy-state batch size must be positive")
        del evaluation
        return None

    def reset_policy_state(
        self,
        policy_state: Any,
        done: np.ndarray,
        *,
        evaluation: bool,
    ) -> Any:
        """Reset recurrent state for lanes whose episodes have ended."""
        return self._reset_policy_state(
            policy_state,
            done,
            evaluation=evaluation,
        )

    def _reset_policy_state(
        self,
        policy_state: Any,
        done: np.ndarray,
        *,
        evaluation: bool,
    ) -> Any:
        done = np.asarray(done, dtype=np.bool_)
        if done.ndim != 1:
            raise ValueError("Policy-state done mask must be one-dimensional")
        del evaluation
        return policy_state

    def select_action_with_state(
        self,
        observation: Any,
        policy_state: Any,
        *,
        deterministic: bool,
    ) -> ActionOutput[Any]:
        """Select one action while carrying recurrent policy state."""
        return self._select_action_with_state(
            observation,
            policy_state,
            deterministic=deterministic,
        )

    def _select_action_with_state(
        self,
        observation: Any,
        policy_state: Any,
        *,
        deterministic: bool,
    ) -> ActionOutput[Any]:
        output = self.select_action(
            observation,
            deterministic=deterministic,
        )
        if isinstance(output, ActionOutput):
            return ActionOutput(
                action=output.action,
                info=output.info,
                policy_state=(
                    policy_state
                    if output.policy_state is None
                    else output.policy_state
                ),
            )
        return ActionOutput(
            action=output,
            policy_state=policy_state,
        )

    def select_actions_with_state(
        self,
        observations: Any,
        policy_state: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[Any]:
        """Select a vector action while carrying per-lane policy state."""
        return self._select_actions_with_state(
            observations,
            policy_state,
            deterministic=deterministic,
        )

    def _select_actions_with_state(
        self,
        observations: Any,
        policy_state: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[Any]:
        output = self.select_actions(
            observations,
            deterministic=deterministic,
        )
        return BatchActionOutput(
            actions=output.actions,
            action_info=output.action_info,
            policy_state=(
                policy_state
                if output.policy_state is None
                else output.policy_state
            ),
        )

    def process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        """Consume one transition from every environment lane."""
        return self._process_transition_batch(transitions)

    def prepare_transition(
        self,
        transition: Transition[Any, Any],
    ) -> Transition[Any, Any]:
        """Prepare one collected transition for trainer consumption."""
        preparation_hook = self.transform_transition
        public_override = (
            getattr(preparation_hook, "__func__", None)
            is not RLTrainer.transform_transition
        )
        legacy_hook = getattr(self, "_prepare_transition", None)
        if not public_override and callable(legacy_hook):
            raise RuntimeError(
                "_prepare_transition() is no longer an extension hook; "
                "rename it to transform_transition()"
            )
        elif not public_override:
            prepared_transition = transition
        else:
            prepared_transition = preparation_hook(
                replace(
                    transition,
                    info=deepcopy(transition.info),
                    action_info=deepcopy(transition.action_info),
                )
            )
        if not isinstance(prepared_transition, Transition):
            raise TypeError("transform_transition must return a Transition")
        self._validate_rewards(
            prepared_transition.reward,
            source="Prepared transition",
        )
        return prepared_transition

    def transform_transition(
        self,
        transition: Transition[Any, Any],
    ) -> Transition[Any, Any]:
        """Customize one scalar transition before trainer consumption."""
        return transition

    def prepare_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> TransitionBatch[Any, Any]:
        """Prepare one vector step for trainer consumption."""
        preparation_hook = self.transform_transition_batch
        public_override = (
            getattr(preparation_hook, "__func__", None)
            is not RLTrainer.transform_transition_batch
        )
        legacy_hook = getattr(self, "_prepare_transition_batch", None)
        if not public_override and callable(legacy_hook):
            raise RuntimeError(
                "_prepare_transition_batch() is no longer an extension hook; "
                "rename it to transform_transition_batch()"
            )
        elif not public_override:
            prepared_transitions = transitions
        else:
            prepared_transitions = preparation_hook(
                replace(
                    transitions,
                    terminated=np.asarray(transitions.terminated).copy(),
                    truncated=np.asarray(transitions.truncated).copy(),
                    infos=deepcopy(transitions.infos),
                    action_info=deepcopy(transitions.action_info),
                )
            )
        if not isinstance(prepared_transitions, TransitionBatch):
            raise TypeError(
                "transform_transition_batch must return a TransitionBatch"
            )
        original_size = transitions.size
        if prepared_transitions.size != original_size:
            raise ValueError(
                "Prepared transition batch must preserve its environment lanes"
            )
        for name in ("rewards", "terminated", "truncated"):
            if np.asarray(getattr(prepared_transitions, name)).shape != (
                original_size,
            ):
                raise ValueError(
                    f"Prepared transition batch {name} must have shape "
                    f"({original_size},)"
                )
        for name in ("infos", "action_info"):
            values = getattr(prepared_transitions, name)
            if values and len(values) != original_size:
                raise ValueError(
                    f"Prepared transition batch {name} must contain one "
                    "mapping per environment lane"
                )
        self._validate_rewards(
            prepared_transitions.rewards,
            source="Prepared transition",
        )
        return prepared_transitions

    def _validate_rewards(self, rewards: Any, *, source: str) -> None:
        values = np.asarray(rewards)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{source} rewards must be numeric")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{source} rewards must be finite")

    def transform_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> TransitionBatch[Any, Any]:
        """Customize one vector transition before trainer consumption."""
        return transitions

    def should_update(
        self,
        global_step: int,
        transitions: TransitionBatch[Any, Any],
    ) -> bool:
        """Return whether an eligible replay-based update should run."""
        legacy_hook = getattr(self, "_should_update", None)
        if callable(legacy_hook):
            raise RuntimeError(
                "_should_update() is no longer an extension hook; rename it "
                "to should_update()"
            )
        del global_step, transitions
        return True

    def _process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        update_logs: list[dict[str, float]] = []
        batch_global_step = self.global_step
        try:
            for environment_index in range(transitions.size):
                self.global_step = (
                    batch_global_step - transitions.size + environment_index + 1
                )
                logs = self.process_transition(
                    Transition(
                        observation=self.environment.batch_item(
                            transitions.observations,
                            environment_index,
                        ),
                        action=self.environment.batch_item(
                            transitions.actions,
                            environment_index,
                        ),
                        reward=float(transitions.rewards[environment_index]),
                        next_observation=self.environment.batch_item(
                            transitions.next_observations,
                            environment_index,
                        ),
                        terminated=bool(
                            transitions.terminated[environment_index]
                        ),
                        truncated=bool(transitions.truncated[environment_index]),
                        info=(
                            transitions.infos[environment_index]
                            if transitions.infos
                            else {}
                        ),
                        action_info=(
                            transitions.action_info[environment_index]
                            if transitions.action_info
                            else {}
                        ),
                    )
                )
                if logs is not None:
                    update_logs.append(logs)
        finally:
            self.global_step = batch_global_step
        return update_logs

    @abstractmethod
    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return algorithm-specific state not held by models or optimizers."""

    @abstractmethod
    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific state."""
