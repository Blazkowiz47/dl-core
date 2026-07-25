"""Episode-driven trainer foundation for reinforcement-learning algorithms."""

from __future__ import annotations

import logging
import random
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dl_core.environments import make_environment
from dl_core.utils import ArtifactManager, set_seeds
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
        self._perform_training()

    def _perform_training(self) -> None:
        if self.environment.num_envs > 1:
            self._perform_vector_training()
            return
        last_evaluation_episode = -1
        while not self.stop_training:
            if self.total_timesteps > 0 and self.global_step >= self.total_timesteps:
                break
            if self.max_episodes is not None and self.current_episode >= self.max_episodes:
                break

            result = self.run_episode(training=True, episode=self.current_episode)
            episode_logs = {
                "episode": result.episode,
                "global_step": self.global_step,
                **result.metrics,
                "episode/terminated": result.terminated,
                "episode/truncated": result.truncated,
            }
            self.episode_metrics.append(episode_logs)

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
            self.evaluation_episodes > 0
            and last_evaluation_episode != self.current_episode
        ):
            self.evaluate()
        self.save_checkpoint("latest.pth")

    def _perform_vector_training(self) -> None:
        num_envs = self.environment.num_envs
        episode_numbers = np.arange(
            self.current_episode,
            self.current_episode + num_envs,
            dtype=np.int64,
        )
        next_episode_number = int(episode_numbers[-1]) + 1
        seeds = [self.seed + int(episode) for episode in episode_numbers]
        observations, reset_infos = self.environment.reset_batch(seeds)
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
        while not self.stop_training:
            if self.total_timesteps > 0 and self.global_step >= self.total_timesteps:
                break
            if self.max_episodes is not None and self.current_episode >= self.max_episodes:
                break

            action_output = self.select_actions(
                observations,
                deterministic=False,
            )
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

            (
                next_observations,
                rewards,
                environment_terminated,
                environment_truncated,
                lane_infos,
                final_observations,
            ) = self.environment.step_batch(actions)
            self.collector_step += 1
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
            update_logs = self.process_transition_batch(
                TransitionBatch(
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
            )
            for logs in update_logs:
                self.update_step += 1
                self.callbacks.on_update_end(
                    self.update_step,
                    {
                        **logs,
                        "update": float(self.update_step),
                        "global_step": float(self.global_step),
                    },
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

            budget_reached = (
                self.total_timesteps > 0
                and self.global_step >= self.total_timesteps
            )
            episode_budget_reached = (
                self.max_episodes is not None
                and self.current_episode >= self.max_episodes
            )
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
            observations = next_observations

            if (
                next_evaluation_episode is not None
                and self.evaluation_episodes > 0
                and self.current_episode >= next_evaluation_episode
            ):
                self.evaluate()
                last_evaluation_episode = self.current_episode
                while next_evaluation_episode <= self.current_episode:
                    next_evaluation_episode += self.evaluation_frequency
            if (
                next_checkpoint_episode is not None
                and self.current_episode >= next_checkpoint_episode
            ):
                self.save_checkpoint(
                    f"episode_{self.current_episode:08d}.pth"
                )
                while next_checkpoint_episode <= self.current_episode:
                    next_checkpoint_episode += self.checkpoint_frequency

        for environment_index in range(num_envs):
            for manager in self.episode_managers.values():
                manager.abort_episode(environment_index, phase="train")
        if (
            self.evaluation_episodes > 0
            and last_evaluation_episode != self.current_episode
        ):
            self.evaluate()
        self.save_checkpoint("latest.pth")

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
        while not (terminated or truncated):
            action_output = self.select_action(observation, deterministic=not training)
            if isinstance(action_output, ActionOutput):
                action = action_output.action
                action_info = action_output.info
            else:
                action = action_output
                action_info = {}
            (
                next_observations,
                rewards,
                terminated_batch,
                truncated_batch,
                lane_infos,
                final_observations,
            ) = environment.step_batch([action])
            if training:
                self.collector_step += 1
            next_observation = final_observations[0]
            reward = float(rewards[0])
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
                update_logs = self.process_transition(transition)
                if update_logs is not None:
                    self.update_step += 1
                    update_logs = {
                        **update_logs,
                        "update": float(self.update_step),
                        "global_step": float(self.global_step),
                    }
                    self.callbacks.on_update_end(self.update_step, update_logs)
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
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

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

    def process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        """Consume one transition from every environment lane."""
        return self._process_transition_batch(transitions)

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
