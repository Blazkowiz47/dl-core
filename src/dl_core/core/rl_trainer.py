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
from .config_metadata import config_field
from .registry import ACCELERATOR_REGISTRY, CALLBACK_REGISTRY
from .rl_types import ActionOutput, Environment, EpisodeResult, Transition


class RLTrainer(ABC):
    """Base class for episode-driven reinforcement-learning trainers."""

    CONFIG_FIELDS = [
        config_field(
            "total_timesteps",
            "int",
            "Maximum number of training environment transitions.",
            required=True,
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
        if self.total_timesteps <= 0 and self.max_episodes is None:
            raise ValueError("RL training requires total_timesteps or max_episodes")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.accelerator: BaseAccelerator
        self.environment: Environment[Any, Any]
        self.evaluation_environment: Environment[Any, Any]
        self.artifact_manager: ArtifactManager
        self.callbacks: CallbackList
        self.models: dict[str, nn.Module] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LRScheduler] = {}

        self.current_episode = 0
        self.current_epoch = 0
        self.global_step = 0
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
                callbacks.on_training_end(final_logs, synchronize=False)
            self.close()
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                accelerator.cleanup()
            if callbacks is not None:
                callbacks.on_training_finalized(final_logs)

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
        self.environment = make_environment(dict(environment_config))
        self.evaluation_environment = make_environment(dict(evaluation_config))

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
        last_evaluation_episode = -1
        while not self.stop_training:
            if self.total_timesteps > 0 and self.global_step >= self.total_timesteps:
                break
            if self.max_episodes is not None and self.current_episode >= self.max_episodes:
                break

            result = self.run_episode(training=True, episode=self.current_episode)
            episode_logs = {
                "episode": result.episode,
                "episode/return": result.episode_return,
                "episode/length": result.length,
                "episode/terminated": result.terminated,
                "episode/truncated": result.truncated,
                "global_step": self.global_step,
            }
            if isinstance(result.final_info.get("is_success"), (bool, int, float)):
                episode_logs["episode/success"] = float(
                    result.final_info["is_success"]
                )
            self.episode_metrics.append(episode_logs)
            self.artifact_manager.append_final_jsonl(
                "metrics/episodes.jsonl",
                episode_logs,
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
        phase = "train" if training else "evaluation"
        episode_seed = self.seed + episode + (0 if training else 1_000_000)
        observation, reset_info = environment.reset(seed=episode_seed)
        self.callbacks.on_episode_start(
            episode,
            {"phase": phase, "global_step": self.global_step, **reset_info},
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
            next_observation, reward, terminated, truncated, final_info = (
                environment.step(action)
            )
            length += 1
            episode_return += float(reward)

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
            if training:
                update_logs = self.process_transition(transition)
                if update_logs:
                    self.update_step += 1
                    self.callbacks.on_update_end(self.update_step, update_logs)
            observation = next_observation

        result = EpisodeResult(
            episode=episode,
            episode_return=episode_return,
            length=length,
            terminated=terminated,
            truncated=truncated,
            final_info=final_info,
        )
        self.callbacks.on_episode_end(
            episode,
            {
                "phase": phase,
                "episode/return": episode_return,
                "episode/length": length,
                "global_step": self.global_step,
            },
        )
        if training:
            self.current_episode += 1
            self.current_epoch = self.current_episode
        return result

    def evaluate(self) -> dict[str, float]:
        """Evaluate the deterministic policy in the independent evaluation environment."""
        return self._evaluate()

    def _evaluate(self) -> dict[str, float]:
        results = [
            self.run_episode(
                training=False,
                episode=(self.current_episode * self.evaluation_episodes) + index,
            )
            for index in range(self.evaluation_episodes)
        ]
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
            "config": self.config,
            "current_episode": self.current_episode,
            "global_step": self.global_step,
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
            "algorithm_state": self.algorithm_state_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }
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
        for name, state in checkpoint.get("models_state_dict", {}).items():
            if name not in self.models:
                raise KeyError(f"Checkpoint contains unknown model: {name}")
            self.accelerator.unwrap_model(self.models[name]).load_state_dict(state)
        for name, state in checkpoint.get("optimizers_state_dict", {}).items():
            if name not in self.optimizers:
                raise KeyError(f"Checkpoint contains unknown optimizer: {name}")
            self.optimizers[name].load_state_dict(state)
        for name, state in checkpoint.get("schedulers_state_dict", {}).items():
            if name not in self.schedulers:
                raise KeyError(f"Checkpoint contains unknown scheduler: {name}")
            self.schedulers[name].load_state_dict(state)

        self.current_episode = int(checkpoint.get("current_episode", 0))
        self.current_epoch = self.current_episode
        self.global_step = int(checkpoint.get("global_step", 0))
        self.update_step = int(checkpoint.get("update_step", 0))
        self.episode_metrics = list(checkpoint.get("episode_metrics", []))
        self.evaluation_metrics = list(checkpoint.get("evaluation_metrics", []))
        self.load_algorithm_state_dict(checkpoint.get("algorithm_state", {}))

        callback_states = checkpoint.get("callback_states", {})
        for index, callback in enumerate(self.callbacks.callbacks):
            key = f"callback_{index}_{callback.__class__.__name__}"
            if key in callback_states:
                callback.set_state(callback_states[key])
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    def close(self) -> None:
        """Close environments owned by this trainer."""
        self._close()

    def _close(self) -> None:
        for attribute in ("environment", "evaluation_environment"):
            environment = getattr(self, attribute, None)
            if environment is not None:
                environment.close()

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

    @abstractmethod
    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return algorithm-specific state not held by models or optimizers."""

    @abstractmethod
    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific state."""
