"""DreamerV3-inspired trainer for discrete-action environments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.nn import functional

from dl_core.core import (
    ActionOutput,
    BatchActionOutput,
    DreamerWorldModelProtocol,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    RLTrainer,
    SequenceBatch,
    SequenceReplayBuffer,
    Transition,
    TransitionBatch,
    WorldModelState,
    config_field,
    register_trainer,
)
from dl_core.models import DreamerWorldModel


@dataclass(slots=True)
class DreamerPolicyState:
    """Latent policy state carried between environment observations."""

    world_state: WorldModelState
    previous_actions: torch.Tensor
    is_first: torch.Tensor


@dataclass(slots=True)
class ImaginedTrajectory:
    """Policy quantities and model predictions from latent imagination."""

    features: torch.Tensor
    log_probabilities: torch.Tensor
    entropies: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    next_values: torch.Tensor


@register_trainer("dreamer")
class DreamerTrainer(RLTrainer):
    """Learn a categorical world model and policy through latent imagination."""

    CONFIG_FIELDS = RLTrainer.CONFIG_FIELDS + [
        config_field(
            "gamma",
            "float",
            "Reward discount factor.",
            default=0.997,
        ),
        config_field(
            "lambda_",
            "float",
            "Lambda-return trace parameter.",
            default=0.95,
        ),
        config_field(
            "buffer_size",
            "int",
            "Maximum transitions retained across sequence-replay lanes.",
            default=100000,
        ),
        config_field(
            "batch_size",
            "int",
            "Sequences sampled per model update.",
            default=16,
        ),
        config_field(
            "sequence_length",
            "int",
            "Learning transitions in each sampled sequence.",
            default=50,
        ),
        config_field(
            "burn_in",
            "int",
            "Context transitions preceding each learning sequence.",
            default=0,
        ),
        config_field(
            "learning_starts",
            "int",
            "Transitions collected before gradient updates begin.",
            default=1000,
        ),
        config_field(
            "train_frequency",
            "int",
            "Environment transitions between update cycles.",
            default=1,
        ),
        config_field(
            "gradient_steps",
            "int",
            "Sequence gradient steps in each update cycle.",
            default=1,
        ),
        config_field(
            "imagination_horizon",
            "int",
            "Latent transitions used for actor and critic updates.",
            default=15,
        ),
        config_field(
            "free_nats",
            "float",
            "Minimum categorical KL contribution per latent state.",
            default=1.0,
        ),
        config_field(
            "dynamics_kl_weight",
            "float",
            "Weight of posterior-stop-gradient dynamics KL.",
            default=0.5,
        ),
        config_field(
            "representation_kl_weight",
            "float",
            "Weight of prior-stop-gradient representation KL.",
            default=0.1,
        ),
        config_field(
            "reconstruction_weight",
            "float",
            "Observation reconstruction-loss weight.",
            default=1.0,
        ),
        config_field(
            "reward_weight",
            "float",
            "Symlog reward-regression loss weight.",
            default=1.0,
        ),
        config_field(
            "continuation_weight",
            "float",
            "Episode-continuation prediction loss weight.",
            default=1.0,
        ),
        config_field(
            "actor_entropy_weight",
            "float",
            "Categorical policy entropy bonus weight.",
            default=3e-4,
        ),
        config_field(
            "actor_unimix",
            "float",
            "Uniform mixture added to categorical action probabilities.",
            default=0.01,
        ),
        config_field(
            "normalize_advantages",
            "bool",
            "Standardize imagined policy advantages within an update.",
            default=True,
        ),
        config_field(
            "target_critic_tau",
            "float",
            "Soft-update coefficient for the target critic.",
            default=0.02,
        ),
        config_field(
            "checkpoint_replay_buffer",
            "bool",
            "Include sequence replay in resumable checkpoints.",
            default=True,
        ),
    ]

    def setup_algorithm(self) -> None:
        """Create Dreamer models, optimizers, sequence replay, and schedules."""
        if not isinstance(self.environment.action_space, Discrete):
            raise TypeError("DreamerTrainer requires a Discrete action space")
        if not isinstance(self.environment.observation_space, (Box, Discrete)):
            raise TypeError(
                "DreamerTrainer requires a Box or Discrete observation space"
            )
        if self.evaluation_environment.action_space != self.environment.action_space:
            raise ValueError("Training and evaluation action spaces must match")
        if (
            self.evaluation_environment.observation_space
            != self.environment.observation_space
        ):
            raise ValueError("Training and evaluation observation spaces must match")

        observation_space = self.environment.observation_space
        if isinstance(observation_space, Discrete):
            observation_shape: tuple[int, ...] = ()
            observation_dtype = np.int64
            input_dim = int(observation_space.n)
        else:
            observation_shape = observation_space.shape
            observation_dtype = observation_space.dtype
            input_dim = int(np.prod(observation_shape))

        self.gamma = float(self.trainer_config.get("gamma", 0.997))
        self.lambda_ = float(self.trainer_config.get("lambda_", 0.95))
        self.buffer_size = int(
            self.trainer_config.get("buffer_size", 100000)
        )
        self.batch_size = int(self.trainer_config.get("batch_size", 16))
        self.sequence_length = int(
            self.trainer_config.get("sequence_length", 50)
        )
        self.burn_in = int(self.trainer_config.get("burn_in", 0))
        self.learning_starts = int(
            self.trainer_config.get("learning_starts", 1000)
        )
        self.train_frequency = int(
            self.trainer_config.get("train_frequency", 1)
        )
        self.gradient_steps = int(
            self.trainer_config.get("gradient_steps", 1)
        )
        self.imagination_horizon = int(
            self.trainer_config.get("imagination_horizon", 15)
        )
        self.free_nats = float(self.trainer_config.get("free_nats", 1.0))
        self.dynamics_kl_weight = float(
            self.trainer_config.get("dynamics_kl_weight", 0.5)
        )
        self.representation_kl_weight = float(
            self.trainer_config.get("representation_kl_weight", 0.1)
        )
        self.reconstruction_weight = float(
            self.trainer_config.get("reconstruction_weight", 1.0)
        )
        self.reward_weight = float(
            self.trainer_config.get("reward_weight", 1.0)
        )
        self.continuation_weight = float(
            self.trainer_config.get("continuation_weight", 1.0)
        )
        self.actor_entropy_weight = float(
            self.trainer_config.get("actor_entropy_weight", 3e-4)
        )
        self.actor_unimix = float(
            self.trainer_config.get("actor_unimix", 0.01)
        )
        self.normalize_advantages = bool(
            self.trainer_config.get("normalize_advantages", True)
        )
        self.target_critic_tau = float(
            self.trainer_config.get("target_critic_tau", 0.02)
        )
        self.checkpoint_replay_buffer = bool(
            self.trainer_config.get("checkpoint_replay_buffer", True)
        )
        self.overlap_environment_steps = False

        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not np.isfinite(self.lambda_) or not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda_ must be finite and in [0, 1]")
        if min(
            self.buffer_size,
            self.batch_size,
            self.sequence_length,
            self.train_frequency,
            self.gradient_steps,
            self.imagination_horizon,
        ) <= 0:
            raise ValueError("Dreamer replay and update sizes must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in cannot be negative")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        loss_parameters = (
            self.free_nats,
            self.dynamics_kl_weight,
            self.representation_kl_weight,
            self.reconstruction_weight,
            self.reward_weight,
            self.continuation_weight,
            self.actor_entropy_weight,
        )
        if not all(np.isfinite(value) for value in loss_parameters) or any(
            value < 0.0 for value in loss_parameters
        ):
            raise ValueError(
                "Dreamer loss weights must be finite and nonnegative"
            )
        if (
            not np.isfinite(self.actor_unimix)
            or not 0.0 <= self.actor_unimix < 1.0
        ):
            raise ValueError("actor_unimix must be finite and in [0, 1)")
        if (
            not np.isfinite(self.target_critic_tau)
            or not 0.0 < self.target_critic_tau <= 1.0
        ):
            raise ValueError(
                "target_critic_tau must be finite and in (0, 1]"
            )
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError(
                "DreamerTrainer requires gradient_accumulation_steps=1"
            )

        model_section = self.config.get("models", {})
        if not isinstance(model_section, dict):
            raise TypeError("models must be a mapping")
        world_model_config = model_section.get("world_model", {})
        actor_config = model_section.get("actor", {})
        critic_config = model_section.get("critic", {})
        if not isinstance(world_model_config, dict):
            raise TypeError("models.world_model must be a mapping")
        if not isinstance(actor_config, dict):
            raise TypeError("models.actor must be a mapping")
        if not isinstance(critic_config, dict):
            raise TypeError("models.critic must be a mapping")

        world_model_config = dict(world_model_config)
        world_model_name = str(
            world_model_config.pop("name", "dreamer_world_model")
        )
        world_model_config["input_dim"] = input_dim
        world_model_config["action_dim"] = int(
            self.environment.action_space.n
        )
        world_model = MODEL_REGISTRY.get(
            world_model_name,
            world_model_config,
        )
        if not isinstance(world_model, DreamerWorldModel):
            raise TypeError(
                "DreamerTrainer currently requires DreamerWorldModel"
            )
        self.models["world_model"] = world_model

        actor_config = dict(actor_config)
        actor_name = str(actor_config.pop("name", "dreamer_actor"))
        actor_config["feature_dim"] = world_model.feature_size
        actor_config["action_dim"] = int(self.environment.action_space.n)
        self.models["actor"] = MODEL_REGISTRY.get(actor_name, actor_config)

        critic_config = dict(critic_config)
        critic_name = str(critic_config.pop("name", "dreamer_critic"))
        critic_config["feature_dim"] = world_model.feature_size
        self.models["critic"] = MODEL_REGISTRY.get(
            critic_name,
            critic_config,
        )
        self.models["target_critic"] = copy.deepcopy(self.models["critic"])
        self.models["target_critic"].eval()
        for parameter in self.models["target_critic"].parameters():
            parameter.requires_grad_(False)

        optimizer_section = self.config.get("optimizers", {})
        if not isinstance(optimizer_section, dict):
            raise TypeError("optimizers must be a mapping")
        default_optimizers = {
            "world_model": {"name": "adam", "lr": 1e-4},
            "actor": {"name": "adam", "lr": 3e-5},
            "critic": {"name": "adam", "lr": 3e-5},
        }
        for component_name in ("world_model", "actor", "critic"):
            optimizer_config = optimizer_section.get(
                component_name,
                default_optimizers[component_name],
            )
            if not isinstance(optimizer_config, dict):
                raise TypeError(
                    f"optimizers.{component_name} must be a mapping"
                )
            optimizer_config = dict(optimizer_config)
            optimizer_name = str(optimizer_config.pop("name", "adam"))
            parameters = [
                parameter
                for parameter in self.models[component_name].parameters()
                if parameter.requires_grad
            ]
            if not parameters:
                raise ValueError(
                    f"Dreamer {component_name} has no trainable parameters"
                )
            self.optimizers[component_name] = OPTIMIZER_REGISTRY.get(
                optimizer_name,
                parameters,
                **optimizer_config,
            )

        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.buffer_size,
            num_environments=self.environment.num_envs,
            observation_shape=observation_shape,
            action_shape=(),
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            observation_dtype=observation_dtype,
            action_dtype=np.int64,
            seed=self.seed,
        )

    def initialize_policy_state(
        self,
        batch_size: int,
        *,
        evaluation: bool,
    ) -> DreamerPolicyState:
        """Create zero latent state for new environment lanes."""
        return self._initialize_policy_state(
            batch_size,
            evaluation=evaluation,
        )

    def _initialize_policy_state(
        self,
        batch_size: int,
        *,
        evaluation: bool,
    ) -> DreamerPolicyState:
        if batch_size <= 0:
            raise ValueError("Policy-state batch size must be positive")
        del evaluation
        device = self.accelerator.get_device()
        world_model: DreamerWorldModelProtocol = self.accelerator.unwrap_model(
            self.models["world_model"]
        )
        if not isinstance(world_model, DreamerWorldModel):
            raise TypeError("Dreamer world model contract is invalid")
        return DreamerPolicyState(
            world_state=world_model.initial_state(
                batch_size,
                device=device,
            ),
            previous_actions=torch.zeros(
                batch_size,
                self.environment.action_space.n,
                device=device,
            ),
            is_first=torch.ones(
                batch_size,
                dtype=torch.bool,
                device=device,
            ),
        )

    def reset_policy_state(
        self,
        policy_state: DreamerPolicyState,
        done: np.ndarray,
        *,
        evaluation: bool,
    ) -> DreamerPolicyState:
        """Reset only latent-state lanes whose episodes completed."""
        return self._reset_policy_state(
            policy_state,
            done,
            evaluation=evaluation,
        )

    def _reset_policy_state(
        self,
        policy_state: DreamerPolicyState,
        done: np.ndarray,
        *,
        evaluation: bool,
    ) -> DreamerPolicyState:
        if not isinstance(policy_state, DreamerPolicyState):
            raise TypeError("Dreamer policy state is invalid")
        del evaluation
        done_tensor = torch.as_tensor(
            np.asarray(done, dtype=np.bool_),
            device=policy_state.is_first.device,
        )
        if done_tensor.shape != policy_state.is_first.shape:
            raise ValueError("Policy-state done mask shape is invalid")
        keep = (~done_tensor).to(
            policy_state.world_state.deterministic.dtype
        ).unsqueeze(-1)
        return DreamerPolicyState(
            world_state=WorldModelState(
                deterministic=(
                    policy_state.world_state.deterministic * keep
                ),
                stochastic=(
                    policy_state.world_state.stochastic
                    * keep.unsqueeze(-1)
                ),
                logits=(
                    policy_state.world_state.logits * keep.unsqueeze(-1)
                ),
            ),
            previous_actions=policy_state.previous_actions * keep,
            is_first=torch.logical_or(policy_state.is_first, done_tensor),
        )

    def build_action_distribution(
        self,
        logits: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Build the categorical policy distribution used in acting and learning."""
        return self._build_action_distribution(logits)

    def _build_action_distribution(
        self,
        logits: torch.Tensor,
    ) -> torch.distributions.Categorical:
        if logits.shape[-1] != self.environment.action_space.n:
            raise ValueError("Dreamer actor action dimension is invalid")
        log_probabilities = functional.log_softmax(logits, dim=-1)
        if self.actor_unimix > 0.0:
            log_probabilities = torch.logaddexp(
                log_probabilities + math.log1p(-self.actor_unimix),
                torch.full_like(
                    log_probabilities,
                    math.log(
                        self.actor_unimix
                        / self.environment.action_space.n
                    ),
                ),
            )
        return torch.distributions.Categorical(logits=log_probabilities)

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
    ) -> ActionOutput[int]:
        """Select one action from a newly initialized latent state."""
        return self._select_action(
            observation,
            deterministic=deterministic,
        )

    def _select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
    ) -> ActionOutput[int]:
        state = self.initialize_policy_state(1, evaluation=deterministic)
        return self._select_action_with_state(
            observation,
            state,
            deterministic=deterministic,
        )

    def select_action_with_state(
        self,
        observation: Any,
        policy_state: DreamerPolicyState,
        *,
        deterministic: bool,
    ) -> ActionOutput[int]:
        """Select one action and return its updated latent policy state."""
        return self._select_action_with_state(
            observation,
            policy_state,
            deterministic=deterministic,
        )

    def _select_action_with_state(
        self,
        observation: Any,
        policy_state: DreamerPolicyState,
        *,
        deterministic: bool,
    ) -> ActionOutput[int]:
        output = self._select_actions_with_state(
            np.expand_dims(np.asarray(observation), axis=0),
            policy_state,
            deterministic=deterministic,
        )
        return ActionOutput(
            action=output.actions[0],
            info=output.action_info[0],
            policy_state=output.policy_state,
        )

    def select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        """Select batched actions from newly initialized latent states."""
        return self._select_actions(
            observations,
            deterministic=deterministic,
        )

    def _select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        observation_batch = np.asarray(observations)
        state = self.initialize_policy_state(
            int(observation_batch.shape[0]),
            evaluation=deterministic,
        )
        return self._select_actions_with_state(
            observation_batch,
            state,
            deterministic=deterministic,
        )

    def select_actions_with_state(
        self,
        observations: Any,
        policy_state: DreamerPolicyState,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        """Infer posterior state and select one action per environment lane."""
        return self._select_actions_with_state(
            observations,
            policy_state,
            deterministic=deterministic,
        )

    def _select_actions_with_state(
        self,
        observations: Any,
        policy_state: DreamerPolicyState,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        if not isinstance(policy_state, DreamerPolicyState):
            raise TypeError("Dreamer policy state is invalid")
        observation_batch = np.asarray(observations)
        if observation_batch.shape[0] == 0:
            raise ValueError(
                "Dreamer action selection requires at least one observation"
            )
        if observation_batch.shape[0] != policy_state.is_first.shape[0]:
            raise ValueError(
                "Observation batch and Dreamer policy state do not align"
            )
        if any(
            not self.environment.observation_space.contains(observation)
            for observation in observation_batch
        ):
            raise ValueError("Observation is outside the configured space")

        was_training = {
            name: self.models[name].training
            for name in ("world_model", "actor")
        }
        try:
            self.models["world_model"].eval()
            self.models["actor"].eval()
            with torch.no_grad(), self.accelerator.autocast_context():
                observations_tensor = self.transform_observations(
                    observation_batch
                )
                world_model = self.models["world_model"]
                if not isinstance(
                    self.accelerator.unwrap_model(world_model),
                    DreamerWorldModel,
                ):
                    raise TypeError("Dreamer world model contract is invalid")
                embeddings = world_model.encode(observations_tensor)
                world_step = world_model.observe_step(
                    policy_state.world_state,
                    policy_state.previous_actions,
                    embeddings,
                    policy_state.is_first,
                    deterministic=deterministic,
                )
                features = world_model.features(world_step.state)
                logits = self.models["actor"](features)
                distribution = self.build_action_distribution(logits)
                if deterministic:
                    action_indices = distribution.logits.argmax(dim=-1)
                else:
                    action_indices = distribution.sample()
                entropy = distribution.entropy()
                next_state = DreamerPolicyState(
                    world_state=WorldModelState(
                        deterministic=world_step.state.deterministic.detach(),
                        stochastic=world_step.state.stochastic.detach(),
                        logits=world_step.state.logits.detach(),
                    ),
                    previous_actions=functional.one_hot(
                        action_indices,
                        self.environment.action_space.n,
                    ).float(),
                    is_first=torch.zeros_like(policy_state.is_first),
                )
        finally:
            for name, training in was_training.items():
                self.models[name].train(training)

        action_start = int(self.environment.action_space.start)
        return BatchActionOutput(
            actions=[
                int(action) + action_start
                for action in action_indices.detach().cpu().tolist()
            ],
            action_info=[
                {"policy_entropy": float(value)}
                for value in entropy.detach().cpu().tolist()
            ],
            policy_state=next_state,
        )

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float] | None:
        """Store one transition and run scheduled Dreamer updates."""
        logs = self._process_transition_batch(
            TransitionBatch(
                observations=np.expand_dims(
                    np.asarray(transition.observation),
                    axis=0,
                ),
                actions=np.asarray([transition.action]),
                rewards=np.asarray([transition.reward], dtype=np.float32),
                next_observations=np.expand_dims(
                    np.asarray(transition.next_observation),
                    axis=0,
                ),
                terminated=np.asarray(
                    [transition.terminated],
                    dtype=np.bool_,
                ),
                truncated=np.asarray(
                    [transition.truncated],
                    dtype=np.bool_,
                ),
                infos=[transition.info],
                action_info=[transition.action_info],
            )
        )
        return logs[-1] if logs else None

    def process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        """Insert a vector step and run every crossed Dreamer update cycle."""
        return self._process_transition_batch(transitions)

    def _process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if any(
            not observation_space.contains(observation)
            for observation in np.asarray(transitions.observations)
        ):
            raise ValueError(
                "Transition observation is outside the configured space"
            )
        if any(
            not observation_space.contains(observation)
            for observation in np.asarray(transitions.next_observations)
        ):
            raise ValueError(
                "Transition next observation is outside the configured space"
            )
        if any(
            not action_space.contains(action)
            for action in np.asarray(transitions.actions)
        ):
            raise ValueError("Transition action is outside the configured space")
        if not np.isfinite(transitions.rewards).all():
            raise ValueError("Transition reward must be finite")

        previous_global_step = self.global_step - transitions.size
        replay_add_start = perf_counter()
        replay_add_result = self.replay_buffer.add_batch(transitions)
        replay_add_ms = (perf_counter() - replay_add_start) * 1000.0
        if (
            self.global_step < self.learning_starts
            or self.replay_buffer.num_sequences == 0
        ):
            return []

        replay_ready_step = (
            previous_global_step
            + int(
                np.flatnonzero(
                    replay_add_result.available_after_environment > 0
                )[0]
            )
            + 1
        )
        first_ready_step = max(
            previous_global_step + 1,
            self.learning_starts,
            replay_ready_step,
        )
        first_update_step = (
            (first_ready_step + self.train_frequency - 1)
            // self.train_frequency
            * self.train_frequency
        )
        update_logs: list[dict[str, float]] = []
        for scheduled_step in range(
            first_update_step,
            self.global_step + 1,
            self.train_frequency,
        ):
            if not self.should_update(scheduled_step, transitions):
                continue
            gradient_logs: list[dict[str, float]] = []
            replay_sample_ms = 0.0
            model_update_ms = 0.0
            sample_ages: list[float] = []
            for _ in range(self.gradient_steps):
                replay_sample_start = perf_counter()
                sequences = self.replay_buffer.sample(
                    self.batch_size,
                    self.accelerator.get_device(),
                )
                replay_sample_ms += (
                    perf_counter() - replay_sample_start
                ) * 1000.0
                model_update_start = perf_counter()
                gradient_logs.append(self.update_model(sequences))
                model_update_ms += (
                    perf_counter() - model_update_start
                ) * 1000.0
                sample_ages.append(
                    float(sequences.sample_ages.float().mean().item())
                )
            logs = {
                name: float(
                    np.mean(
                        [
                            gradient_log[name]
                            for gradient_log in gradient_logs
                        ]
                    )
                )
                for name in gradient_logs[0]
            }
            logs.update(
                {
                    "dreamer/replay_size": float(len(self.replay_buffer)),
                    "dreamer/replay_sequences": float(
                        self.replay_buffer.num_sequences
                    ),
                    "dreamer/replay_sample_age": float(
                        np.mean(sample_ages)
                    ),
                    "dreamer/timing/replay_add_ms": replay_add_ms,
                    "dreamer/timing/replay_sample_ms": replay_sample_ms,
                    "dreamer/timing/model_update_ms": model_update_ms,
                }
            )
            update_logs.append(logs)
        return update_logs

    def update_model(
        self,
        sequences: SequenceBatch,
    ) -> dict[str, float]:
        """Update world model, actor, and critic from replay sequences."""
        return self._update_model(sequences)

    def _update_model(
        self,
        sequences: SequenceBatch,
    ) -> dict[str, float]:
        world_model = self.models["world_model"]
        actor = self.models["actor"]
        critic = self.models["critic"]
        target_critic = self.models["target_critic"]
        if not isinstance(
            self.accelerator.unwrap_model(world_model),
            DreamerWorldModel,
        ):
            raise TypeError("Dreamer world model contract is invalid")
        if sequences.burn_in != self.burn_in:
            raise ValueError("Replay sequence burn-in does not match trainer")

        observations = self.transform_observations(
            sequences.observations
        )
        action_indices = (
            sequences.actions.long()
            - int(self.environment.action_space.start)
        )
        if (
            action_indices.min().item() < 0
            or action_indices.max().item()
            >= self.environment.action_space.n
        ):
            raise ValueError("Replay sequence contains an invalid action")

        with self.accelerator.autocast_context():
            world_output = world_model(
                observations,
                action_indices,
                sequences.is_first,
            )
            transition_slice = slice(self.burn_in, None)
            observation_slice = slice(self.burn_in + 1, None)
            reconstruction_loss = functional.mse_loss(
                world_output.reconstructions[:, observation_slice],
                world_output.observation_targets[:, observation_slice],
            )
            reward_targets = sequences.rewards[:, transition_slice]
            reward_targets = torch.sign(reward_targets) * torch.log1p(
                reward_targets.abs()
            )
            reward_loss = functional.mse_loss(
                world_output.reward_predictions[:, transition_slice],
                reward_targets,
            )
            continuation_targets = (
                ~sequences.terminated[:, transition_slice]
            ).float()
            continuation_loss = functional.binary_cross_entropy_with_logits(
                world_output.continue_logits[:, transition_slice],
                continuation_targets,
            )

            posterior_logits = world_output.states.logits[
                :, observation_slice
            ]
            prior_logits = world_output.prior_logits[:, observation_slice]
            dynamics_kl = torch.distributions.kl_divergence(
                torch.distributions.Categorical(
                    logits=posterior_logits.detach()
                ),
                torch.distributions.Categorical(logits=prior_logits),
            )
            dynamics_kl = (
                dynamics_kl.sum(dim=-1)
                .clamp_min(self.free_nats)
                .mean()
            )
            representation_kl = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=posterior_logits),
                torch.distributions.Categorical(
                    logits=prior_logits.detach()
                ),
            )
            representation_kl = (
                representation_kl.sum(dim=-1)
                .clamp_min(self.free_nats)
                .mean()
            )
            world_model_loss = (
                self.reconstruction_weight * reconstruction_loss
                + self.reward_weight * reward_loss
                + self.continuation_weight * continuation_loss
                + self.dynamics_kl_weight * dynamics_kl
                + self.representation_kl_weight * representation_kl
            )
            if not torch.isfinite(world_model_loss):
                raise FloatingPointError(
                    "Dreamer world-model loss must be finite"
                )

        self.optimizers["world_model"].zero_grad(set_to_none=True)
        self.accelerator.backward(world_model_loss, world_model)
        self.accelerator.optimizer_step(
            self.optimizers["world_model"],
            world_model,
        )

        start_state = WorldModelState(
            deterministic=world_output.states.deterministic[
                :, self.burn_in : -1
            ].reshape(-1, world_output.states.deterministic.shape[-1]).detach(),
            stochastic=world_output.states.stochastic[
                :, self.burn_in : -1
            ].reshape(
                -1,
                world_output.states.stochastic.shape[-2],
                world_output.states.stochastic.shape[-1],
            ).detach(),
            logits=world_output.states.logits[
                :, self.burn_in : -1
            ].reshape(
                -1,
                world_output.states.logits.shape[-2],
                world_output.states.logits.shape[-1],
            ).detach(),
        )
        imagined_features: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        imagined_rewards: list[torch.Tensor] = []
        imagined_continues: list[torch.Tensor] = []
        imagined_next_values: list[torch.Tensor] = []
        imagined_state = start_state
        for _ in range(self.imagination_horizon):
            features = world_model.features(imagined_state).detach()
            with self.accelerator.autocast_context():
                action_logits = actor(features)
                distribution = self.build_action_distribution(action_logits)
                action_indices = distribution.sample()
            imagined_features.append(features)
            log_probabilities.append(
                distribution.log_prob(action_indices)
            )
            entropies.append(distribution.entropy())
            imagined_actions = functional.one_hot(
                action_indices,
                self.environment.action_space.n,
            ).float()
            with torch.no_grad(), self.accelerator.autocast_context():
                imagined_state = world_model.imagine_step(
                    imagined_state,
                    imagined_actions,
                )
                next_features = world_model.features(imagined_state)
                predicted_rewards = world_model.predict_rewards(
                    next_features
                )
                predicted_rewards = (
                    torch.sign(predicted_rewards)
                    * torch.expm1(predicted_rewards.abs())
                )
                predicted_continues = torch.sigmoid(
                    world_model.predict_continue_logits(next_features)
                )
                next_values = target_critic(next_features)
                next_values = (
                    torch.sign(next_values)
                    * torch.expm1(next_values.abs())
                )
            imagined_state = WorldModelState(
                deterministic=imagined_state.deterministic.detach(),
                stochastic=imagined_state.stochastic.detach(),
                logits=imagined_state.logits.detach(),
            )
            imagined_rewards.append(predicted_rewards)
            imagined_continues.append(predicted_continues)
            imagined_next_values.append(next_values)

        trajectory = ImaginedTrajectory(
            features=torch.stack(imagined_features, dim=1),
            log_probabilities=torch.stack(log_probabilities, dim=1),
            entropies=torch.stack(entropies, dim=1),
            rewards=torch.stack(imagined_rewards, dim=1),
            continues=torch.stack(imagined_continues, dim=1),
            next_values=torch.stack(imagined_next_values, dim=1),
        )
        discounts = self.gamma * trajectory.continues
        returns = self.compute_lambda_returns(
            trajectory.rewards,
            discounts,
            trajectory.next_values,
        )
        weights = torch.cumprod(
            torch.cat(
                (
                    torch.ones_like(discounts[:, :1]),
                    discounts[:, :-1],
                ),
                dim=1,
            ),
            dim=1,
        ).detach()
        if (
            not torch.isfinite(returns).all()
            or not torch.isfinite(weights).all()
        ):
            raise FloatingPointError(
                "Dreamer imagined returns and weights must be finite"
            )

        with torch.no_grad(), self.accelerator.autocast_context():
            baseline = target_critic(trajectory.features)
            baseline = torch.sign(baseline) * torch.expm1(baseline.abs())
            advantages = returns - baseline
            if self.normalize_advantages:
                advantages = (
                    advantages - advantages.mean()
                ) / advantages.std(unbiased=False).clamp_min(1e-8)
        actor_loss = -(
            weights
            * (
                trajectory.log_probabilities * advantages.detach()
                + self.actor_entropy_weight * trajectory.entropies
            )
        ).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Dreamer actor loss must be finite")
        self.optimizers["actor"].zero_grad(set_to_none=True)
        self.accelerator.backward(actor_loss, actor)
        self.accelerator.optimizer_step(
            self.optimizers["actor"],
            actor,
        )

        with self.accelerator.autocast_context():
            value_predictions = critic(trajectory.features.detach())
            value_targets = torch.sign(returns.detach()) * torch.log1p(
                returns.detach().abs()
            )
            critic_loss = (
                weights
                * functional.mse_loss(
                    value_predictions,
                    value_targets,
                    reduction="none",
                )
            ).mean()
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("Dreamer critic loss must be finite")
        self.optimizers["critic"].zero_grad(set_to_none=True)
        self.accelerator.backward(critic_loss, critic)
        self.accelerator.optimizer_step(
            self.optimizers["critic"],
            critic,
        )

        with torch.no_grad():
            critic_parameters = self.accelerator.unwrap_model(
                critic
            ).parameters()
            target_parameters = self.accelerator.unwrap_model(
                target_critic
            ).parameters()
            for target_parameter, critic_parameter in zip(
                target_parameters,
                critic_parameters,
                strict=True,
            ):
                target_parameter.lerp_(
                    critic_parameter,
                    self.target_critic_tau,
                )

        return {
            "dreamer/world_model_loss": float(
                world_model_loss.detach().item()
            ),
            "dreamer/reconstruction_loss": float(
                reconstruction_loss.detach().item()
            ),
            "dreamer/reward_loss": float(reward_loss.detach().item()),
            "dreamer/continuation_loss": float(
                continuation_loss.detach().item()
            ),
            "dreamer/dynamics_kl": float(dynamics_kl.detach().item()),
            "dreamer/representation_kl": float(
                representation_kl.detach().item()
            ),
            "dreamer/actor_loss": float(actor_loss.detach().item()),
            "dreamer/critic_loss": float(critic_loss.detach().item()),
            "dreamer/policy_entropy": float(
                trajectory.entropies.detach().mean().item()
            ),
            "dreamer/imagined_return": float(
                returns.detach().mean().item()
            ),
            "dreamer/imagined_continue": float(
                trajectory.continues.detach().mean().item()
            ),
            "dreamer/latent_deterministic_std": float(
                world_output.states.deterministic.detach().std().item()
            ),
            "dreamer/latent_stochastic_mean": float(
                world_output.states.stochastic.detach().mean().item()
            ),
        }

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute backward-view lambda returns for imagined transitions."""
        return self._compute_lambda_returns(
            rewards,
            discounts,
            next_values,
        )

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        if rewards.ndim != 2:
            raise ValueError("Lambda-return tensors must be two-dimensional")
        if discounts.shape != rewards.shape:
            raise ValueError("Lambda-return discounts must match rewards")
        if next_values.shape != rewards.shape:
            raise ValueError("Lambda-return values must match rewards")
        accumulated_return = next_values[:, -1]
        returns: list[torch.Tensor] = []
        for step in range(rewards.shape[1] - 1, -1, -1):
            bootstrap = (
                (1.0 - self.lambda_) * next_values[:, step]
                + self.lambda_ * accumulated_return
            )
            accumulated_return = (
                rewards[:, step] + discounts[:, step] * bootstrap
            )
            returns.append(accumulated_return)
        return torch.stack(list(reversed(returns)), dim=1)

    def transform_observations(
        self,
        observations: Any,
    ) -> torch.Tensor:
        """Convert environment observations to model-ready float tensors."""
        return self._transform_observations(observations)

    def _transform_observations(
        self,
        observations: Any,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            observations,
            device=self.accelerator.get_device(),
        )
        if isinstance(self.environment.observation_space, Discrete):
            indices = (
                tensor.long()
                - int(self.environment.observation_space.start)
            )
            return functional.one_hot(
                indices,
                num_classes=self.environment.observation_space.n,
            ).float()
        observation_rank = len(self.environment.observation_space.shape)
        if tensor.ndim == observation_rank + 1:
            return tensor.float().reshape(tensor.shape[0], -1)
        if tensor.ndim == observation_rank + 2:
            return tensor.float().reshape(*tensor.shape[:2], -1)
        raise ValueError(
            "Dreamer observations must contain batch or batch-time dimensions"
        )

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return replay and environment-contract state for checkpoints."""
        return {
            "replay_buffer": (
                self.replay_buffer.state_dict()
                if self.checkpoint_replay_buffer
                else None
            ),
            "checkpoint_replay_buffer": self.checkpoint_replay_buffer,
            "training_parameters": {
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "sequence_length": self.sequence_length,
                "burn_in": self.burn_in,
                "learning_starts": self.learning_starts,
                "train_frequency": self.train_frequency,
                "gradient_steps": self.gradient_steps,
                "imagination_horizon": self.imagination_horizon,
                "free_nats": self.free_nats,
                "dynamics_kl_weight": self.dynamics_kl_weight,
                "representation_kl_weight": (
                    self.representation_kl_weight
                ),
                "reconstruction_weight": self.reconstruction_weight,
                "reward_weight": self.reward_weight,
                "continuation_weight": self.continuation_weight,
                "actor_entropy_weight": self.actor_entropy_weight,
                "actor_unimix": self.actor_unimix,
                "normalize_advantages": self.normalize_advantages,
                "target_critic_tau": self.target_critic_tau,
            },
            "observation_space": repr(self.environment.observation_space),
            "action_space": repr(self.environment.action_space),
        }

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay state after validating its training contract."""
        if not isinstance(state, dict):
            raise TypeError("Checkpoint algorithm state must be a mapping")
        if state.get("observation_space") != repr(
            self.environment.observation_space
        ):
            raise ValueError("Checkpoint observation space does not match")
        if state.get("action_space") != repr(
            self.environment.action_space
        ):
            raise ValueError("Checkpoint action space does not match")
        expected_training_parameters = {
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "burn_in": self.burn_in,
            "learning_starts": self.learning_starts,
            "train_frequency": self.train_frequency,
            "gradient_steps": self.gradient_steps,
            "imagination_horizon": self.imagination_horizon,
            "free_nats": self.free_nats,
            "dynamics_kl_weight": self.dynamics_kl_weight,
            "representation_kl_weight": self.representation_kl_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "reward_weight": self.reward_weight,
            "continuation_weight": self.continuation_weight,
            "actor_entropy_weight": self.actor_entropy_weight,
            "actor_unimix": self.actor_unimix,
            "normalize_advantages": self.normalize_advantages,
            "target_critic_tau": self.target_critic_tau,
        }
        if state.get("training_parameters") != expected_training_parameters:
            raise ValueError(
                "Checkpoint Dreamer training parameters do not match config"
            )
        if (
            state.get("checkpoint_replay_buffer")
            is not self.checkpoint_replay_buffer
        ):
            raise ValueError(
                "Checkpoint replay-buffer policy does not match config"
            )
        if self.checkpoint_replay_buffer:
            replay_state = state.get("replay_buffer")
            if not isinstance(replay_state, dict):
                raise ValueError("Checkpoint replay buffer state is missing")
            self.replay_buffer.load_state_dict(replay_state)
