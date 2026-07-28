"""Soft actor-critic trainer for bounded continuous control."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as functional

from dl_core.core import (
    BatchActionOutput,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    RLTrainer,
    ReplayBuffer,
    Transition,
    TransitionBatch,
    config_field,
    register_trainer,
)


class _EntropyTemperature(nn.Module):
    """Positive entropy temperature represented in log space."""

    def __init__(self, initial_alpha: float) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha)))

    def forward(self) -> torch.Tensor:
        return self._forward()

    def _forward(self) -> torch.Tensor:
        return self.log_alpha.exp()


@register_trainer("sac")
class SACTrainer(RLTrainer):
    """Replay-based SAC with twin critics and optional entropy tuning."""

    REQUIRED_CONFIG_SECTIONS = ("environment", "models")

    CONFIG_FIELDS = RLTrainer.CONFIG_FIELDS + [
        config_field("gamma", "float", "Reward discount factor.", default=0.99),
        config_field(
            "n_step",
            "int",
            "Transitions combined in each replay return.",
            default=1,
        ),
        config_field(
            "buffer_size",
            "int",
            "Maximum transitions retained in replay memory.",
            default=1000000,
        ),
        config_field("batch_size", "int", "Transitions sampled per update.", default=256),
        config_field(
            "learning_starts",
            "int",
            "Uniform-random transitions collected before updates begin.",
            default=5000,
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
            "Replay gradient steps in each update cycle.",
            default=1,
        ),
        config_field(
            "tau",
            "float",
            "Polyak interpolation weight for target critics.",
            default=0.005,
        ),
        config_field(
            "initial_alpha",
            "float",
            "Initial entropy-temperature coefficient.",
            default=0.2,
        ),
        config_field(
            "automatic_entropy_tuning",
            "bool",
            "Learn the entropy temperature during replay updates.",
            default=True,
        ),
        config_field(
            "target_entropy",
            "float | None",
            "Entropy target; defaults to minus the flattened action dimension.",
            default=None,
        ),
        config_field(
            "log_std_min",
            "float",
            "Minimum Gaussian policy log standard deviation.",
            default=-20.0,
        ),
        config_field(
            "log_std_max",
            "float",
            "Maximum Gaussian policy log standard deviation.",
            default=2.0,
        ),
        config_field(
            "checkpoint_replay_buffer",
            "bool",
            "Include replay contents in resumable checkpoints.",
            default=True,
        ),
    ]

    def setup_algorithm(self) -> None:
        """Create the actor, critics, optimizers, replay memory, and SAC state."""
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if not isinstance(observation_space, (Box, Discrete)):
            raise TypeError("SACTrainer requires a Box or Discrete observation space")
        if not isinstance(action_space, Box):
            raise TypeError("SACTrainer requires a continuous Box action space")
        if not np.issubdtype(action_space.dtype, np.floating):
            raise TypeError("SAC actions require a floating-point Box")
        if not np.isfinite(action_space.low).all() or not np.isfinite(
            action_space.high
        ).all():
            raise ValueError("SAC actions require finite Box bounds")
        if not np.all(action_space.low < action_space.high):
            raise ValueError("SAC action-space lower bounds must be below upper bounds")
        if self.evaluation_environment.observation_space != observation_space:
            raise ValueError("Training and evaluation observation spaces must match")
        if self.evaluation_environment.action_space != action_space:
            raise ValueError("Training and evaluation action spaces must match")

        input_dim = (
            int(observation_space.n)
            if isinstance(observation_space, Discrete)
            else int(np.prod(observation_space.shape))
        )
        action_dim = int(np.prod(action_space.shape))
        self.gamma = float(self.trainer_config.get("gamma", 0.99))
        self.n_step = int(self.trainer_config.get("n_step", 1))
        self.buffer_size = int(self.trainer_config.get("buffer_size", 1000000))
        self.batch_size = int(self.trainer_config.get("batch_size", 256))
        self.learning_starts = int(self.trainer_config.get("learning_starts", 5000))
        self.train_frequency = int(self.trainer_config.get("train_frequency", 1))
        self.gradient_steps = int(self.trainer_config.get("gradient_steps", 1))
        self.tau = float(self.trainer_config.get("tau", 0.005))
        self.initial_alpha = float(self.trainer_config.get("initial_alpha", 0.2))
        self.automatic_entropy_tuning = bool(
            self.trainer_config.get("automatic_entropy_tuning", True)
        )
        configured_target_entropy = self.trainer_config.get("target_entropy")
        self.target_entropy = (
            -float(action_dim)
            if configured_target_entropy is None
            else float(configured_target_entropy)
        )
        self.log_std_min = float(self.trainer_config.get("log_std_min", -20.0))
        self.log_std_max = float(self.trainer_config.get("log_std_max", 2.0))
        self.checkpoint_replay_buffer = bool(
            self.trainer_config.get("checkpoint_replay_buffer", True)
        )
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive")
        if self.buffer_size <= 0 or self.batch_size <= 0:
            raise ValueError("buffer_size and batch_size must be positive")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.train_frequency <= 0 or self.gradient_steps <= 0:
            raise ValueError("train_frequency and gradient_steps must be positive")
        if not np.isfinite(self.tau) or not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be finite and in (0, 1]")
        if not np.isfinite(self.initial_alpha) or self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be finite and positive")
        if not np.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        if (
            not np.isfinite(self.log_std_min)
            or not np.isfinite(self.log_std_max)
            or self.log_std_min >= self.log_std_max
        ):
            raise ValueError("log_std_min must be finite and below log_std_max")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("SACTrainer requires gradient_accumulation_steps=1")

        model_section = self.config.get("models")
        if not isinstance(model_section, dict):
            raise ValueError(
                "SACTrainer requires models.actor.name and "
                "models.critics.name; dl-core does not provide default models"
            )
        actor_config = model_section.get("actor")
        critic_config = model_section.get("critics")
        if not isinstance(actor_config, dict):
            raise ValueError(
                "SACTrainer requires models.actor.name; dl-core does not "
                "provide a default actor"
            )
        if not isinstance(critic_config, dict):
            raise ValueError(
                "SACTrainer requires models.critics.name; dl-core does not "
                "provide default critics"
            )
        actor_config = dict(actor_config)
        critic_config = dict(critic_config)
        actor_name = actor_config.pop("name", None)
        critic_name = critic_config.pop("name", None)
        if not isinstance(actor_name, str) or not actor_name.strip():
            raise ValueError(
                "SACTrainer requires models.actor.name; dl-core does not "
                "provide a default actor"
            )
        if not isinstance(critic_name, str) or not critic_name.strip():
            raise ValueError(
                "SACTrainer requires models.critics.name; dl-core does not "
                "provide default critics"
            )
        actor_name = actor_name.strip()
        critic_name = critic_name.strip()
        actor_config.update({"input_dim": input_dim, "action_dim": action_dim})
        critic_config.update({"input_dim": input_dim, "action_dim": action_dim})
        self.models["actor"] = MODEL_REGISTRY.get(actor_name, actor_config)
        self.models["critics"] = MODEL_REGISTRY.get(critic_name, critic_config)
        self.models["target_critics"] = copy.deepcopy(self.models["critics"])
        self.models["target_critics"].eval()
        for parameter in self.models["target_critics"].parameters():
            parameter.requires_grad_(False)
        if self.automatic_entropy_tuning:
            self.models["temperature"] = _EntropyTemperature(self.initial_alpha)

        optimizer_section = self.config.get("optimizers", {})
        if not isinstance(optimizer_section, dict):
            raise TypeError("optimizers must be a mapping")
        optimizer_parameters: list[tuple[str, list[nn.Parameter]]] = [
            (
                "actor",
                [
                    parameter
                    for parameter in self.models["actor"].parameters()
                    if parameter.requires_grad
                ],
            ),
            (
                "critics",
                [
                    parameter
                    for parameter in self.models["critics"].parameters()
                    if parameter.requires_grad
                ],
            ),
        ]
        if self.automatic_entropy_tuning:
            optimizer_parameters.append(
                (
                    "temperature",
                    list(self.models["temperature"].parameters()),
                )
            )
        for optimizer_key, parameters in optimizer_parameters:
            if not parameters:
                raise ValueError(f"SAC {optimizer_key} has no trainable parameters")
            if "name" in optimizer_section:
                optimizer_config = dict(optimizer_section)
            else:
                optimizer_config = optimizer_section.get(
                    optimizer_key,
                    {"name": "adam", "lr": 3e-4},
                )
                if not isinstance(optimizer_config, dict):
                    raise TypeError(f"optimizers.{optimizer_key} must be a mapping")
                optimizer_config = dict(optimizer_config)
            optimizer_name = str(optimizer_config.pop("name", "adam"))
            self.optimizers[optimizer_key] = OPTIMIZER_REGISTRY.get(
                optimizer_name,
                parameters,
                **optimizer_config,
            )

        observation_shape = (
            () if isinstance(observation_space, Discrete) else observation_space.shape
        )
        observation_dtype = (
            np.int64
            if isinstance(observation_space, Discrete)
            else observation_space.dtype
        )
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            observation_shape=observation_shape,
            action_shape=action_space.shape,
            observation_dtype=observation_dtype,
            action_dtype=action_space.dtype,
            gamma=self.gamma,
            n_step=self.n_step,
            seed=self.seed,
        )
        self.random_generator = np.random.default_rng(self.seed)
        device = self.accelerator.get_device()
        with np.errstate(over="ignore", invalid="ignore"):
            action_low = action_space.low.astype(np.float64)
            action_high = action_space.high.astype(np.float64)
            action_scale = (action_high - action_low) / 2.0
            action_bias = (action_high + action_low) / 2.0
        self.action_scale = torch.as_tensor(
            action_scale,
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        self.action_bias = torch.as_tensor(
            action_bias,
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        if (
            not torch.isfinite(self.action_scale).all()
            or not torch.isfinite(self.action_bias).all()
            or not torch.all(self.action_scale > 0.0)
            or not torch.all(
                self.action_bias - self.action_scale
                < self.action_bias + self.action_scale
            )
        ):
            raise ValueError("SAC action bounds must be representable in float32")
        self.fixed_alpha = torch.tensor(
            self.initial_alpha,
            dtype=torch.float32,
            device=device,
        )

    def select_action(self, observation: Any, *, deterministic: bool) -> np.ndarray:
        """Select a bounded deterministic or stochastic continuous action."""
        return self._select_action(observation, deterministic=deterministic)

    def _select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        return self._select_actions(
            np.expand_dims(np.asarray(observation), axis=0),
            deterministic=deterministic,
        ).actions[0]

    def select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[np.ndarray]:
        """Select bounded continuous actions with one actor call."""
        return self._select_actions(observations, deterministic=deterministic)

    def _select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[np.ndarray]:
        observation_batch = np.asarray(observations)
        if observation_batch.ndim == 0 or observation_batch.shape[0] == 0:
            raise ValueError("SAC action selection requires at least one observation")
        if any(
            not self.environment.observation_space.contains(observation)
            for observation in observation_batch
        ):
            raise ValueError("Observation is outside the configured space")
        action_space = self.environment.action_space
        batch_size = int(observation_batch.shape[0])
        warmup = np.zeros(batch_size, dtype=np.bool_)
        if not deterministic:
            warmup = (
                self.global_step + np.arange(batch_size) < self.learning_starts
            )
        action_batch = np.empty(
            (batch_size, *action_space.shape),
            dtype=action_space.dtype,
        )
        if warmup.any():
            action_batch[warmup] = self.random_generator.uniform(
                action_space.low,
                action_space.high,
                size=(int(warmup.sum()), *action_space.shape),
            ).astype(action_space.dtype)

        policy = ~warmup
        if policy.any():
            observation_tensor = self._observations_to_tensor(
                observation_batch[policy]
            )
            actor = self.models["actor"]
            was_training = actor.training
            try:
                if deterministic:
                    actor.eval()
                with torch.no_grad(), self.accelerator.autocast_context():
                    actions, _ = self._sample_action_and_log_probability(
                        observation_tensor,
                        deterministic=deterministic,
                    )
            finally:
                actor.train(was_training)
            action_batch[policy] = (
                actions.reshape(-1, *action_space.shape)
                .detach()
                .cpu()
                .numpy()
                .astype(action_space.dtype)
            )
        action_batch = np.clip(
            action_batch,
            action_space.low,
            action_space.high,
        )
        return BatchActionOutput(
            actions=[
                np.asarray(action, dtype=action_space.dtype)
                .reshape(action_space.shape)
                .copy()
                for action in action_batch
            ]
        )

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float] | None:
        """Store one transition and run scheduled SAC replay updates."""
        logs = self._process_transition_batch(
            TransitionBatch(
                observations=np.expand_dims(
                    np.asarray(transition.observation),
                    axis=0,
                ),
                actions=np.expand_dims(np.asarray(transition.action), axis=0),
                rewards=np.asarray([transition.reward], dtype=np.float32),
                next_observations=np.expand_dims(
                    np.asarray(transition.next_observation),
                    axis=0,
                ),
                terminated=np.asarray([transition.terminated], dtype=np.bool_),
                truncated=np.asarray([transition.truncated], dtype=np.bool_),
                infos=[transition.info],
                action_info=[transition.action_info],
            )
        )
        return logs[-1] if logs else None

    def process_transition_batch(
        self,
        transitions: TransitionBatch[Any, Any],
    ) -> list[dict[str, float]]:
        """Insert a vector step and run every crossed SAC update cycle."""
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
            raise ValueError("Transition observation is outside the configured space")
        if any(
            not observation_space.contains(observation)
            for observation in np.asarray(transitions.next_observations)
        ):
            raise ValueError("Transition next observation is outside the configured space")
        if any(
            not action_space.contains(action)
            for action in np.asarray(transitions.actions)
        ):
            raise ValueError("Transition action is outside the configured space")
        if not np.isfinite(transitions.rewards).all():
            raise ValueError("Transition reward must be finite")
        previous_global_step = self.global_step - transitions.size
        previous_replay_size = len(self.replay_buffer)
        added_per_environment = self.replay_buffer.add_batch(transitions)
        if (
            len(self.replay_buffer) < self.batch_size
            or self.global_step < self.learning_starts
        ):
            return []
        replay_ready_step = previous_global_step + 1
        if previous_replay_size < self.batch_size:
            required_entries = self.batch_size - previous_replay_size
            replay_ready_step = previous_global_step + int(
                np.flatnonzero(
                    np.cumsum(added_per_environment) >= required_entries
                )[0]
            ) + 1
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
        scheduled_updates = [
            scheduled_step
            for scheduled_step in range(
                first_update_step,
                self.global_step + 1,
                self.train_frequency,
            )
            if self.should_update(scheduled_step, transitions)
        ]
        if not scheduled_updates:
            return []

        update_logs: list[dict[str, float]] = []
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        q_means: list[float] = []
        target_q_means: list[float] = []
        for gradient_update in range(
            len(scheduled_updates) * self.gradient_steps
        ):
            batch = self.replay_buffer.sample(
                self.batch_size,
                self.accelerator.get_device(),
            )
            observations = self._observations_to_tensor(batch.observations)
            next_observations = self._observations_to_tensor(batch.next_observations)
            actions = batch.actions.float()
            with self.accelerator.autocast_context():
                with torch.no_grad():
                    next_actions, next_log_probabilities = (
                        self._sample_action_and_log_probability(
                            next_observations,
                            deterministic=False,
                        )
                    )
                    target_q1, target_q2 = self._q_values(
                        self.models["target_critics"],
                        next_observations,
                        next_actions,
                    )
                    next_q = torch.minimum(target_q1, target_q2) - (
                        self._alpha().detach() * next_log_probabilities
                    )
                    targets = batch.rewards + (
                        batch.discounts * (~batch.terminated).float() * next_q
                    )
                    if not torch.isfinite(targets).all():
                        raise FloatingPointError("SAC critic targets must be finite")
                current_q1, current_q2 = self._q_values(
                    self.models["critics"],
                    observations,
                    actions,
                )
                critic_loss = functional.mse_loss(
                    current_q1,
                    targets,
                ) + functional.mse_loss(current_q2, targets)
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError("SAC critic loss must be finite")
            self.optimizers["critics"].zero_grad(set_to_none=True)
            self.accelerator.backward(critic_loss, self.models["critics"])
            self.accelerator.optimizer_step(
                self.optimizers["critics"],
                self.models["critics"],
            )

            critic_parameters = list(self.models["critics"].parameters())
            critic_gradient_flags = [
                parameter.requires_grad for parameter in critic_parameters
            ]
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                with self.accelerator.autocast_context():
                    new_actions, log_probabilities = (
                        self._sample_action_and_log_probability(
                            observations,
                            deterministic=False,
                        )
                    )
                    policy_q1, policy_q2 = self._q_values(
                        self.models["critics"],
                        observations,
                        new_actions,
                    )
                    actor_loss = (
                        self._alpha().detach() * log_probabilities
                        - torch.minimum(policy_q1, policy_q2)
                    ).mean()
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError("SAC actor loss must be finite")
                self.optimizers["actor"].zero_grad(set_to_none=True)
                self.accelerator.backward(actor_loss, self.models["actor"])
                self.accelerator.optimizer_step(
                    self.optimizers["actor"],
                    self.models["actor"],
                )
            finally:
                for parameter, requires_grad in zip(
                    critic_parameters,
                    critic_gradient_flags,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)

            alpha_loss = torch.zeros((), device=self.accelerator.get_device())
            if self.automatic_entropy_tuning:
                temperature = self.models["temperature"]
                log_alpha = self.accelerator.unwrap_model(temperature).log_alpha
                alpha_loss = -(
                    log_alpha * (log_probabilities.detach() + self.target_entropy)
                ).mean()
                if not torch.isfinite(alpha_loss):
                    raise FloatingPointError("SAC temperature loss must be finite")
                self.optimizers["temperature"].zero_grad(set_to_none=True)
                self.accelerator.backward(alpha_loss, temperature)
                self.accelerator.optimizer_step(
                    self.optimizers["temperature"],
                    temperature,
                )

            source_critics = self.accelerator.unwrap_model(self.models["critics"])
            target_critics = self.accelerator.unwrap_model(
                self.models["target_critics"]
            )
            with torch.no_grad():
                for target_parameter, source_parameter in zip(
                    target_critics.parameters(),
                    source_critics.parameters(),
                    strict=True,
                ):
                    target_parameter.mul_(1.0 - self.tau)
                    target_parameter.add_(source_parameter, alpha=self.tau)
                for target_buffer, source_buffer in zip(
                    target_critics.buffers(),
                    source_critics.buffers(),
                    strict=True,
                ):
                    target_buffer.copy_(source_buffer)

            critic_losses.append(float(critic_loss.detach().item()))
            actor_losses.append(float(actor_loss.detach().item()))
            alpha_losses.append(float(alpha_loss.detach().item()))
            q_means.append(
                float(
                    torch.minimum(current_q1, current_q2).detach().mean().item()
                )
            )
            target_q_means.append(float(targets.detach().mean().item()))

            if (gradient_update + 1) % self.gradient_steps == 0:
                update_logs.append(
                    {
                        "sac/critic_loss": float(np.mean(critic_losses)),
                        "sac/actor_loss": float(np.mean(actor_losses)),
                        "sac/alpha_loss": float(np.mean(alpha_losses)),
                        "sac/alpha": float(self._alpha().detach().item()),
                        "sac/q_mean": float(np.mean(q_means)),
                        "sac/target_q_mean": float(np.mean(target_q_means)),
                        "sac/replay_size": float(len(self.replay_buffer)),
                    }
                )
                critic_losses = []
                actor_losses = []
                alpha_losses = []
                q_means = []
                target_q_means = []

        return update_logs

    def _observations_to_tensor(self, observations: Any) -> torch.Tensor:
        tensor = torch.as_tensor(observations, device=self.accelerator.get_device())
        if isinstance(self.environment.observation_space, Discrete):
            indices = tensor.long() - int(self.environment.observation_space.start)
            return functional.one_hot(
                indices,
                num_classes=self.environment.observation_space.n,
            ).float()
        return tensor.float()

    def _sample_action_and_log_probability(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.models["actor"](observations)
        if not isinstance(output, Mapping):
            raise TypeError("SAC actors must return a mapping")
        mean = output.get("mean")
        log_std = output.get("log_std")
        expected_shape = (
            observations.shape[0],
            int(np.prod(self.environment.action_space.shape)),
        )
        if not isinstance(mean, torch.Tensor) or not isinstance(log_std, torch.Tensor):
            raise TypeError("SAC actors must return 'mean' and 'log_std' tensors")
        if mean.shape != expected_shape or log_std.shape != expected_shape:
            raise ValueError("SAC actor outputs must have shape [batch, action_dimensions]")
        if not mean.is_floating_point() or not log_std.is_floating_point():
            raise TypeError("SAC actor outputs must use floating-point dtypes")
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise FloatingPointError("SAC actor outputs must be finite")
        # Gaussian statistics and the change-of-variables correction are kept in
        # float32 because exp(-20) underflows in fp16 under autocast.
        mean = mean.float()
        log_std = log_std.float().clamp(self.log_std_min, self.log_std_max)
        distribution = Normal(mean, log_std.exp())
        raw_actions = mean if deterministic else distribution.rsample()
        squashed_actions = torch.tanh(raw_actions)
        scale = self.action_scale.to(dtype=squashed_actions.dtype)
        bias = self.action_bias.to(dtype=squashed_actions.dtype)
        actions = squashed_actions * scale + bias
        log_tanh_jacobian = 2.0 * (
            math.log(2.0)
            - raw_actions
            - functional.softplus(-2.0 * raw_actions)
        )
        log_probabilities = (
            distribution.log_prob(raw_actions)
            - log_tanh_jacobian
            - torch.log(scale)
        ).sum(dim=1)
        if not torch.isfinite(actions).all() or not torch.isfinite(
            log_probabilities
        ).all():
            raise FloatingPointError("SAC actions and log probabilities must be finite")
        return actions, log_probabilities

    def _q_values(
        self,
        model: nn.Module,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = model(observations, actions)
        if not isinstance(output, Mapping):
            raise TypeError("SAC critics must return a mapping")
        q1 = output.get("q1")
        q2 = output.get("q2")
        expected_shape = (observations.shape[0],)
        if not isinstance(q1, torch.Tensor) or not isinstance(q2, torch.Tensor):
            raise TypeError("SAC critics must return 'q1' and 'q2' tensors")
        if q1.shape != expected_shape or q2.shape != expected_shape:
            raise ValueError("SAC critic outputs must have shape [batch]")
        if not q1.is_floating_point() or not q2.is_floating_point():
            raise TypeError("SAC critic outputs must use floating-point dtypes")
        if not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
            raise FloatingPointError("SAC critic outputs must be finite")
        return q1, q2

    def _alpha(self) -> torch.Tensor:
        if self.automatic_entropy_tuning:
            alpha = self.models["temperature"]()
        else:
            alpha = self.fixed_alpha
        if not torch.isfinite(alpha) or alpha <= 0.0:
            raise FloatingPointError("SAC entropy temperature must be finite and positive")
        return alpha

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return entropy, exploration, and optional replay state."""
        return {
            "alpha": float(self._alpha().detach().item()),
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "target_entropy": self.target_entropy,
            "random_generator_state": self.random_generator.bit_generator.state,
            "replay_buffer": (
                self.replay_buffer.state_dict()
                if self.checkpoint_replay_buffer
                else None
            ),
            "checkpoint_replay_buffer": self.checkpoint_replay_buffer,
            "training_parameters": {
                "gamma": self.gamma,
                "n_step": self.n_step,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "learning_starts": self.learning_starts,
                "train_frequency": self.train_frequency,
                "gradient_steps": self.gradient_steps,
                "tau": self.tau,
                "log_std_min": self.log_std_min,
                "log_std_max": self.log_std_max,
            },
            "observation_space": repr(self.environment.observation_space),
            "action_space": repr(self.environment.action_space),
        }

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore entropy, exploration, and optional replay state."""
        if not isinstance(state, dict):
            raise TypeError("Checkpoint algorithm state must be a mapping")
        if state.get("observation_space") != repr(self.environment.observation_space):
            raise ValueError("Checkpoint observation space does not match")
        if state.get("action_space") != repr(self.environment.action_space):
            raise ValueError("Checkpoint action space does not match")
        if state.get("automatic_entropy_tuning") is not self.automatic_entropy_tuning:
            raise ValueError("Checkpoint entropy-tuning policy does not match config")
        target_entropy = float(state.get("target_entropy", float("nan")))
        if not np.isfinite(target_entropy) or target_entropy != self.target_entropy:
            raise ValueError("Checkpoint target entropy does not match config")
        alpha = float(state.get("alpha", float("nan")))
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("Checkpoint alpha must be finite and positive")
        if not np.isclose(alpha, float(self._alpha().detach().item())):
            raise ValueError("Checkpoint alpha does not match restored model state")
        if state.get("checkpoint_replay_buffer") is not self.checkpoint_replay_buffer:
            raise ValueError("Checkpoint replay-buffer policy does not match config")
        expected_training_parameters = {
            "gamma": self.gamma,
            "n_step": self.n_step,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "train_frequency": self.train_frequency,
            "gradient_steps": self.gradient_steps,
            "tau": self.tau,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }
        saved_training_parameters = state.get("training_parameters")
        if (
            isinstance(saved_training_parameters, dict)
            and "n_step" not in saved_training_parameters
        ):
            saved_training_parameters = {
                **saved_training_parameters,
                "n_step": 1,
            }
        if saved_training_parameters != expected_training_parameters:
            raise ValueError("Checkpoint SAC training parameters do not match config")

        generator_state = state.get("random_generator_state")
        if not isinstance(generator_state, dict):
            raise ValueError("Checkpoint exploration generator state is invalid")
        restored_generator = np.random.default_rng()
        try:
            restored_generator.bit_generator.state = generator_state
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Checkpoint exploration generator state is invalid"
            ) from error

        restored_replay_buffer = self.replay_buffer
        replay_state = state.get("replay_buffer")
        if self.checkpoint_replay_buffer:
            if not isinstance(replay_state, dict):
                raise ValueError("Checkpoint replay buffer state is missing")
            restored_replay_buffer = ReplayBuffer(
                capacity=self.replay_buffer.capacity,
                observation_shape=self.replay_buffer.observation_shape,
                action_shape=self.replay_buffer.action_shape,
                observation_dtype=self.replay_buffer.observation_dtype,
                action_dtype=self.replay_buffer.action_dtype,
                gamma=self.gamma,
                n_step=self.n_step,
                seed=self.seed,
            )
            restored_replay_buffer.load_state_dict(replay_state)

        self.random_generator = restored_generator
        self.replay_buffer = restored_replay_buffer
