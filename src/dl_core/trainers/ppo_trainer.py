"""Proximal policy optimization trainer for discrete and continuous control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical, Distribution, Normal

from dl_core.core import (
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    ActionOutput,
    RLTrainer,
    RolloutBuffer,
    Transition,
    config_field,
    register_trainer,
)


@register_trainer("ppo")
class PPOTrainer(RLTrainer):
    """Clipped PPO with generalized advantage estimation."""

    CONFIG_FIELDS = RLTrainer.CONFIG_FIELDS + [
        config_field("gamma", "float", "Reward discount factor.", default=0.99),
        config_field(
            "gae_lambda",
            "float",
            "Generalized advantage estimation trace parameter.",
            default=0.95,
        ),
        config_field(
            "rollout_steps",
            "int",
            "Maximum transitions collected before a policy update.",
            default=2048,
        ),
        config_field(
            "update_epochs",
            "int",
            "Optimization passes over each rollout.",
            default=10,
        ),
        config_field(
            "minibatch_size",
            "int",
            "Maximum samples in each PPO minibatch.",
            default=64,
        ),
        config_field(
            "clip_range",
            "float",
            "Symmetric policy-ratio clipping range.",
            default=0.2,
        ),
        config_field(
            "value_clip_range",
            "float | None",
            "Optional symmetric value-update clipping range.",
            default=0.2,
        ),
        config_field(
            "value_loss_coefficient",
            "float",
            "Weight applied to the value objective.",
            default=0.5,
        ),
        config_field(
            "entropy_coefficient",
            "float",
            "Weight applied to the entropy bonus.",
            default=0.01,
        ),
        config_field(
            "normalize_advantages",
            "bool",
            "Normalize advantages within each rollout.",
            default=True,
        ),
    ]

    def setup_algorithm(self) -> None:
        """Create actor-critic, optimizer, rollout storage, and update RNG."""
        if getattr(self.environment, "num_envs", 1) > 1:
            raise NotImplementedError(
                "PPO vector collection requires the batched rollout buffer"
            )
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if not isinstance(observation_space, (Box, Discrete)):
            raise TypeError("PPOTrainer requires a Box or Discrete observation space")
        if not isinstance(action_space, (Box, Discrete)):
            raise TypeError("PPOTrainer requires a Box or Discrete action space")
        if self.evaluation_environment.observation_space != observation_space:
            raise ValueError("Training and evaluation observation spaces must match")
        if self.evaluation_environment.action_space != action_space:
            raise ValueError("Training and evaluation action spaces must match")
        if isinstance(action_space, Box):
            if not np.issubdtype(action_space.dtype, np.floating):
                raise TypeError("PPO continuous actions require a floating-point Box")
            if not np.isfinite(action_space.low).all() or not np.isfinite(
                action_space.high
            ).all():
                raise ValueError("PPO continuous actions require finite Box bounds")
            if not np.all(action_space.low < action_space.high):
                raise ValueError("PPO action-space lower bounds must be below upper bounds")

        input_dim = (
            int(observation_space.n)
            if isinstance(observation_space, Discrete)
            else int(np.prod(observation_space.shape))
        )
        action_dim = (
            int(action_space.n)
            if isinstance(action_space, Discrete)
            else int(np.prod(action_space.shape))
        )
        self.gamma = float(self.trainer_config.get("gamma", 0.99))
        self.gae_lambda = float(self.trainer_config.get("gae_lambda", 0.95))
        self.rollout_steps = int(self.trainer_config.get("rollout_steps", 2048))
        self.update_epochs = int(self.trainer_config.get("update_epochs", 10))
        self.minibatch_size = int(self.trainer_config.get("minibatch_size", 64))
        self.clip_range = float(self.trainer_config.get("clip_range", 0.2))
        value_clip_range = self.trainer_config.get("value_clip_range", 0.2)
        self.value_clip_range = (
            float(value_clip_range) if value_clip_range is not None else None
        )
        self.value_loss_coefficient = float(
            self.trainer_config.get("value_loss_coefficient", 0.5)
        )
        self.entropy_coefficient = float(
            self.trainer_config.get("entropy_coefficient", 0.01)
        )
        self.normalize_advantages = bool(
            self.trainer_config.get("normalize_advantages", True)
        )
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.rollout_steps <= 0 or self.update_epochs <= 0:
            raise ValueError("rollout_steps and update_epochs must be positive")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if not np.isfinite(self.clip_range) or self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        if self.value_clip_range is not None and (
            not np.isfinite(self.value_clip_range) or self.value_clip_range <= 0.0
        ):
            raise ValueError("value_clip_range must be positive when provided")
        if (
            not np.isfinite(self.value_loss_coefficient)
            or not np.isfinite(self.entropy_coefficient)
            or self.value_loss_coefficient < 0.0
            or self.entropy_coefficient < 0.0
        ):
            raise ValueError("loss coefficients must be finite and non-negative")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("PPOTrainer requires gradient_accumulation_steps=1")

        model_section = self.config.get("models", {})
        if not isinstance(model_section, dict):
            raise TypeError("models must be a mapping")
        policy_config = model_section.get("policy", {})
        if not isinstance(policy_config, dict):
            raise TypeError("models.policy must be a mapping")
        policy_config = dict(policy_config)
        model_name = str(policy_config.pop("name", "ppo_actor_critic"))
        policy_config.update(
            {
                "input_dim": input_dim,
                "action_dim": action_dim,
                "continuous_actions": isinstance(action_space, Box),
            }
        )
        self.models["policy"] = MODEL_REGISTRY.get(model_name, policy_config)
        trainable_parameters = [
            parameter
            for parameter in self.models["policy"].parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("PPO policy has no trainable parameters")

        optimizer_config = self.config.get(
            "optimizers",
            {"name": "adam", "lr": 3e-4},
        )
        if not isinstance(optimizer_config, dict):
            raise TypeError("optimizers must be a flat mapping")
        optimizer_config = dict(optimizer_config)
        optimizer_name = str(optimizer_config.pop("name", "adam"))
        self.optimizers["policy"] = OPTIMIZER_REGISTRY.get(
            optimizer_name,
            trainable_parameters,
            **optimizer_config,
        )
        self.rollout_buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            num_envs=self.environment.num_envs,
        )
        self.random_generator = np.random.default_rng(self.seed)

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
    ) -> ActionOutput[Any]:
        """Sample or deterministically select an actor-critic action."""
        if not self.environment.observation_space.contains(observation):
            raise ValueError("Observation is outside the configured space")
        observation_batch = np.expand_dims(np.asarray(observation), axis=0)
        policy = self.models["policy"]
        was_training = policy.training
        try:
            if deterministic:
                policy.eval()
            with torch.no_grad(), self.accelerator.autocast_context():
                distribution, value = self._distribution_and_value(
                    self._observations_to_tensor(observation_batch)
                )
                if isinstance(distribution, Categorical):
                    policy_action = (
                        torch.argmax(distribution.logits, dim=1)
                        if deterministic
                        else distribution.sample()
                    )
                    log_probability = distribution.log_prob(policy_action)
                    environment_action: Any = int(policy_action.item()) + int(
                        self.environment.action_space.start
                    )
                    stored_action: Any = int(policy_action.item())
                else:
                    raw_action = (
                        distribution.mean if deterministic else distribution.sample()
                    )
                    log_probability = distribution.log_prob(raw_action).sum(dim=1)
                    bounded_action = torch.tanh(raw_action)
                    action_space = self.environment.action_space
                    action_scale = torch.as_tensor(
                        (action_space.high - action_space.low) / 2.0,
                        dtype=bounded_action.dtype,
                        device=bounded_action.device,
                    ).reshape(1, -1)
                    action_bias = torch.as_tensor(
                        (action_space.high + action_space.low) / 2.0,
                        dtype=bounded_action.dtype,
                        device=bounded_action.device,
                    ).reshape(1, -1)
                    environment_action = (
                        (bounded_action * action_scale) + action_bias
                    ).reshape(action_space.shape)
                    environment_action = environment_action.cpu().numpy().astype(
                        action_space.dtype,
                        copy=False,
                    )
                    stored_action = raw_action.reshape(action_space.shape).cpu().numpy()
        finally:
            policy.train(was_training)
        return ActionOutput(
            action=environment_action,
            info={
                "policy_action": stored_action,
                "log_probability": float(log_probability.item()),
                "value": float(value.item()),
            },
        )

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float] | None:
        """Collect one on-policy transition and update at rollout boundaries."""
        for field_name in ("policy_action", "log_probability", "value"):
            if field_name not in transition.action_info:
                raise ValueError(f"PPO transition is missing action_info.{field_name}")
        next_value = 0.0
        if not transition.terminated:
            next_observation_batch = np.expand_dims(
                np.asarray(transition.next_observation),
                axis=0,
            )
            with torch.no_grad(), self.accelerator.autocast_context():
                _, next_value_tensor = self._distribution_and_value(
                    self._observations_to_tensor(next_observation_batch)
                )
            next_value = float(next_value_tensor.item())
        self.rollout_buffer.add(
            observation=transition.observation,
            action=transition.action_info["policy_action"],
            reward=transition.reward,
            value=float(transition.action_info["value"]),
            log_probability=float(transition.action_info["log_probability"]),
            next_value=next_value,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )
        budget_exhausted = (
            self.total_timesteps > 0 and self.global_step >= self.total_timesteps
        ) or (
            self.max_episodes is not None
            and transition.done
            and self.current_episode + 1 >= self.max_episodes
        )
        if len(self.rollout_buffer) < self.rollout_steps and not budget_exhausted:
            return None

        batch = self.rollout_buffer.compute_batch(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.accelerator.get_device(),
        )
        advantages = batch.advantages
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        sample_count = len(batch.returns)
        for _ in range(self.update_epochs):
            indices = self.random_generator.permutation(sample_count)
            for start in range(0, sample_count, self.minibatch_size):
                minibatch_indices = torch.as_tensor(
                    indices[start : start + self.minibatch_size],
                    device=self.accelerator.get_device(),
                )
                observations = self._observations_to_tensor(
                    batch.observations[minibatch_indices]
                )
                with self.accelerator.autocast_context():
                    distribution, values = self._distribution_and_value(observations)
                    actions = batch.actions[minibatch_indices]
                    if isinstance(distribution, Categorical):
                        new_log_probabilities = distribution.log_prob(actions.long())
                        entropy = distribution.entropy()
                    else:
                        actions = actions.float().reshape(values.shape[0], -1)
                        new_log_probabilities = distribution.log_prob(actions).sum(dim=1)
                        entropy = distribution.entropy().sum(dim=1)
                    old_log_probabilities = batch.old_log_probabilities[
                        minibatch_indices
                    ]
                    minibatch_advantages = advantages[minibatch_indices]
                    log_ratio = new_log_probabilities - old_log_probabilities
                    ratio = torch.exp(log_ratio)
                    unclipped_objective = ratio * minibatch_advantages
                    clipped_objective = torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    ) * minibatch_advantages
                    policy_loss = -torch.min(
                        unclipped_objective,
                        clipped_objective,
                    ).mean()
                    returns = batch.returns[minibatch_indices]
                    if self.value_clip_range is None:
                        value_loss = 0.5 * torch.mean((values - returns) ** 2)
                    else:
                        old_values = batch.old_values[minibatch_indices]
                        clipped_values = old_values + torch.clamp(
                            values - old_values,
                            -self.value_clip_range,
                            self.value_clip_range,
                        )
                        value_loss = 0.5 * torch.max(
                            (values - returns) ** 2,
                            (clipped_values - returns) ** 2,
                        ).mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + (self.value_loss_coefficient * value_loss)
                        - (self.entropy_coefficient * entropy_mean)
                    )

                self.optimizers["policy"].zero_grad(set_to_none=True)
                self.accelerator.backward(loss, self.models["policy"])
                self.accelerator.optimizer_step(
                    self.optimizers["policy"],
                    self.models["policy"],
                )
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
                    )
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy_mean.detach().item()))
                approximate_kls.append(float(approximate_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

        self.rollout_buffer.clear()
        return {
            "ppo/policy_loss": float(np.mean(policy_losses)),
            "ppo/value_loss": float(np.mean(value_losses)),
            "ppo/entropy": float(np.mean(entropies)),
            "ppo/approximate_kl": float(np.mean(approximate_kls)),
            "ppo/clip_fraction": float(np.mean(clip_fractions)),
            "ppo/rollout_size": float(sample_count),
        }

    def _observations_to_tensor(self, observations: Any) -> torch.Tensor:
        tensor = torch.as_tensor(observations, device=self.accelerator.get_device())
        if isinstance(self.environment.observation_space, Discrete):
            indices = tensor.long() - int(self.environment.observation_space.start)
            return torch.nn.functional.one_hot(
                indices,
                num_classes=self.environment.observation_space.n,
            ).float()
        return tensor.float()

    def _distribution_and_value(
        self,
        observations: torch.Tensor,
    ) -> tuple[Distribution, torch.Tensor]:
        output = self.models["policy"](observations)
        if not isinstance(output, dict) or not isinstance(
            output.get("value"), torch.Tensor
        ):
            raise TypeError("PPO policy must return a dict containing a value tensor")
        value = output["value"]
        if value.shape != (observations.shape[0],):
            raise ValueError("PPO value output must have shape [batch]")
        if not value.is_floating_point():
            raise TypeError("PPO value output must use a floating-point dtype")
        if isinstance(self.environment.action_space, Discrete):
            logits = output.get("logits")
            if not isinstance(logits, torch.Tensor) or logits.shape != (
                observations.shape[0],
                self.environment.action_space.n,
            ):
                raise ValueError("PPO discrete policy logits have an invalid shape")
            if not logits.is_floating_point():
                raise TypeError("PPO policy logits must use a floating-point dtype")
            return Categorical(logits=logits), value
        mean = output.get("mean")
        log_std = output.get("log_std")
        action_dim = int(np.prod(self.environment.action_space.shape))
        expected_shape = (observations.shape[0], action_dim)
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(log_std, torch.Tensor)
            or mean.shape != expected_shape
            or log_std.shape != expected_shape
        ):
            raise ValueError("PPO continuous policy parameters have an invalid shape")
        if not mean.is_floating_point() or not log_std.is_floating_point():
            raise TypeError("PPO policy parameters must use floating-point dtypes")
        return Normal(mean, torch.exp(log_std.clamp(-20.0, 2.0))), value

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return partial rollout and minibatch-generator state."""
        return {
            "rollout_buffer": self.rollout_buffer.state_dict(),
            "random_generator_state": self.random_generator.bit_generator.state,
            "observation_space": repr(self.environment.observation_space),
            "action_space": repr(self.environment.action_space),
        }

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore partial rollout and minibatch-generator state."""
        if not isinstance(state, dict):
            raise TypeError("Checkpoint algorithm state must be a mapping")
        if state.get("observation_space") != repr(self.environment.observation_space):
            raise ValueError("Checkpoint observation space does not match")
        if state.get("action_space") != repr(self.environment.action_space):
            raise ValueError("Checkpoint action space does not match")
        rollout_state = state.get("rollout_buffer")
        if not isinstance(rollout_state, dict):
            raise ValueError("Checkpoint rollout buffer state is invalid")
        generator_state = state.get("random_generator_state")
        if not isinstance(generator_state, dict):
            raise ValueError("Checkpoint minibatch generator state is invalid")
        restored_buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            num_envs=self.environment.num_envs,
        )
        restored_buffer.load_state_dict(rollout_state)
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if restored_buffer.observations is None or restored_buffer.actions is None:
            restored_observations = np.asarray([])
            restored_actions = np.asarray([])
        else:
            restored_observations = restored_buffer.observations[
                : len(restored_buffer)
            ].reshape(
                -1,
                *restored_buffer.observations.shape[2:],
            )
            restored_actions = restored_buffer.actions[
                : len(restored_buffer)
            ].reshape(
                -1,
                *restored_buffer.actions.shape[2:],
            )
        if any(
            not observation_space.contains(observation)
            for observation in restored_observations
        ):
            raise ValueError("Checkpoint rollout contains an invalid observation")
        if isinstance(action_space, Discrete):
            if any(
                action.shape != ()
                or not np.issubdtype(action.dtype, np.integer)
                or not 0 <= int(action) < action_space.n
                for action in restored_actions
            ):
                raise ValueError("Checkpoint rollout contains an invalid policy action")
        elif any(
            action.shape != action_space.shape
            or not np.issubdtype(action.dtype, np.floating)
            or not np.isfinite(action).all()
            for action in restored_actions
        ):
            raise ValueError("Checkpoint rollout contains an invalid policy action")
        restored_generator = np.random.default_rng()
        try:
            restored_generator.bit_generator.state = generator_state
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint minibatch generator state is invalid") from error
        self.rollout_buffer = restored_buffer
        self.random_generator = restored_generator
