"""Deep Q-network trainer for discrete-action Gymnasium environments."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
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


@register_trainer("dqn")
class DQNTrainer(RLTrainer):
    """Replay-based DQN with target networks and optional Double-DQN targets."""

    CONFIG_FIELDS = RLTrainer.CONFIG_FIELDS + [
        config_field("gamma", "float", "Reward discount factor.", default=0.99),
        config_field(
            "buffer_size",
            "int",
            "Maximum transitions retained in replay memory.",
            default=100000,
        ),
        config_field("batch_size", "int", "Transitions sampled per update.", default=64),
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
            "Replay gradient steps in each update cycle.",
            default=1,
        ),
        config_field(
            "target_update_frequency",
            "int",
            "Environment transitions between hard target-network updates.",
            default=1000,
        ),
        config_field(
            "double_dqn",
            "bool",
            "Select next actions online and evaluate them with the target network.",
            default=True,
        ),
        config_field(
            "epsilon_start",
            "float",
            "Initial epsilon-greedy exploration probability.",
            default=1.0,
        ),
        config_field(
            "epsilon_end",
            "float",
            "Final epsilon-greedy exploration probability.",
            default=0.05,
        ),
        config_field(
            "epsilon_decay_steps",
            "int",
            "Transitions used for linear exploration decay.",
            default=10000,
        ),
        config_field(
            "checkpoint_replay_buffer",
            "bool",
            "Include replay contents in resumable checkpoints.",
            default=True,
        ),
    ]

    def setup_algorithm(self) -> None:
        """Create DQN networks, optimizer, replay memory, and exploration state."""
        if not isinstance(self.environment.action_space, Discrete):
            raise TypeError("DQNTrainer requires a Discrete action space")
        if not isinstance(self.environment.observation_space, (Box, Discrete)):
            raise TypeError("DQNTrainer requires a Box or Discrete observation space")
        if self.evaluation_environment.action_space != self.environment.action_space:
            raise ValueError("Training and evaluation action spaces must match")
        if (
            self.evaluation_environment.observation_space
            != self.environment.observation_space
        ):
            raise ValueError("Training and evaluation observation spaces must match")
        if isinstance(self.environment.observation_space, Discrete):
            observation_shape: tuple[int, ...] = ()
            observation_dtype = np.int64
            input_dim = int(self.environment.observation_space.n)
        else:
            observation_shape = self.environment.observation_space.shape
            observation_dtype = self.environment.observation_space.dtype
            input_dim = int(np.prod(observation_shape))

        self.gamma = float(self.trainer_config.get("gamma", 0.99))
        self.buffer_size = int(self.trainer_config.get("buffer_size", 100000))
        self.batch_size = int(self.trainer_config.get("batch_size", 64))
        self.learning_starts = int(self.trainer_config.get("learning_starts", 1000))
        self.train_frequency = int(self.trainer_config.get("train_frequency", 1))
        self.gradient_steps = int(self.trainer_config.get("gradient_steps", 1))
        self.target_update_frequency = int(
            self.trainer_config.get("target_update_frequency", 1000)
        )
        self.double_dqn = bool(self.trainer_config.get("double_dqn", True))
        self.epsilon_start = float(self.trainer_config.get("epsilon_start", 1.0))
        self.epsilon_end = float(self.trainer_config.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(
            self.trainer_config.get("epsilon_decay_steps", 10000)
        )
        self.checkpoint_replay_buffer = bool(
            self.trainer_config.get("checkpoint_replay_buffer", True)
        )
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.buffer_size <= 0 or self.batch_size <= 0:
            raise ValueError("buffer_size and batch_size must be positive")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.train_frequency <= 0 or self.gradient_steps <= 0:
            raise ValueError("train_frequency and gradient_steps must be positive")
        if self.target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("DQNTrainer requires gradient_accumulation_steps=1")

        model_section = self.config.get("models", {})
        if not isinstance(model_section, dict):
            raise TypeError("models must be a mapping")
        q_network_config = model_section.get("q_network", {})
        if not isinstance(q_network_config, dict):
            raise TypeError("models.q_network must be a mapping")
        q_network_config = dict(q_network_config)
        model_name = str(q_network_config.pop("name", "dqn_mlp"))
        q_network_config["input_dim"] = input_dim
        q_network_config["action_dim"] = int(self.environment.action_space.n)
        self.models["online"] = MODEL_REGISTRY.get(model_name, q_network_config)
        self.models["target"] = copy.deepcopy(self.models["online"])
        self.models["target"].eval()
        for parameter in self.models["target"].parameters():
            parameter.requires_grad_(False)

        optimizer_config = self.config.get(
            "optimizers",
            {"name": "adam", "lr": 1e-3},
        )
        if not isinstance(optimizer_config, dict):
            raise TypeError("optimizers must be a flat mapping")
        optimizer_config = dict(optimizer_config)
        optimizer_name = str(optimizer_config.pop("name", "adam"))
        trainable_parameters = [
            parameter
            for parameter in self.models["online"].parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("DQN q_network has no trainable parameters")
        self.optimizers["q_network"] = OPTIMIZER_REGISTRY.get(
            optimizer_name,
            trainable_parameters,
            **optimizer_config,
        )
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            observation_shape=observation_shape,
            action_shape=(),
            observation_dtype=observation_dtype,
            action_dtype=np.int64,
            seed=self.seed,
        )
        self.epsilon = self.epsilon_start
        self.random_generator = np.random.default_rng(self.seed)

    def select_action(self, observation: Any, *, deterministic: bool) -> int:
        """Select an epsilon-greedy discrete action."""
        return self._select_action(observation, deterministic=deterministic)

    def _select_action(self, observation: Any, *, deterministic: bool) -> int:
        output = self._select_actions(
            np.expand_dims(np.asarray(observation), axis=0),
            deterministic=deterministic,
        )
        return output.actions[0]

    def select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        """Select epsilon-greedy actions with one batched Q-network call."""
        return self._select_actions(observations, deterministic=deterministic)

    def _select_actions(
        self,
        observations: Any,
        *,
        deterministic: bool,
    ) -> BatchActionOutput[int]:
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        observation_batch = np.asarray(observations)
        if observation_batch.shape[0] == 0:
            raise ValueError("DQN action selection requires at least one observation")
        if any(
            not observation_space.contains(observation)
            for observation in observation_batch
        ):
            raise ValueError("Observation is outside the configured space")

        batch_size = int(observation_batch.shape[0])
        explore = np.zeros(batch_size, dtype=np.bool_)
        if not deterministic:
            explore = self.random_generator.random(batch_size) < self.epsilon
        action_indices = np.empty(batch_size, dtype=np.int64)
        action_indices[explore] = self.random_generator.integers(
            action_space.n,
            size=int(explore.sum()),
        )
        greedy = ~explore
        if greedy.any():
            was_training = self.models["online"].training
            try:
                self.models["online"].eval()
                with torch.no_grad(), self.accelerator.autocast_context():
                    q_values = self._q_values(
                        self.models["online"],
                        self._observations_to_tensor(observation_batch[greedy]),
                    )
            finally:
                self.models["online"].train(was_training)
            action_indices[greedy] = (
                torch.argmax(q_values, dim=1).detach().cpu().numpy()
            )
        return BatchActionOutput(
            actions=[
                int(action_index) + int(action_space.start)
                for action_index in action_indices
            ]
        )

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float] | None:
        """Store one transition and run scheduled replay updates."""
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
        """Insert a vector step and run every crossed DQN update cycle."""
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

        previous_global_step = self.global_step - transitions.size
        previous_replay_size = len(self.replay_buffer)
        self.replay_buffer.add_batch(transitions)
        decay_fraction = min(self.global_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + (
            (self.epsilon_end - self.epsilon_start) * decay_fraction
        )
        first_ready_step = max(
            previous_global_step + 1,
            self.learning_starts,
            previous_global_step + max(self.batch_size - previous_replay_size, 1),
        )
        first_update_step = (
            (first_ready_step + self.train_frequency - 1)
            // self.train_frequency
            * self.train_frequency
        )
        update_logs: list[dict[str, float]] = []
        last_scheduled_step = previous_global_step
        for scheduled_step in range(
            first_update_step,
            self.global_step + 1,
            self.train_frequency,
        ):
            if (
                (scheduled_step - 1) // self.target_update_frequency
                > last_scheduled_step // self.target_update_frequency
            ):
                self._synchronize_target_network()
            losses: list[float] = []
            q_means: list[float] = []
            target_means: list[float] = []
            for _ in range(self.gradient_steps):
                batch = self.replay_buffer.sample(
                    self.batch_size,
                    self.accelerator.get_device(),
                )
                observations = self._observations_to_tensor(batch.observations)
                next_observations = self._observations_to_tensor(
                    batch.next_observations
                )
                action_indices = batch.actions.long() - int(action_space.start)
                with self.accelerator.autocast_context():
                    current_q_values = self._q_values(
                        self.models["online"],
                        observations,
                    ).gather(1, action_indices.reshape(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        target_next_q_values = self._q_values(
                            self.models["target"],
                            next_observations,
                        )
                        if self.double_dqn:
                            online_next_actions = torch.argmax(
                                self._q_values(
                                    self.models["online"],
                                    next_observations,
                                ),
                                dim=1,
                                keepdim=True,
                            )
                            next_q_values = target_next_q_values.gather(
                                1,
                                online_next_actions,
                            ).squeeze(1)
                        else:
                            next_q_values = target_next_q_values.max(dim=1).values
                        targets = batch.rewards + (
                            self.gamma
                            * (~batch.terminated).float()
                            * next_q_values
                        )

                    loss = functional.smooth_l1_loss(current_q_values, targets)
                self.optimizers["q_network"].zero_grad(set_to_none=True)
                self.accelerator.backward(loss, self.models["online"])
                self.accelerator.optimizer_step(
                    self.optimizers["q_network"],
                    self.models["online"],
                )
                losses.append(float(loss.detach().item()))
                q_means.append(float(current_q_values.detach().mean().item()))
                target_means.append(float(targets.detach().mean().item()))
            update_logs.append(
                {
                    "dqn/loss": float(np.mean(losses)),
                    "dqn/q_mean": float(np.mean(q_means)),
                    "dqn/target_q_mean": float(np.mean(target_means)),
                    "dqn/epsilon": self.epsilon,
                    "dqn/replay_size": float(len(self.replay_buffer)),
                }
            )
            if scheduled_step % self.target_update_frequency == 0:
                self._synchronize_target_network()
            last_scheduled_step = scheduled_step

        if (
            self.global_step // self.target_update_frequency
            > last_scheduled_step // self.target_update_frequency
        ):
            self._synchronize_target_network()
        return update_logs

    def _synchronize_target_network(self) -> None:
        self.accelerator.unwrap_model(self.models["target"]).load_state_dict(
            self.accelerator.unwrap_model(self.models["online"]).state_dict()
        )

    def _observations_to_tensor(self, observations: Any) -> torch.Tensor:
        tensor = torch.as_tensor(observations, device=self.accelerator.get_device())
        if isinstance(self.environment.observation_space, Discrete):
            indices = tensor.long() - int(self.environment.observation_space.start)
            return functional.one_hot(
                indices,
                num_classes=self.environment.observation_space.n,
            ).float()
        return tensor.float()

    def _q_values(self, model: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        output = model(observations)
        if isinstance(output, dict):
            output = output.get("q_values")
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise TypeError(
                "DQN models must return [batch, actions] tensors or {'q_values': tensor}"
            )
        if output.shape[0] != observations.shape[0]:
            raise ValueError("DQN model output batch dimension does not match its input")
        if output.shape[1] != self.environment.action_space.n:
            raise ValueError("DQN model action dimension does not match the environment")
        if not output.is_floating_point():
            raise TypeError("DQN model Q-values must use a floating-point dtype")
        return output

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return exploration and optional replay state."""
        return {
            "epsilon": self.epsilon,
            "random_generator_state": self.random_generator.bit_generator.state,
            "replay_buffer": (
                self.replay_buffer.state_dict()
                if self.checkpoint_replay_buffer
                else None
            ),
            "checkpoint_replay_buffer": self.checkpoint_replay_buffer,
            "observation_space": repr(self.environment.observation_space),
            "action_space": repr(self.environment.action_space),
        }

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exploration and optional replay state."""
        if not isinstance(state, dict):
            raise TypeError("Checkpoint algorithm state must be a mapping")
        if state.get("observation_space") != repr(self.environment.observation_space):
            raise ValueError("Checkpoint observation space does not match")
        if state.get("action_space") != repr(self.environment.action_space):
            raise ValueError("Checkpoint action space does not match")
        epsilon = float(state.get("epsilon", -1.0))
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("Checkpoint epsilon must be finite and in [0, 1]")
        generator_state = state.get("random_generator_state")
        if not isinstance(generator_state, dict):
            raise ValueError("Checkpoint exploration generator state is invalid")
        if bool(state.get("checkpoint_replay_buffer")) != self.checkpoint_replay_buffer:
            raise ValueError("Checkpoint replay-buffer policy does not match config")
        replay_state = state.get("replay_buffer")
        if self.checkpoint_replay_buffer:
            if not isinstance(replay_state, dict):
                raise ValueError("Checkpoint replay buffer state is missing")
            self.replay_buffer.load_state_dict(replay_state)
        self.epsilon = epsilon
        self.random_generator.bit_generator.state = generator_state
