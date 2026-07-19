"""Tabular Q-learning trainer for finite discrete environments."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Discrete

from dl_core.core import RLTrainer, Transition, config_field, register_trainer


@register_trainer("q_learning")
class QLearningTrainer(RLTrainer):
    """Tabular Q-learning with epsilon-greedy exploration."""

    CONFIG_FIELDS = RLTrainer.CONFIG_FIELDS + [
        config_field(
            "learning_rate",
            "float",
            "Q-value update step size.",
            default=0.1,
        ),
        config_field(
            "gamma",
            "float",
            "Discount applied to non-terminal next-state values.",
            default=0.99,
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
            "Environment transitions used for linear epsilon decay.",
            default=10000,
        ),
    ]

    def setup_algorithm(self) -> None:
        """Initialize the Q-table and exploration state."""
        if not isinstance(self.environment.observation_space, Discrete):
            raise TypeError("QLearningTrainer requires a Discrete observation space")
        if not isinstance(self.environment.action_space, Discrete):
            raise TypeError("QLearningTrainer requires a Discrete action space")
        if not isinstance(self.evaluation_environment.observation_space, Discrete):
            raise TypeError(
                "QLearningTrainer requires a Discrete evaluation observation space"
            )
        if not isinstance(self.evaluation_environment.action_space, Discrete):
            raise TypeError(
                "QLearningTrainer requires a Discrete evaluation action space"
            )
        if (
            self.evaluation_environment.observation_space.n
            != self.environment.observation_space.n
            or self.evaluation_environment.observation_space.start
            != self.environment.observation_space.start
            or self.evaluation_environment.action_space.n
            != self.environment.action_space.n
            or self.evaluation_environment.action_space.start
            != self.environment.action_space.start
        ):
            raise ValueError(
                "Training and evaluation environments must use identical "
                "Discrete observation and action spaces"
            )

        self.learning_rate = float(self.trainer_config.get("learning_rate", 0.1))
        self.gamma = float(self.trainer_config.get("gamma", 0.99))
        self.epsilon_start = float(self.trainer_config.get("epsilon_start", 1.0))
        self.epsilon_end = float(self.trainer_config.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(
            self.trainer_config.get("epsilon_decay_steps", 10000)
        )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")

        self.q_table = np.zeros(
            (
                self.environment.observation_space.n,
                self.environment.action_space.n,
            ),
            dtype=np.float64,
        )
        self.epsilon = self.epsilon_start
        self.random_generator = np.random.default_rng(self.seed)

    def select_action(self, observation: Any, *, deterministic: bool) -> int:
        """Choose an epsilon-greedy action for a discrete observation."""
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if not observation_space.contains(observation):
            raise ValueError(f"Observation is outside the configured space: {observation}")
        state_index = int(observation) - int(observation_space.start)

        if not deterministic and self.random_generator.random() < self.epsilon:
            action_index = int(self.random_generator.integers(action_space.n))
        elif deterministic:
            action_index = int(np.argmax(self.q_table[state_index]))
        else:
            best_actions = np.flatnonzero(
                self.q_table[state_index] == np.max(self.q_table[state_index])
            )
            action_index = int(self.random_generator.choice(best_actions))
        return action_index + int(action_space.start)

    def process_transition(
        self,
        transition: Transition[Any, Any],
    ) -> dict[str, float]:
        """Apply one terminal-aware tabular Q-learning update."""
        observation_space = self.environment.observation_space
        action_space = self.environment.action_space
        if not observation_space.contains(transition.observation):
            raise ValueError("Transition observation is outside the configured space")
        if not observation_space.contains(transition.next_observation):
            raise ValueError("Transition next observation is outside the configured space")
        if not action_space.contains(transition.action):
            raise ValueError("Transition action is outside the configured space")

        state_index = int(transition.observation) - int(observation_space.start)
        next_state_index = int(transition.next_observation) - int(
            observation_space.start
        )
        action_index = int(transition.action) - int(action_space.start)
        current_value = self.q_table[state_index, action_index]
        next_value = 0.0
        if not transition.terminated:
            next_value = float(np.max(self.q_table[next_state_index]))
        target = transition.reward + (self.gamma * next_value)
        td_error = target - current_value
        self.q_table[state_index, action_index] += self.learning_rate * td_error

        decay_fraction = min(self.global_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + (
            (self.epsilon_end - self.epsilon_start) * decay_fraction
        )
        return {
            "q_learning/td_error": float(td_error),
            "q_learning/q_value": float(self.q_table[state_index, action_index]),
            "q_learning/epsilon": self.epsilon,
        }

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return Q-table and exploration-generator state."""
        return {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "random_generator_state": self.random_generator.bit_generator.state,
            "observation_space": {
                "n": int(self.environment.observation_space.n),
                "start": int(self.environment.observation_space.start),
            },
            "action_space": {
                "n": int(self.environment.action_space.n),
                "start": int(self.environment.action_space.start),
            },
        }

    def load_algorithm_state_dict(self, state: dict[str, Any]) -> None:
        """Restore Q-table and exploration-generator state."""
        if not isinstance(state, dict):
            raise TypeError("Checkpoint algorithm state must be a mapping")
        expected_observation_space = {
            "n": int(self.environment.observation_space.n),
            "start": int(self.environment.observation_space.start),
        }
        expected_action_space = {
            "n": int(self.environment.action_space.n),
            "start": int(self.environment.action_space.start),
        }
        if state.get("observation_space") != expected_observation_space:
            raise ValueError(
                "Checkpoint observation space does not match the configured environment"
            )
        if state.get("action_space") != expected_action_space:
            raise ValueError(
                "Checkpoint action space does not match the configured environment"
            )
        if "q_table" not in state:
            raise ValueError("Checkpoint does not contain a Q-table")
        q_table = np.asarray(state["q_table"], dtype=np.float64)
        if q_table.shape != self.q_table.shape:
            raise ValueError(
                f"Checkpoint Q-table shape {q_table.shape} does not match "
                f"{self.q_table.shape}"
            )
        if not np.isfinite(q_table).all():
            raise ValueError("Checkpoint Q-table must contain only finite values")
        self.q_table[...] = q_table
        if "epsilon" not in state:
            raise ValueError("Checkpoint does not contain epsilon")
        epsilon = float(state["epsilon"])
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("Checkpoint epsilon must be finite and in [0, 1]")
        self.epsilon = epsilon
        if "random_generator_state" not in state:
            raise ValueError("Checkpoint does not contain random generator state")
        random_generator_state = state["random_generator_state"]
        if not isinstance(random_generator_state, dict):
            raise TypeError("Checkpoint random generator state must be a mapping")
        self.random_generator.bit_generator.state = random_generator_state
