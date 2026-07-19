# Technical: 6. Reinforcement Learning

Reinforcement learning is a first-class part of `deep-learning-core`. The RL
trainer hierarchy is episode-driven and remains separate from the
dataset-driven `EpochTrainer` hierarchy.

## Environment Contract

RL environments use the Gymnasium API:

```python
observation, info = environment.reset(seed=42)
observation, reward, terminated, truncated, info = environment.step(action)
environment.close()
```

The `Environment` protocol accepts native Gymnasium environments and compatible
third-party implementations. `Transition` and `EpisodeResult` provide common
value objects for algorithm implementations.

## Environment Registry

The built-in `gymnasium` adapter creates an environment from its public ID:

```python
from dl_core.environments import make_environment

environment = make_environment(
    {
        "name": "gymnasium",
        "id": "FrozenLake-v1",
        "kwargs": {"is_slippery": False},
    }
)
```

Extension packages and experiment repositories can register compatible
environments without changing the trainer implementation:

```python
from dl_core.core import register_environment

@register_environment("custom_environment")
class CustomEnvironment:
    ...
```

Registered environments participate in the standard discovery commands:

```bash
dl-core list environment
dl-core describe environment gymnasium
```

Robotics-specific worlds, robots, sensors, and physics backends are deliberately
outside the core environment contract and can be layered on through a future
companion package.

## RL Trainer Lifecycle

`RLTrainer` is a sibling of `EpochTrainer`. It owns episode and environment-step
counters, independent training and evaluation environments, callback dispatch,
artifact persistence, and resumable checkpoints. Algorithm implementations
provide action selection, transition processing, and their additional state.

The common trainer configuration is episode-driven:

```yaml
environment:
  name: gymnasium
  id: FrozenLake-v1
  kwargs:
    is_slippery: false

trainer:
  q_learning:
    total_timesteps: 20000
    max_episode_steps: 200
    evaluation_frequency: 20
    evaluation_episodes: 5
    checkpoint_frequency: 100
```

Set a positive `total_timesteps`, `max_episodes`, or both. When both are set,
training stops at the first budget reached.

Training and evaluation environments are separate instances. Evaluation uses
deterministic action selection, evaluation model mode, and a distinct seed
range, so evaluation does not consume training-environment state. Set
`evaluation_episodes: 0` to disable evaluation. Checkpoints retain the trainer
identity, common counters, model, optimizer, scheduler, accelerator, callback,
random-generator, metric-history, and algorithm-specific state. Environment
simulator state is not serialized; a resumed run begins at a new episode
boundary. A checkpoint is rejected when its trainer implementation or component
names do not match the configured trainer.

RL callbacks can implement `on_episode_start`, `on_episode_end`,
`on_update_end`, and `on_evaluation_end`. The existing run-level
`on_training_start`, `on_training_end`, and `on_training_finalized` hooks remain
shared with epoch training.

The initial RL runtime supports CPU and single-GPU algorithms. Distributed
environment collection is rejected explicitly until its synchronization and
sampling semantics are defined.

## Tabular Q-Learning

`QLearningTrainer` is registered as `q_learning`. It is intended for finite
`Discrete` observation and action spaces and does not create a PyTorch model.
The trainer applies the standard one-step update:

```text
Q(s, a) <- Q(s, a) + learning_rate *
    (reward + gamma * max(Q(next_state, :)) - Q(s, a))
```

The next-state term is omitted only for true environment termination. A
truncation, including a configured episode time limit, retains bootstrapping.
This distinction follows the Gymnasium termination contract.

```yaml
trainer:
  q_learning:
    total_timesteps: 20000
    max_episode_steps: 200
    learning_rate: 0.1
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.05
    epsilon_decay_steps: 10000
```

Exploratory actions and greedy tie-breaking use a trainer-owned random generator
whose state is included in checkpoints. Deterministic evaluation always chooses
the lowest-index maximizing action. Non-zero starts on Gymnasium `Discrete`
spaces are supported. Training and evaluation spaces must have matching sizes
and starts. Checkpoint loading validates both space definitions, the Q-table,
epsilon, and exploration-generator state before resuming.

The standard configuration validator recognizes registered `RLTrainer`
subclasses and requires `environment` in place of the supervised `dataset`,
`models`, and `optimizers` sections.

## Deep Q-Networks

`DQNTrainer` is registered as `dqn`. It supports `Discrete` actions with either
`Discrete` or `Box` observations. The built-in `dqn_mlp` model flattens vector
observations; image and structured observations should use a registered custom
Q-network returning a `[batch, actions]` tensor or `{"q_values": tensor}`.

```yaml
models:
  q_network:
    name: dqn_mlp
    hidden_sizes: [128, 128]

optimizers:
  name: adam
  lr: 0.001

trainer:
  dqn:
    total_timesteps: 100000
    gamma: 0.99
    buffer_size: 100000
    batch_size: 64
    learning_starts: 1000
    train_frequency: 1
    gradient_steps: 1
    target_update_frequency: 1000
    double_dqn: true
    epsilon_start: 1.0
    epsilon_end: 0.05
    epsilon_decay_steps: 50000
    checkpoint_replay_buffer: true
```

DQN uses uniform replay, a hard-updated target network, Huber loss, and Double
DQN targets by default. True termination removes the bootstrap target;
truncation retains it. Replay sampling and epsilon exploration use separately
checkpointed generators. Saving replay memory makes checkpoints larger but
allows exact off-policy continuation; set `checkpoint_replay_buffer: false` to
resume with an empty buffer. Gradient accumulation is currently rejected for
DQN because each replay update is an independent optimizer step.
