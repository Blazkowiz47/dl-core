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
`Discrete` or `Box` observations, including non-zero `Discrete` starts. The
built-in `dqn_mlp` model flattens `Box` observations; image-shaped observations
can instead use a registered custom Q-network returning a floating-point
`[batch, actions]` tensor or `{"q_values": tensor}`. Structured `Dict` and
`Tuple` observation spaces are not currently supported. Custom networks receive
`Box` batches in their original shape and `Discrete` observations as one-hot
batches.

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
truncation retains it. Training and evaluation action and observation spaces
must match exactly, including `Box` bounds and dtypes. Target synchronization is
scheduled by environment transitions even when a synchronization step falls
between replay updates, and model forwards honor the configured accelerator's
autocast context. Replay sampling and epsilon exploration use separately
checkpointed generators. Saving replay memory makes checkpoints larger but
allows exact off-policy continuation; set `checkpoint_replay_buffer: false` to
resume with an empty buffer. Gradient accumulation is currently rejected for
DQN because each replay update is an independent optimizer step.

## Proximal Policy Optimization

`PPOTrainer` is registered as `ppo`. It supports `Discrete` and finite `Box`
actions with `Discrete` or `Box` observations. The built-in
`ppo_actor_critic` model uses a shared MLP encoder, categorical logits for
discrete actions, and a diagonal Gaussian for continuous actions.

```yaml
models:
  policy:
    name: ppo_actor_critic
    hidden_sizes: [64, 64]

optimizers:
  name: adam
  lr: 0.0003

trainer:
  ppo:
    total_timesteps: 1000000
    gamma: 0.99
    gae_lambda: 0.95
    rollout_steps: 2048
    update_epochs: 10
    minibatch_size: 64
    clip_range: 0.2
    value_clip_range: 0.2
    value_loss_coefficient: 0.5
    entropy_coefficient: 0.01
    normalize_advantages: true
```

GAE stops recursive propagation at both termination and truncation boundaries,
but the one-step value target continues to bootstrap across truncation. PPO
collects across episode boundaries and updates at the configured rollout length
or when the final training budget is reached, so no final partial rollout is
discarded. Continuous actions use a tanh transform into the environment bounds;
PPO stores the corresponding raw Gaussian action because the fixed transform
Jacobian cancels in the old/new probability ratio.

The initial PPO implementation collects a single environment stream. This keeps
episode callbacks and deterministic checkpoint continuation identical to the
other trainers. A later vector-environment collector can feed the same policy
and rollout contracts without changing the public trainer configuration.
