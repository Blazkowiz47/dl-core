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

An RL config can be resolved without stepping the environment or running an
update:

```bash
dl-run --config experiments/sac.yaml --validate-only
```

The preflight creates the environment, algorithm models, optimizers, and
callbacks in a temporary artifact directory, reports their resolved types and
spaces, and closes the runtime immediately.

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

Training collection can use a Gymnasium vector environment. The built-in
adapter creates one with same-step autoreset so completed transitions retain
their real `final_obs` while collection immediately receives the next episode's
initial observation:

```yaml
environment:
  name: gymnasium_vector
  id: CartPole-v1
  num_envs: 8
  vectorization_mode: sync

evaluation_environment:
  name: gymnasium
  id: CartPole-v1
```

`global_step` counts transitions, while `collector_step` counts vector
environment calls. `select_actions` and `process_transition_batch` expose one
whole collector step to vector-aware algorithms. Existing custom trainers that
only implement `select_action` and `process_transition` remain compatible
through stable environment-index dispatch. Evaluation deliberately uses one
scalar environment so videos and deterministic episode artifacts have
unambiguous identity. A transition budget can overshoot by at most
`num_envs - 1` because a vector step is atomic. An episode budget has the same
maximum overshoot when several lanes complete in one vector step.

Tabular Q-learning consumes vector steps in stable lane order. DQN performs at
most one batched action-selection inference per vector step, inserts the complete
transition batch into replay, and applies every update or target-sync boundary
crossed by that atomic step in schedule order. PPO performs batched policy and
next-value inference, stores independent `[time, environment]` streams, and
flattens them only after per-lane GAE is complete. `rollout_steps` counts
synchronized collector calls, so an update uses
`rollout_steps * num_envs` samples. SAC samples continuous actions in one actor
call, inserts vector transitions directly into replay, and runs the gradient
work for every update-frequency boundary crossed by the atomic vector step.
Each crossed boundary emits its own update log, and warm-up is applied to the
first `learning_starts` transitions even when that boundary falls inside a
vector step.

Replay storage accepts transition batches directly and preserves configured
observation/action dtypes. PPO rollout storage is preallocated as
`[time, environment, ...]`; generalized advantages are propagated only within
the same environment stream before the rollout is flattened for minibatches.

RL callbacks can implement `on_episode_start`, `on_episode_end`,
`on_update_end`, and `on_evaluation_end`. The existing run-level
`on_training_start`, `on_training_end`, and `on_training_finalized` hooks remain
shared with epoch training.

## Episode Managers

Episode managers are the reinforcement-learning counterpart to the metric
managers used by `EpochTrainer`. They accumulate environment transitions,
compute episode summaries, and optionally persist complete trajectories while
callbacks remain responsible for external logging and side effects.

The built-in `standard` manager always computes return, length, reward
statistics, termination, truncation, and success when the environment exposes
`is_success`. Complete trajectories preserve the initial observation followed
by one observation per transition, so an episode of length `T` contains `T + 1`
observations and `T` actions, rewards, termination flags, and truncation flags.

```yaml
episode_managers:
  standard:
    capture_phases: [evaluation]
    capture_every_n_episodes: 10
    max_captured_episodes: 20
    info_keys: [is_success, collision]
    capture_action_info: false
```

Captured array-valued trajectories are portable compressed NumPy archives under
`final/episodes/<phase>/`; the episode index and scalar summary streams remain
JSONL. Array dtypes are preserved and arbitrary Python objects are not pickled.
Environments with array-valued observations and actions, including nested
dictionaries and tuples, can persist their environment-boundary trajectory.
Exact hidden simulator state is available only when a concrete environment
provides a separate state snapshot capability.

Episode managers use the normal component workflow:

```bash
dl-core list episode_manager
dl-core describe episode_manager standard
dl-core add episode_manager PathAnalysis
```

The built-in local metric tracker and the MLflow and W&B companion callbacks
record episode, algorithm-update, and evaluation metrics as well as supervised
epoch metrics. Training-episode series stay separate from the aggregate metrics
reported by evaluation groups.

Custom algorithms can start from the episode lifecycle scaffold:

```bash
dl-core add trainer MyPolicy --base rltrainer
```

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

`PPOTrainer` is registered as `ppo`. It supports `Discrete` and finite,
floating-point `Box` actions with `Discrete` or `Box` observations. Integer and
boolean `Box` actions are rejected because they do not define a continuous
policy. The built-in
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
Jacobian cancels in the old/new probability ratio. The reported entropy and
entropy bonus use the pre-squash Gaussian entropy.

Custom policies receive `Box` observations in their original batched shape and
`Discrete` observations as one-hot batches. They must return a floating-point
`value` tensor shaped `[batch]`, plus either `logits` shaped `[batch, actions]`
or continuous `mean` and `log_std` tensors shaped `[batch, action_dimensions]`.
Policy modules should avoid dropout and other stochastic training-mode layers,
because PPO must reproduce the behavior-policy probability for stored actions.

PPO accepts scalar or vector training environments. Vector collection evaluates
all lanes in one policy call, computes GAE independently along each environment
stream, and flattens the time/environment axes only for minibatch optimization.
`rollout_steps` counts synchronized collector steps, so a full vector rollout
contains `rollout_steps * num_envs` training samples. Evaluation remains scalar
to preserve deterministic episode-level reporting.

Partial rollouts are stored in checkpoints. Because environment state is not
part of an RL checkpoint, resuming marks the final stored transition in every
unfinished lane as truncated. Its one-step value target is retained, while GAE
cannot propagate from the newly reset environment into the earlier rollout
fragment.

## Soft Actor-Critic

`SACTrainer` is registered as `sac`. It supports finite, floating-point `Box`
actions with `Discrete` or `Box` observations. Its built-in
`sac_gaussian_actor` uses a state-dependent diagonal Gaussian, while
`sac_twin_q_network` maintains two independent observation-action value
estimates to reduce overestimation bias.

```yaml
models:
  actor:
    name: sac_gaussian_actor
    hidden_sizes: [256, 256]
  critics:
    name: sac_twin_q_network
    hidden_sizes: [256, 256]

optimizers:
  actor:
    name: adam
    lr: 0.0003
  critics:
    name: adam
    lr: 0.0003
  temperature:
    name: adam
    lr: 0.0003

trainer:
  sac:
    total_timesteps: 1000000
    gamma: 0.99
    buffer_size: 1000000
    batch_size: 256
    learning_starts: 5000
    train_frequency: 1
    gradient_steps: 1
    tau: 0.005
    initial_alpha: 0.2
    automatic_entropy_tuning: true
    target_entropy: null
    log_std_min: -20.0
    log_std_max: 2.0
    checkpoint_replay_buffer: true
```

Before `learning_starts`, SAC samples uniformly within the action bounds. It
then uses reparameterized Gaussian actions followed by a tanh transform and
affine scaling into the environment bounds. The policy objective includes the
full transformed-action log density, including the tanh Jacobian and action
scale. Gaussian density and transform calculations stay in float32 under mixed
precision to avoid underflow at small standard deviations. The default entropy
target is the negative flattened action dimension; set
`automatic_entropy_tuning: false` to keep `initial_alpha` fixed.

The replay target uses the lower target-critic estimate. True termination
removes the bootstrap term, while truncation retains it. Target critics receive
a Polyak update after every replay gradient step. Training and evaluation spaces
must match exactly, and unbounded, integer, or boolean action spaces are
rejected. A flat optimizer mapping can be used to share one optimizer type and
configuration across the actor, critics, and learned temperature.

Custom actors receive the same observation batches as PPO and must return
floating-point `mean` and `log_std` tensors shaped
`[batch, action_dimensions]`. Custom twin critics receive observation and
bounded action batches and must return floating-point `q1` and `q2` tensors
shaped `[batch]`. All outputs must be finite. Custom actors should avoid dropout
and other stochastic training-mode layers because their extra randomness is not
part of the reported Gaussian density. Replay contents, both sampling
generators, and continuation-sensitive SAC settings are checkpointed and
validated by default; disabling replay checkpointing reduces checkpoint size
but resumes with empty replay memory. Gradient accumulation is currently
rejected because SAC performs distinct critic, actor, and temperature optimizer
steps in each replay update.
