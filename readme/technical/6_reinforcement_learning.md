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

Training and evaluation environments are separate instances. Evaluation uses
deterministic action selection and a distinct seed range, so evaluation does not
consume training-environment state. Checkpoints retain common counters, model,
optimizer, scheduler, callback, random-generator, metric-history, and
algorithm-specific state. Environment simulator state is not serialized; a
resumed run begins at a new episode boundary.

RL callbacks can implement `on_episode_start`, `on_episode_end`,
`on_update_end`, and `on_evaluation_end`. The existing run-level
`on_training_start`, `on_training_end`, and `on_training_finalized` hooks remain
shared with epoch training.

The initial RL runtime supports CPU and single-GPU algorithms. Distributed
environment collection is rejected explicitly until its synchronization and
sampling semantics are defined.
