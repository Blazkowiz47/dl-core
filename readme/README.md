# Welcome to the `dl-core` Documentation

This documentation is split into quick references, workflow guides, and
technical notes. The goal is the same as in the original framework repo, but
focused on the extracted package and the experiment-repo workflow around it.

Current public release: `deep-learning-core==0.1.4`.
Current development version: `0.1.4`.

## What's New in Development?

- tar datasets now use optional WebDataset pipelines for grouped streaming,
  buffered shuffling, resampling, and distributed shard splitting
- the custom sidecar index, tar handle pool, and round-robin tar sampler have
  been removed

## What's New in 0.1.4?

- `IterationTrainer` supports fixed-batch training, deterministic finite-loader
  cycling, iteration-based lifecycle frequencies, and exact cursor resume
- callers now choose `EpochTrainer`, `IterationTrainer`, or `RLTrainer`
  explicitly; the ambiguous `BaseTrainer` alias is removed
- indexed tar datasets read grouped samples directly from uncompressed archives
  and reuse worker-local handles without extracting files
- deterministic round-robin tar batches preserve group balance and partition
  complete batches across distributed ranks

Previous versions are recorded in the [release history](../RELEASES.md).

## Companion Packages

- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure): Azure execution and
  Azure dataset foundations
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow): local MLflow
  integration
- [`dl-robotics`](https://github.com/Blazkowiz47/dl-robotics): fast 2D MAPF
  environments, episode metrics, and GIF/MP4 artifacts
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb): Weights & Biases
  integration

## Structure

### 1. [`tldr/`](./tldr/1_install_and_verify.md)

Go here if you want the shortest path to a working setup.

- [Install and Verify](./tldr/1_install_and_verify.md)
- [Create and Run an Experiment](./tldr/2_create_and_run_an_experiment.md)

### 2. [`guide/`](./guide/1_getting_started.md)

Go here if you want a workflow-oriented explanation of how `dl-core` is meant
to be used from an experiment repository.

- [Getting Started](./guide/1_getting_started.md)
- [Creating an Experiment Repository](./guide/2_creating_an_experiment_repository.md)
- [Local Components and Sweeps](./guide/3_local_components_and_sweeps.md)

### 3. [`technical/`](./technical/1_configuration.md)

Go here if you need the package internals, config reference, or extension
mechanics.

- [Configuration](./technical/1_configuration.md)
- [Entry Points](./technical/2_entry_points.md)
- [Sweep System](./technical/3_sweep_system.md)
- [Local Component Loading](./technical/4_local_component_loading.md)
- [Testing](./technical/5_testing.md)
- [Reinforcement Learning](./technical/6_reinforcement_learning.md)

## Common Tasks

### Verify the package

```bash
uv run dl-core list
```

### Scaffold a new experiment repository

```bash
uv run dl-init --name my-exp --root-dir .
```

To scaffold the current directory in place:

```bash
uv run dl-init --root-dir .
```

### Run a local training job

```bash
uv run dl-run --config configs/base.yaml --validate-only
cp configs/base.yaml experiments/debug.yaml
uv run dl-run --config experiments/debug.yaml --validate-only
uv run dl-run --config experiments/debug.yaml
```

Generated `configs/base.yaml` files include root-level `seed` and
`deterministic` defaults so reproducibility can be controlled explicitly. Keep
concrete single-run configs under `experiments/`, including debug runs.

### Smoke-check generated dataset and model helpers

```bash
uv run python scripts/temporary/test_dataset.py
uv run python scripts/temporary/test_model.py
```

### Run a sweep

```bash
uv run dl-sweep experiments/lr_sweep.yaml
```
