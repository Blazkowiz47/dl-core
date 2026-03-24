# Welcome to the `dl-core` Documentation

This documentation is split into quick references, workflow guides, and
technical notes. The goal is the same as in the original framework repo, but
focused on the extracted package and the experiment-repo workflow around it.

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

## Common Tasks

### Verify the package

```bash
uv run dl-run --show-registry
```

### Scaffold a new experiment repository

```bash
uv run dl-init-experiment --name my-exp --root-dir .
```

To scaffold the current directory in place:

```bash
uv run dl-init-experiment --root-dir .
```

### Run a local training job

```bash
uv run dl-run --config configs/base.yaml
```

### Run a sweep

```bash
uv run dl-sweep --sweep configs/sweeps/example_sweep.yaml
```
