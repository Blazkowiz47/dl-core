# Technical: 3. Sweep System

The sweep system builds concrete run configs from a base config plus a template
or user sweep file.

## Base Sweep Template

The bundled base template looks like this conceptually:

```yaml
template_name: my_experiment_sweep
base_config: ./base.yaml

fixed:
  accelerator: preset:accelerators.cpu
  executor: preset:executors.local
  runtime:
    log_level: INFO
  trainer:
    my_exp:
      name: my_exp
      epochs: 1

default_grid:
  optimizers.lr: [1e-4, 5e-4]

tracking:
  group: my_experiment
  run_name_template: "lr_{optimizers.lr}"

seeds: [2025]
```

## User Sweeps

The normal user file extends the base template:

```yaml
extends_template: "../base_sweep.yaml"
description: "Basic learning rate sweep"

grid:
  optimizers.lr: [0.001, 0.0001]
```

## Generated Output

When you run a sweep, `dl-core`:

1. loads the base config
2. resolves presets
3. applies fixed parameters
4. expands the grid across seeds
5. writes concrete configs under `configs/sweeps/<sweep_name>/`
6. sets `runtime.name` per generated run

That last step is important because it prevents artifact collisions between run
directories.

## Tracking Metadata

Sweep templates support a `tracking` block used for:

- run name templates
- group names
- description templates
- auto-generated tags

The tracking block is deliberately backend-neutral in `dl-core`. Backend
adapters can consume it however they need.
