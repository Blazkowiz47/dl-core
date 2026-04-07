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
      epochs: 1

default_grid:
  optimizers.lr: [1e-4, 5e-4]

tracking:
  # experiment_name: my_project
  # Optional tracker destination override. Defaults to the repository root name.
  # sweep_name: my_custom_sweep
  # Optional sweep grouping override. Defaults to the sweep filename.
  run_name_template: "lr_{optimizers.lr}"

seeds: [2025]
```

## User Sweeps

The normal user file extends the base template:

```yaml
extends_template: "../configs/base_sweep.yaml"
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
5. writes concrete configs next to the sweep file under `experiments/<sweep_name>/`
6. resolves a concrete run name for each generated config
   using `tracking.run_name_template` when present, otherwise the config filename stem

That last step is important because it prevents artifact collisions between run
directories without requiring an explicit `runtime.name` in the scaffold.

During execution, the local sweep path also writes:

- `experiments/<sweep_name>/sweep_tracking.json`
- `artifacts/sweeps/<sweep>/<run>/final/metrics/summary.json`
- `artifacts/sweeps/<sweep>/<run>/final/metrics/history.json`
- `artifacts/sweeps/<sweep>/<run>/final/run_info.json`

That local artifact contract is what powers `dl-analyze`.

## Tracking Metadata

Sweep templates support a `tracking` block used for:

- tracker experiment destination overrides
- run name templates
- sweep grouping names
- description templates
- auto-generated tags

The tracking block is deliberately backend-neutral in `dl-core`. Backend
adapters can consume it however they need.

## Local Analysis

For local runs, `dl-core` does not depend on MLflow or W&B to analyze a sweep.
Instead, each run writes normalized summary and history files into its artifact
directory, and the sweep tracker records where those files live.

That means local analysis is always:

```bash
uv run dl-analyze --sweep experiments/lr_sweep.yaml
```

You can also rank explicitly by one or more metrics:

```bash
uv run dl-analyze --sweep experiments/lr_sweep.yaml \
  --metric test/eer --mode min \
  --metric test/accuracy --mode max \
  --rank-method rank-sum
```

`dl-analyze` supports three ranking modes:

- `lexicographic`: rank by metric 1, then metric 2 as a tie-breaker, and so on
- `rank-sum`: rank each metric independently, sum the ranks, and sort by the total
- `pareto`: group runs by Pareto front instead of forcing a single scalar score

Cloud-specific adapters can override how metrics are fetched later, but the
default analyzer stays file-based and local-first. Azure-backed analysis only
fetches the requested metric histories for ranking.
