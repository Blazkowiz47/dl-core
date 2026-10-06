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
      iterations: 20

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
2. resolves presets and expands the grid across seeds
3. for each run, applies fixed defaults to the base config, then explicit
   sweep-level accelerator and executor choices, then grid values (grid wins)
4. resolves a concrete run name using `tracking.run_name_template` when present,
   otherwise the grid values and seed; final names must be unique, and template
   authors choose which fields define a run's identity
5. writes concrete configs next to the sweep file under `experiments/<sweep_name>/`

Unique names prevent config and artifact collisions without requiring an
explicit `runtime.name` in the scaffold. A fresh execution asks for confirmation
before replacing existing sweep data; `--overwrite` permits an intentional
non-interactive replacement. `--preview` writes nothing, while `--dry-run`
writes generated run YAML but does not replace the tracker. Resume uses the
stored run names to keep the original selection and tracker rows even if grid
order changes.

During execution, the local sweep path also writes:

- `experiments/<sweep_name>/sweep_tracking.json`
- `artifacts/sweeps/<sweep>/<run>/final/metrics/summary.json`
- `artifacts/sweeps/<sweep>/<run>/final/metrics/history.json`
- `artifacts/sweeps/<sweep>/<run>/final/run_info.json`

No `latest` symlink is created under `artifacts/sweeps/<sweep>/`; consumers
should use the concrete run directories tracked in `sweep_tracking.json`.

That local artifact contract is what powers `dl-analyze`.

## Local Execution and Stopping

The standard local execution hook owns each worker's `Popen` handle in the sweep
parent, for both sequential and parallel execution. Each run starts in an isolated
session with stdin disconnected. This keeps the terminal's first Ctrl-C from
reaching training workers before the user selects runs to stop.

The supervisor polls process output and terminal input every 50 ms. Signal
handlers only record requests. Menu reads use nonblocking stdin, with its original
mode restored before other I/O, so Ctrl-C flushing input after `select()` cannot
leave a read waiting. Stop-all remains set across counter resets and takes
precedence over Enter. During the menu, output continues to drain into each run's
`final/logs/sweep.log`, while console output and new launches pause. Displayed row
numbers are a fixed snapshot of the owned active runs and are independent of
tracker indices.

Ctrl-C opens the menu, comma-separated row numbers select runs, Enter continues,
and another Ctrl-C or `all` stops every owned run. Invalid selections have no
effect. EOF and noninteractive Ctrl-C use stop-all. The selection sequence resets
after Enter or after selected runs finish shutting down.

A stop request keeps the tracker row claimed as `running` until termination is
confirmed. The supervisor signals the run's process group and known descendants,
including descendants that created separate sessions. It first sends SIGINT,
then SIGTERM after 10 seconds, and SIGKILL after another 3 seconds. If termination
is still unconfirmed after 2 more seconds, the result is `unknown` and remains
ineligible for automatic retry. Confirmed user stops become `stopped`; completed
results stay completed and unstarted jobs remain pending.

Process handles, pipes, signal handlers and local tracker lifecycle are cleaned
up on normal completion, launch errors, output errors, and interrupts. The same
process ownership applies to `dl-run`, whose first Ctrl-C stops its single run
directly.

Subclasses that customize `build_command()` keep this supervision. Existing
`execute_run()` overrides, including wrappers calling `super()`, retain the
shared sequential or process-pool dispatch so their hooks and returned results
are honored. The selective menu is unavailable on that path; the custom hook
controls its execution and cleanup.

## Local Resume Modes

| Flag | Claimable statuses |
| --- | --- |
| `--resume` | `pending` |
| `--resume-failed` | `failed` |
| `--resume-stopped` | `stopped` |
| `--resume-all` | `pending`, `failed`, `stopped` |

These modes reuse the tracker, original run selection, run names, and tracking
context. They are mutually exclusive and cannot be combined with `--overwrite`.
The effective executor is resolved before filtering so nonlocal executors keep
their existing resume behavior. Its name is reused during dispatch. Mixed executor
names in the expanded grid are rejected before filtering unless `--executor local`
explicitly selects local execution for all runs. The selected statuses are checked
again under the tracker lock immediately before a run is claimed, preventing
overlapping commands from launching the same run.

Only runs claimed by the current invocation contribute to the local command's
outcome. Historical statuses are reported separately. Completion and empty
selections return 0, current failures return 1, current unknown results return 3,
and stop-all returns 130. Selective stops are reported separately from failures.

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
