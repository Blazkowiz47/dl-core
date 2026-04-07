# Technical: 2. Entry Points

`dl-core` exposes several console entrypoints.

## `dl-core`

Creates local component scaffolds inside an experiment repository and lists or
describes registered components.

```bash
uv run dl-core list
uv run dl-core list sampler
uv run dl-core add augmentation Custom1
uv run dl-core add dataset LocalDataset
uv run dl-core add dataset FrameDataset --base frame
uv run dl-core add callback EpochLogger --base metric_logger
uv run dl-core add optimizer AdamwWrapper --base adamw
```

Useful flags:

- `--json` on `dl-core list`
- `--root-dir` on `dl-core list`
- `--base` for selecting a non-default scaffold base
- `--root-dir`
- `--force`

Dataset scaffold bases:

- plain `dl-core`: `base`, `frame`, `text_sequence`, `adaptive_computation`
- with `dl-azure`: `azure_compute`, `azure_streaming`,
  `azure_compute_frame`, `azure_streaming_frame`,
  `azure_compute_multiframe`, `azure_streaming_multiframe`

Non-dataset component scaffolds can also use `--base` with a registered
component name such as `metric_logger`, `standard`, `adamw`, or `cosine`,
or with a fully qualified class path.

## `dl-init`

Creates a standalone experiment repository scaffold.

```bash
uv run dl-init --name my-exp --root-dir .
```

To initialize the target directory itself instead of creating a nested
subdirectory, omit `--name`:

```bash
uv run dl-init --root-dir .
```

Important arguments:

- `--name` (optional)
- `--root-dir`
- dynamically added extension flags such as `--with-azure`, `--with-mlflow`,
  and `--with-wandb` when the relevant extra package is installed

`dl-init-experiment` remains available as a compatibility alias for older
scripts.

## `dl-run`

Runs a single local training job through the local executor.

```bash
uv run dl-run --config configs/base.yaml
```

Useful flags:

- `--show-registry`
- `--validate-only`
- `--dry-run`
- `--log-level`

Notes:

- `--validate-only` now performs a lightweight preflight: it validates the
  config, resolves the configured components, safely instantiates the dataset,
  models, criterions, optimizer, and optional scheduler, then exits without
  starting training

## `dl-sweep`

Generates run configs from a sweep spec and dispatches them through the
configured executor.

```bash
uv run dl-sweep experiments/lr_sweep.yaml
```

Useful flags:

- `--dry-run`
- `--resume`
- `--max-workers`
- `--compute`
- `--environment`

## `dl-analyze`

Reads local or backend-provided sweep tracking and writes a compact analysis
report.

```bash
uv run dl-analyze --sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml --name pareto_eer
uv run dl-analyze --sweep experiments/lr_sweep.yaml --metric test/eer --mode min
uv run dl-analyze --sweep experiments/lr_sweep.yaml \
  --metric test/eer --mode min \
  --metric test/accuracy --mode max \
  --rank-method rank-sum
```

Useful flags:

- `--json`
- `--force`
- `--name`
- `--metric`
- `--mode`
- `--rank-method`

Notes:

- ranking defaults to `test/accuracy` with `max`
- `--metric` and `--mode` are repeatable and matched by order
- fetched remote metric histories are cached in `analysis_cache.json`
- reports are written under `analysis/v1.md`, `analysis/v2.md`, and so on
- supported rank methods are `lexicographic`, `rank-sum`, and `pareto`

## `dl-train-worker`

Internal worker entrypoint used by executors. This is the direct trainer
process, not the normal user-facing entrypoint.

```bash
uv run dl-train-worker --config configs/base.yaml
```

In normal usage, prefer `dl-run` or `dl-sweep`.
