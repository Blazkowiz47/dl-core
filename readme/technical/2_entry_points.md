# Technical: 2. Entry Points

`dl-core` exposes six console entrypoints.

## `dl-core`

Creates local component scaffolds inside an experiment repository.

```bash
uv run dl-core add augmentation Custom1
uv run dl-core add dataset LocalDataset
uv run dl-core add dataset FrameDataset --base frame
```

Useful flags:

- `--base` for dataset scaffolds
- `--root-dir`
- `--force`

Dataset scaffold bases:

- plain `dl-core`: `base`, `frame`, `text_sequence`, `adaptive_computation`
- with `dl-azure`: `azure_compute`, `azure_streaming`,
  `azure_compute_frame`, `azure_streaming_frame`,
  `azure_compute_multiframe`, `azure_streaming_multiframe`

## `dl-init-experiment`

Creates a standalone experiment repository scaffold.

```bash
uv run dl-init-experiment --name my-exp --root-dir .
```

To initialize the target directory itself instead of creating a nested
subdirectory, omit `--name`:

```bash
uv run dl-init-experiment --root-dir .
```

Important arguments:

- `--name` (optional)
- `--root-dir`
- dynamically added extension flags such as `--with-azure`, `--with-mlflow`,
  and `--with-wandb` when the relevant extra package is installed

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

Reads local sweep tracking plus per-run metric summaries and prints a compact
ranking/report.

```bash
uv run dl-analyze --sweep experiments/lr_sweep.yaml
```

Useful flags:

- `--json`

## `dl-train-worker`

Internal worker entrypoint used by executors. This is the direct trainer
process, not the normal user-facing entrypoint.

```bash
uv run dl-train-worker --config configs/base.yaml
```

In normal usage, prefer `dl-run` or `dl-sweep`.
