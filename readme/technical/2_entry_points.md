# Technical: 2. Entry Points

`dl-core` exposes five console entrypoints.

## `dl-core`

Creates local component scaffolds inside an experiment repository.

```bash
uv run dl-core add augmentation Custom1
```

Useful flags:

- `--root-dir`
- `--package-name`
- `--force`

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
- `--package-name`
- `--with-azure`

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
uv run dl-sweep --sweep experiments/lr_sweep.yaml
```

Useful flags:

- `--dry-run`
- `--resume`
- `--max-workers`
- `--compute`
- `--environment`

## `dl-train-worker`

Internal worker entrypoint used by executors. This is the direct trainer
process, not the normal user-facing entrypoint.

```bash
uv run dl-train-worker --config configs/base.yaml
```

In normal usage, prefer `dl-run` or `dl-sweep`.
