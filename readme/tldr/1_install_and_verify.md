# TLDR: Install and Verify

The shortest path through `dl-core` is:

- install the package environment
- verify that registries load
- run the package test suite

## Package Repo Setup

```bash
cd dl-core
uv sync
```

## Verify Registry Loading

```bash
uv run dl-run --show-registry
```

You should see built-in accelerators, datasets, trainers, metrics, and models
listed in the registry output.

## Run the Package Tests

```bash
uv run --extra dev pytest
uv run python -m compileall src/dl_core
```

At this point the package itself is ready to be consumed by an experiment
repository.
