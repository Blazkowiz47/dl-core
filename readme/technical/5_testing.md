# Technical: 5. Testing

`dl-core` currently validates itself at three levels.

## 1. Package Tests

```bash
uv run --extra dev pytest
```

These tests cover:

- experiment scaffold generation
- local component loading and registration

## 2. Static Import Sanity

```bash
uv run python -m compileall src/dl_core
```

## 3. Consumer-Repo Smoke Tests

From a generated experiment repository:

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep --sweep experiments/lr_sweep.yaml
```

These smoke tests assume you have either:

- pointed `dataset.rdir` at a real dataset
- or implemented a local dataset wrapper that provides dummy data for testing
