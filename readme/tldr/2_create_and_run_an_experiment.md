# TLDR: Create and Run an Experiment

The default path is:

- scaffold a new experiment repository
- use the generated project-named dataset and trainer wrappers
- keep the default `ResNetExample` model wrapper
- run locally first

## 1. Create the Experiment Repository

```bash
uv run dl-init --name my-exp --root-dir .
cd my-exp
```

Or, if you are already inside the empty target directory:

```bash
uv run dl-init --root-dir .
```

## 2. Install the Experiment Repository

For local development against a sibling checkout of `dl-core`:

```bash
uv add --editable ../dl-core
uv sync
```

## 3. Inspect the Generated Defaults

The scaffold gives you:

- `configs/base.yaml`
- `configs/base_sweep.yaml`
- `configs/presets.yaml`
- `experiments/lr_sweep.yaml`
- `experiments/experiments.log`
- `scripts/temporary/test_dataset.py`
- `scripts/temporary/test_model.py`
- `src/datasets/my_exp.py`
- `src/trainers/my_exp.py`
- `src/models/resnet_example.py`
- `src/bootstrap.py`

## 4. Run Locally

Start with the smoke helpers:

```bash
uv run python scripts/temporary/test_dataset.py
uv run python scripts/temporary/test_model.py
```

Then move on to the full commands:

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep experiments/lr_sweep.yaml
```

Before running, set `dataset.rdir` to a real dataset or replace the local
dataset wrapper with a dummy-data implementation for smoke testing.
