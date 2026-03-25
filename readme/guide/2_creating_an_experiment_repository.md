# Guide: 2. Creating an Experiment Repository

The experiment repository is the user-facing workspace built on top of
`dl-core`.

## Create It

```bash
uv run dl-init-experiment --name my-exp --root-dir .
```

To initialize the current directory itself:

```bash
uv run dl-init-experiment --root-dir .
```

Optional Azure dependency wiring, available when `dl-core[azure]` is installed:

```bash
uv run dl-init-experiment --name my-exp --root-dir . --with-azure
```

Optional local MLflow dependency wiring, available when `dl-core[mlflow]` is
installed:

```bash
uv run dl-init-experiment --name my-exp --root-dir . --with-mlflow
```

## What Gets Generated

```text
my-exp/
  pyproject.toml
  configs/
    base.yaml
    base_sweep.yaml
    presets.yaml
  experiments/
    lr_sweep.yaml
    experiments.log
  src/
    bootstrap.py
    datasets/
      my_exp.py
    models/
      resnet_example.py
    trainers/
      my_exp.py
```

## Why the Wrappers Exist

The scaffold intentionally gives you thin local wrappers so you can:

- keep experiment-specific changes out of `dl-core`
- preserve a stable default path for new projects
- override behavior later without forking the framework package

By default:

- the dataset wrapper is named after the project package
- the trainer wrapper is named after the project package
- the model wrapper stays `ResNetExample`

## First Files To Edit

- `configs/base.yaml`
- `configs/base_sweep.yaml`
- `configs/presets.yaml`
- `experiments/lr_sweep.yaml`
- `experiments/experiments.log`

Start there before editing the wrapper classes.
