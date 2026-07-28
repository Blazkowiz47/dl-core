# Guide: 2. Creating an Experiment Repository

The experiment repository is the user-facing workspace built on top of
`dl-core`.

## Create It

```bash
uv run dl-init --name my-exp --root-dir .
```

To initialize the current directory itself:

```bash
uv run dl-init --root-dir .
```

Optional Azure dependency wiring, available when `dl-core[azure]` is installed:

```bash
uv run dl-init --name my-exp --root-dir . --with-azure
```

Optional local MLflow dependency wiring, available when `dl-core[mlflow]` is
installed:

```bash
uv run dl-init --name my-exp --root-dir . --with-mlflow
```

## What Gets Generated

```text
my-exp/
  AGENTS.md
  CLAUDE.md
  pyproject.toml
  configs/
    base.yaml
    base_sweep.yaml
    presets.yaml
  experiments/
    lr_sweep.yaml
    experiments.log
  scripts/
    temporary/
      README.md
      test_dataset.py
      test_model.py
  src/
    bootstrap.py
    datasets/
      my_exp.py
    models/
      resnet_example.py
    trainers/
      my_exp.py
```

## Why the Local Components Exist

The scaffold gives you local trainer and dataset components plus a complete
project-owned example model so you can:

- keep experiment-specific changes out of `dl-core`
- preserve a stable default path for new projects
- override behavior later without forking the framework package

By default:

- the dataset wrapper is named after the project package
- the trainer wrapper is named after the project package
- the project owns the generated `ResNetExample` architecture

## Migrating an Older ResNet Scaffold

Older generated projects may still import
`dl_core.models.resnet.ResNet`. Model architectures are no longer shipped by
`deep-learning-core`. Generate a fresh temporary scaffold, copy its
`src/models/resnet_example.py` into the older experiment, and declare
`torchvision` in that experiment's dependencies. The existing
`models.resnet_example` configuration key can remain unchanged.

## First Files To Edit

- `configs/base.yaml`
- `scripts/temporary/test_dataset.py`
- `scripts/temporary/test_model.py`
- `configs/base_sweep.yaml`
- `configs/presets.yaml`
- `experiments/lr_sweep.yaml`
- `experiments/experiments.log`
- `AGENTS.md`
- `CLAUDE.md`

Start there before editing the wrapper classes. After updating the dataset or
model wrapper, use:

```bash
uv run python scripts/temporary/test_dataset.py
uv run python scripts/temporary/test_model.py
```

before committing to a full `dl-run`.

Use `configs/base.yaml` for reusable shared defaults. Before a real single run,
copy or derive it into a named file under `experiments/`, for example
`experiments/debug.yaml`, validate that concrete config, and run `dl-run`
against the `experiments/` file.
