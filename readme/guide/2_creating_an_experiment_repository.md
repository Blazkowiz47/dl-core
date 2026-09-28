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
      preview_augmentations.py
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
- the trainer uses an iteration budget measured in training batches per rank
- the project owns the generated `ResNetExample` architecture

This default applies to new scaffolds. Refreshing `AGENTS.md` in an older
project updates its guidance, not its trainer or configs.

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
- `scripts/temporary/preview_augmentations.py`
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
uv run python scripts/temporary/preview_augmentations.py --split train
uv run python scripts/temporary/preview_augmentations.py --split validation
uv run python scripts/temporary/test_model.py
```

before committing to a full `dl-run`. The preview saves post-transform images
to an ignored local directory; adapt it for the project's image keys and
normalization. `test_dataset.py` also supports iterable loaders without a
length or random access.

Use `configs/base.yaml` for reusable shared defaults. Copy it once to
`experiments/debug.yaml` and keep editing that file while prototyping. Before
a run worth keeping, promote the working config to a distinct named file under
`experiments/`, validate it, then run `dl-run` against that file.
