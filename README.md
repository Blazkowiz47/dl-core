# deep-learning-core

Reusable deep learning framework core.

`deep-learning-core` contains the vendor-neutral training framework that can be reused
across many experiment repositories. It is intended to be the public base
package, while optional integrations such as Azure are layered on through
extras and companion extension packages.

## Install

Install from PyPI:

```bash
pip install deep-learning-core
```

Install with Azure support:

```bash
pip install "deep-learning-core[azure]"
```

Install with local MLflow support:

```bash
pip install "deep-learning-core[mlflow]"
```

Install with W&B support:

```bash
pip install "deep-learning-core[wandb]"
```

Install with multiple variants:

```bash
pip install "deep-learning-core[azure,wandb]"
```

Install in a `uv` project:

```bash
uv add deep-learning-core
```

`deep-learning-core` intentionally ships with the full public runtime
dependencies, including `torch`, `torchvision`, and `opencv-python-headless`. The Azure
extra pulls in `deep-learning-azure`, which pins the Azure package versions
used by the validated Azure packaging stack. The MLflow extra pulls in
`deep-learning-mlflow` for local MLflow tracking. The W&B extra pulls in
`deep-learning-wandb` and leaves the `wandb` package itself unpinned.

## Package Variants

- `deep-learning-core`: local training, local sweeps, local sweep analysis, and the
  experiment scaffold
- `deep-learning-core[azure]`: adds the public
  [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
  package for Azure execution and Azure dataset foundations
- `deep-learning-core[mlflow]`: adds the public
  [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
  package for local MLflow integration
- `deep-learning-core[wandb]`: adds the public
  [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)
  package for Weights & Biases integration

The extension packages stay separate so the base package remains reusable and
vendor-neutral.

You can also install the companion packages directly when you want a specific
integration without using extras:

```bash
pip install deep-learning-azure
pip install deep-learning-mlflow
pip install deep-learning-wandb
```

## Scope

- Base abstractions and registries
- Built-in accelerators, callbacks, criterions, metrics, and schedulers
- The standard trainer and standard dataset flow
- Built-in augmentations
- Local execution and sweep orchestration
- Local sweep analysis from saved artifact summaries
- Experiment repository scaffolding via `dl-init-experiment`

## Out Of Scope

- Azure ML wiring unless the Azure extra is installed
- Workspace or datastore conventions
- Experiment-specific datasets, models, and trainers
- User-owned configs and private data

## Quick Start

```bash
uv run dl-core list
uv run dl-init-experiment --name my-exp --root-dir .
```

To initialize the current directory in place, omit `--name`:

```bash
uv run dl-init-experiment --root-dir .
```

The generated experiment repository is the normal consumer entry point. Inside
that repository, run `uv sync`, then run:

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml
```

## First Run Workflow

If you are starting from scratch, the minimum path is:

```bash
pip install deep-learning-core
uv run dl-init-experiment --name my-exp --root-dir .
cd my-exp
uv sync
```

Then:

1. open these generated files first:
   - `src/datasets/my_exp.py`
   - `configs/base.yaml`
   - `scripts/temporary/test_dataset.py`
   - `scripts/temporary/test_model.py`
   - `experiments/lr_sweep.yaml`
   - `AGENTS.md`
2. implement the generated dataset wrapper under `src/datasets/my_exp.py`
3. adjust `configs/base.yaml` so it points at the dataset/model/trainer you want
4. smoke-check the generated helpers:

```bash
uv run python scripts/temporary/test_dataset.py
uv run python scripts/temporary/test_model.py
```

5. start with:

```bash
uv run dl-run --config configs/base.yaml
```

Once that works, move on to:

```bash
uv run dl-sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml
```

If Azure support is installed, `uv run dl-init-experiment --with-azure` will
also scaffold Azure-ready config placeholders and `azure-config.json`.

If local MLflow support is installed,
`uv run dl-init-experiment --with-mlflow` will also scaffold an `mlflow`
callback block and local tracking defaults.

If W&B support is installed, `uv run dl-init-experiment --with-wandb` will also
scaffold a `wandb` callback block, W&B tracking defaults, and `.env.example`.

## Companion Packages

- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Scaffold Commands

Each `dl-core add ...` command creates the new module and updates the matching
local package `__init__.py` export list under `src/`.

Common local component scaffolds:

```bash
uv run dl-core add model MyResNet
uv run dl-core add trainer MyTrainer
uv run dl-core add callback MyMetrics
uv run dl-core add metric_manager MyManager
uv run dl-core add sampler MySampler
uv run dl-core add criterion MyLoss
uv run dl-core add augmentation MyAugmentation
uv run dl-core add metric MyMetric
uv run dl-core add executor MyExecutor
```

Sweep scaffolds are supported too:

```bash
uv run dl-core add sweep DebugSweep
uv run dl-core add sweep AzureEval --tracking azure_mlflow
uv run dl-core add sweep MlflowBaseline --tracking mlflow
uv run dl-core add sweep WandbAblation --tracking wandb
```

Generated sweep files:

- live under `experiments/`
- extend `../configs/base_sweep.yaml`
- include runnable defaults in `fixed`
- start with `grid: {}`
- default the tracker experiment destination to the repository root name unless
  `tracking.experiment_name` overrides it
- let the tracker derive sweep grouping from the filename unless
  `tracking.sweep_name` overrides it

Generated experiment repositories also include empty local
`src/criterions/`, `src/optimizers/`, and `src/schedulers/` packages so
project-specific components can be added there later without extra setup.

You can inspect registered components and built-in base classes directly from
the CLI:

```bash
uv run dl-core list
uv run dl-core list sampler
uv run dl-core list metric_manager --json
uv run dl-core describe dataset my_dataset --root-dir .
uv run dl-core describe model my_resnet --root-dir .
uv run dl-core describe class dl_core.core.FrameWrapper
uv run dl-core describe class dl_azure.datasets.AzureComputeMultiFrameWrapper
uv run dl-core describe dataset my_dataset --root-dir . --json
```

The built-in sampler list now includes `attack`, which balances PAD samples
from the `attack` key in each data dictionary.

The describe command shows:

- resolved class and registered names
- constructor signature
- inheritance chain
- docstring
- declared properties
- class-level attributes
- public methods defined on the class

It does not discover instance attributes created dynamically inside `__init__`
without constructing the class.

Dataset scaffolds can target a specific wrapper base:

```bash
uv run dl-core add dataset MyDataset
uv run dl-core add dataset FrameSet --base frame
uv run dl-core add dataset TextSet --base text_sequence
uv run dl-core add dataset ActSet --base adaptive_computation
```

When `dl-azure` is importable, the dataset scaffold also exposes Azure bases:

```bash
uv run dl-core add dataset AzureFrames --base azure_compute_frame
uv run dl-core add dataset AzureSeq --base azure_compute_multiframe
uv run dl-core add dataset AzureStream --base azure_streaming
uv run dl-core add dataset AzureStreamSeq --base azure_streaming_multiframe
```

Plain `deep-learning-core` currently exposes dataset bases for:

- `BaseWrapper`
- `FrameWrapper`
- `TextSequenceWrapper`
- `AdaptiveComputationDataset`

`TextSequenceWrapper` adds sequence-aware batch padding for tokenized inputs.
`AdaptiveComputationDataset` adds per-class sample stream helpers for
adaptive-time computation trainers. Multiframe dataset bases are still
provided through `dl-azure`.

## Releases

- `Publish` is the production workflow for PyPI.
- Trusted publishing is configured through GitHub Actions environments rather
  than long-lived API tokens.
- The publish action may upload digital attestations alongside the package.
  That is expected behavior from `pypa/gh-action-pypi-publish`.
- Package metadata keeps runtime dependencies unpinned, so the consuming
  environment resolves the latest compatible public releases.

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-core/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-core)

## License

MIT. See [LICENSE](LICENSE).

## Development Validation

```bash
uv run --extra dev pytest
uv run python -m compileall src/dl_core
```
