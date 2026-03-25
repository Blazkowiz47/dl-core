# dl-core

Reusable deep learning framework core.

`dl-core` contains the vendor-neutral training framework that can be reused
across many experiment repositories. It is intended to be the public base
package, while optional integrations such as Azure are layered on through
extras and companion extension packages.

## Install

Current public validation releases are published on TestPyPI. Once the package
is promoted to PyPI, the plain `pip install dl-core` forms below will be the
normal install path.

PyPI install target:

```bash
pip install dl-core
```

Install with Azure support:

```bash
pip install "dl-core[azure]"
```

Install with local MLflow support:

```bash
pip install "dl-core[mlflow]"
```

Install with W&B support:

```bash
pip install "dl-core[wandb]"
```

Install with multiple variants:

```bash
pip install "dl-core[azure,wandb]"
```

Install today from TestPyPI with `pip`:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  dl-core
```

Install today from TestPyPI in a `uv` project:

```toml
[tool.uv.sources]
dl-core = { index = "testpypi" }

[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
explicit = true
```

Then run:

```bash
uv add dl-core
```

Do not use `uv add --index https://test.pypi.org/simple/ dl-core` for this
package. With `uv`'s default `first-index` strategy, that can pull unrelated
dependency names from TestPyPI instead of PyPI.

`dl-core` intentionally ships with the full public runtime dependencies,
including `torch`, `torchvision`, and `opencv-python`. The Azure extra pulls in
`dl-azure`, which pins the Azure package versions used by the validated Azure
packaging stack. The MLflow extra pulls in `dl-mlflow` for local MLflow
tracking. The W&B extra pulls in `dl-wandb` and leaves the `wandb` package
itself unpinned.

## Package Variants

- `dl-core`: local training, local sweeps, local sweep analysis, and the
  experiment scaffold
- `dl-core[azure]`: adds the public
  [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
  package for Azure execution and Azure dataset foundations
- `dl-core[mlflow]`: adds the public
  [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
  package for local MLflow integration
- `dl-core[wandb]`: adds the public
  [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)
  package for Weights & Biases integration

The extension packages stay separate so the base package remains reusable and
vendor-neutral.

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
dl-run --show-registry
dl-init-experiment --name my-exp --root-dir .
```

To initialize the current directory in place, omit `--name`:

```bash
dl-init-experiment --root-dir .
```

The generated experiment repository is the normal consumer entry point. Install
that repository in editable mode, then run:

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep --sweep experiments/lr_sweep.yaml
uv run dl-analyze-sweep --sweep experiments/lr_sweep.yaml
```

If Azure support is installed, `dl-init-experiment --with-azure` will also
scaffold Azure-ready config placeholders and `azure-config.json`.

If local MLflow support is installed, `dl-init-experiment --with-mlflow` will
also scaffold an `mlflow` callback block and local tracking defaults.

If W&B support is installed, `dl-init-experiment --with-wandb` will also
scaffold a `wandb` callback block, W&B tracking defaults, and `.env.example`.

## Companion Packages

- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

To add a new local component scaffold inside the experiment repo:

```bash
uv run dl-core add augmentation Custom1
```

Dataset scaffolds can now target a specific wrapper base:

```bash
uv run dl-core add dataset MyDataset
uv run dl-core add dataset FrameSet --base frame
```

When `dl-azure` is importable, the dataset scaffold also exposes Azure bases:

```bash
uv run dl-core add dataset AzureFrames --base azure_compute_frame
uv run dl-core add dataset AzureSeq --base azure_compute_multiframe
```

Plain `dl-core` currently exposes dataset bases for `BaseWrapper` and
`FrameWrapper`. Multiframe dataset bases are provided through `dl-azure`.

## Releases

- `Publish TestPyPI` is the manual validation workflow for TestPyPI.
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
