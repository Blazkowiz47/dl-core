# dl-core

Reusable deep learning framework core.

`dl-core` contains the vendor-neutral training framework that can be reused
across many experiment repositories. It is intended to be the public base
package, while optional integrations such as Azure are layered on through
extras and companion extension packages.

## Install

Install from PyPI:

```bash
pip install dl-core
```

Install with Azure support:

```bash
pip install "dl-core[azure]"
```

Install with W&B support:

```bash
pip install "dl-core[wandb]"
```

Install from TestPyPI with `pip`:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  dl-core
```

Install from TestPyPI in a `uv` project:

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
`dl-azure`, which pins the Azure package versions used by the current
`pad_candidates` environment. The W&B extra pulls in `dl-wandb` and leaves the
`wandb` package itself unpinned.

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

If W&B support is installed, `dl-init-experiment --with-wandb` will also
scaffold a `wandb` callback block, W&B tracking defaults, and `.env.example`.

To add a new local component scaffold inside the experiment repo:

```bash
uv run dl-core add augmentation Custom1
```

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

- [Documentation Index](https://github.com/Blazkowiz47/dl-core/blob/main/readme/README.md)
- [TLDR: Install and Verify](https://github.com/Blazkowiz47/dl-core/blob/main/readme/tldr/1_install_and_verify.md)
- [TLDR: Create and Run an Experiment](https://github.com/Blazkowiz47/dl-core/blob/main/readme/tldr/2_create_and_run_an_experiment.md)
- [Guide: Getting Started](https://github.com/Blazkowiz47/dl-core/blob/main/readme/guide/1_getting_started.md)
- [Guide: Local Components and Sweeps](https://github.com/Blazkowiz47/dl-core/blob/main/readme/guide/3_local_components_and_sweeps.md)
- [Technical: Configuration](https://github.com/Blazkowiz47/dl-core/blob/main/readme/technical/1_configuration.md)
- [Technical: Entry Points](https://github.com/Blazkowiz47/dl-core/blob/main/readme/technical/2_entry_points.md)
- [Technical: Sweep System](https://github.com/Blazkowiz47/dl-core/blob/main/readme/technical/3_sweep_system.md)

## Development Validation

```bash
uv run --extra dev pytest
uv run python -m compileall src/dl_core
```
