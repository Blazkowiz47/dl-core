# dl-core

Reusable deep learning framework core.

`dl-core` contains the vendor-neutral training framework that can be reused
across many experiment repositories. It is intended to be the public or
generally reusable package, while company-specific cloud integrations live in
separate adapters such as `dl-mobai-azure`.

## Scope

- Base abstractions and registries
- Built-in accelerators, callbacks, criterions, metrics, and schedulers
- The standard trainer and standard dataset flow
- Built-in augmentations
- Local execution and sweep orchestration
- Experiment repository scaffolding via `dl-init-experiment`

## Out Of Scope

- Company-specific Azure ML wiring
- Workspace or datastore conventions
- Experiment-specific datasets, models, and trainers
- User-owned configs and private data

## Quick Start

```bash
uv sync
uv run dl-run --show-registry
uv run dl-init-experiment --name my-exp --root-dir .
uv run dl-core add augmentation Custom1
```

The generated experiment repository is the normal consumer entry point. Install
that repository in editable mode, then run:

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep --sweep configs/sweeps/example_sweep.yaml
```

## Documentation

- [Documentation Index](./readme/README.md)
- [TLDR: Install and Verify](./readme/tldr/1_install_and_verify.md)
- [TLDR: Create and Run an Experiment](./readme/tldr/2_create_and_run_an_experiment.md)
- [Guide: Getting Started](./readme/guide/1_getting_started.md)
- [Guide: Local Components and Sweeps](./readme/guide/3_local_components_and_sweeps.md)
- [Technical: Configuration](./readme/technical/1_configuration.md)

## Development Validation

```bash
uv run --extra dev pytest
uv run python -m compileall src/dl_core
```
