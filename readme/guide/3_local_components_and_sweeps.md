# Guide: 3. Local Components and Sweeps

The generated repository is set up so local wrappers register automatically.

## Local Component Defaults

In a scaffolded project named `my-exp`:

- dataset name: `my_exp`
- trainer name: `my_exp`
- model name: `resnet_example`

Those classes extend the built-in `StandardWrapper`, `StandardTrainer`, and
`ResNet`.

## How Registration Works

`dl-core` does two things at runtime:

1. it imports built-in package modules so core registries populate
2. it finds the nearest project root containing `pyproject.toml` and `src/`
3. it imports top-level packages under `src/`

That means your experiment package is loaded automatically when you run
`dl-run` or `dl-sweep` from inside the experiment repository.

## Adding New Local Components

Use the `dl-core` helper to generate thin local wrappers and stubs without
editing the package structure by hand.

Example:

```bash
uv run dl-core add augmentation Custom1
uv run dl-core add callback EpochLogger
uv run dl-core add sampler PassThroughSampler
```

Supported component types:

- `augmentation`
- `callback`
- `criterion`
- `dataset`
- `executor`
- `metric`
- `metric_manager`
- `model`
- `sampler`
- `trainer`

The command creates the right package on demand under `src/<package>/` and
normalizes the module name for you. Generated components register under the
normalized name and also keep the original provided name as an alias when it
differs.

## Local Training

```bash
uv run dl-run --config configs/base.yaml
```

## Sweeps

```bash
uv run dl-sweep --sweep configs/sweeps/example_sweep.yaml
```

Generated sweep configs are saved under:

```text
configs/sweeps/<sweep_name>/
```

Each generated config gets its own `runtime.name`, so artifact directories do
not collide.
