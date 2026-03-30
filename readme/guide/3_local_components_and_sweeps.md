# Guide: 3. Local Components and Sweeps

The generated repository is set up so local wrappers register automatically.

## Local Component Defaults

In a scaffolded project named `my-exp`:

- dataset name: `my_exp`
- trainer name: `my_exp`
- model name: `resnet_example`

The generated dataset is a visible `BaseWrapper` skeleton, the trainer extends
the built-in `StandardTrainer` on top of `EpochTrainer`, and the model extends
`ResNet`.

## How Registration Works

`dl-core` does two things at runtime:

1. it imports built-in package modules so core registries populate
2. it finds the nearest project root containing `pyproject.toml` and `src/`
3. it imports `bootstrap.py` plus known component packages under `src/`

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
uv run dl-core add optimizer MyOptimizer
uv run dl-core add scheduler MyScheduler
uv run dl-core add dataset LocalDataset
uv run dl-core add dataset FrameDataset --base frame
uv run dl-core add dataset TextDataset --base text_sequence
uv run dl-core add dataset ActDataset --base adaptive_computation
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
- `optimizer`
- `sampler`
- `scheduler`
- `trainer`

The command creates the right component package on demand under `src/` and
normalizes the module name for you. Generated components register under the
normalized name and also keep the original provided name as an alias when it
differs.

For dataset scaffolds, the available `--base` values depend on what is
installed in the current environment:

- plain `dl-core`: `base`, `frame`, `text_sequence`, `adaptive_computation`
- with `dl-azure`: adds `azure_compute`, `azure_streaming`,
  `azure_compute_frame`, `azure_streaming_frame`,
  `azure_compute_multiframe`, and `azure_streaming_multiframe`

The generated dataset stub includes the abstract methods required by the
selected base class so the implementation contract is visible immediately.

For non-dataset components, `--base` can point at a registered component such
as `metric_logger`, `standard`, `adamw`, or `cosine`, or a fully qualified
class path. If you omit `--base`, the scaffold uses the plain base class.

The core dataset bases are intended for different data shapes:

- `base`: generic sample-level datasets
- `frame`: grouped video-frame datasets
- `text_sequence`: tokenized text and sequence datasets with padded batching
- `adaptive_computation`: sample-level datasets with class-stream helpers for
  adaptive-time computation trainers

## Local Training

```bash
uv run dl-run --config configs/base.yaml
```

## Sweeps

```bash
uv run dl-sweep experiments/lr_sweep.yaml
```

Generated sweep configs are saved under:

```text
experiments/<sweep_name>/
```

Each generated config gets its own filename, and run naming falls back to that
config stem unless `runtime.name` overrides it, so artifact directories do not
collide.
