# Guide: 3. Local Components and Sweeps

The generated repository is set up so local wrappers register automatically.

## Local Component Defaults

In a scaffolded project named `my-exp`:

- dataset name: `my_exp`
- trainer name: `my_exp`
- model name: `resnet_example`

The generated dataset is a visible `BaseWrapper` skeleton, the trainer extends
the built-in `StandardTrainer` on top of `EpochTrainer`, and the project owns
its torchvision `ResNetExample` architecture.

## How Registration Works

`dl-core` does two things at runtime:

1. it imports built-in package modules so non-model registries populate
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
uv run dl-core add dataset TarDataset --base tar_shard
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

## Writing local components

Start from the generated method stub and implement only the behavior the
component needs. Match a nearby component before introducing a new pattern.
Keep the implementation local and direct when it remains readable.

- Avoid pass-through helpers and wrapper classes that add no behavior.
- Do not extract one-off logic into a helper unless it is reused more than
  twice or represents a distinct operation that benefits from independent
  testing.
- Do not add configuration options until the component has a real use for
  them.
- Keep component-specific behavior in the component until multiple components
  need the same abstraction.

For models based on `BaseModel`, keep `compute_forward()` in three visible
stages when the architecture allows it:

```python
def compute_forward(self, batch_data: dict, **kwargs) -> dict:
    # 1. Retrieve and prepare inputs.
    inputs = batch_data["image"]

    # 2. Run the model elements in execution order.
    features = self.encoder(inputs)
    logits = self.classifier(features)

    # 3. Build and return the final output dictionary.
    probabilities = torch.softmax(logits, dim=1)
    return {
        "probabilities": probabilities,
        "logits": logits,
        "features": features,
    }
```

Keep losses, metric updates, logging, optimizer operations, and unrelated
state changes outside `compute_forward()`.

When documentation or an example names a PyTorch version, check the
[official PyTorch releases](https://github.com/pytorch/pytorch/releases) and
use the latest stable release. Label an older version as a compatibility pin,
and keep `torchvision` and `torchaudio` compatible with the selected release.

For dataset scaffolds, the available `--base` values depend on what is
installed in the current environment:

- plain `dl-core`: `base`, `frame`, `text_sequence`, `adaptive_computation`
- with `dl-azure`: adds `azure_compute`, `azure_streaming`,
  `azure_compute_frame`, `azure_streaming_frame`,
  `azure_compute_multiframe`, and `azure_streaming_multiframe`

The generated dataset stub includes the abstract methods required by the
selected base class so the implementation contract is visible immediately.

The built-in sampler shipped with `dl-core` is `label`, which balances samples
by a metadata key such as `label` or `attack` using either `undersample` or
`oversample`.

For non-dataset components, `--base` can point at a registered component such
as `metric_logger`, `standard`, `adamw`, or `cosine`, or a fully qualified
class path. If you omit `--base`, the scaffold uses the plain base class.

The built-in callback `dataset_refresh` can rebuild selected dataloaders at
epoch boundaries when a dataset needs fresh split sampling:

```yaml
callbacks:
  dataset_refresh:
    refresh_frequency: 1
    splits: [train]
```

The core dataset bases are intended for different data shapes:

- `base`: generic sample-level datasets
- `frame`: grouped video-frame datasets
- `text_sequence`: tokenized text and sequence datasets with padded batching
- `adaptive_computation`: sample-level datasets with class-stream helpers for
  adaptive-time computation trainers

## Local Training

```bash
uv run dl-run --config configs/base.yaml --validate-only
cp configs/base.yaml experiments/debug.yaml
uv run dl-run --config experiments/debug.yaml
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
