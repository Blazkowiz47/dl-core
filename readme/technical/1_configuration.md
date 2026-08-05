# Technical: 1. Configuration

`dl-core` uses YAML configs with dictionary-shaped component sections. The
generated experiment repository starts from `configs/base.yaml`, while sweep
templates build on top of that base config.

## Core Shape

The common top-level sections are:

- `seed`
- `deterministic`
- `runtime`
- `experiment`
- `accelerator`
- `models`
- `dataset`
- `optimizers`
- optional `ema`
- `trainer`
- `criterions`
- `metric_managers`
- `callbacks`

Generated sweep runs may also contain `executor` and `tracking`.

## Models

`models` is a mapping where the key is the registry name and the value is the
parameter block.

Generated experiment repos default to a self-contained local
`resnet_example` implemented with torchvision:

```yaml
models:
  resnet_example:
    variant: resnet18
    pretrained: false
    num_classes: 2
```

## Dataset

The built-in standard dataset uses:

- `dataset.name`
- `dataset.classes`
- `dataset.rdir`
- `dataset.augmentation`

Example:

```yaml
dataset:
  name: my_exp
  classes: [class0, class1]
  rdir: null
  height: 64
  width: 64
  batch_size: 64
  num_workers: 0
  prefetch_factor: null
  augmentation:
    standard:
      height: 64
      width: 64
```

Notes:

- `rdir` is the root directory for the standard dataset path
- generated repos default to a project-named dataset wrapper that extends the
  built-in standard dataset
- if you need dummy or synthetic data, implement it in your local dataset
  wrapper instead of relying on the built-in standard dataset

Indexed tar wrappers use uncompressed `.tar` files and can declare explicit
shards and required grouped members:

```yaml
dataset:
  name: my_tar_dataset
  auto_split: false
  shards:
    train:
      - path: data/train/attack-000.tar
        group: attack
      - path: data/train/real-000.tar
        group: real
  required_extensions: [png, json]
  max_open_shards: 8
  persistent_workers: true
  batch_size: 32
  batch_sampler:
    type: round_robin_tar
    group_pattern: [attack, real]
    shuffle_within_batch: true
    distributed_drop_last: true
```

The index defaults to `<shard>.tar.idx.json`. `index_checksum: true` adds and
validates SHA-256 checksums; otherwise the index is validated using tar size.
The distributed sampler partitions complete batches between ranks. Workers
inside a rank receive only those selected sample indices and keep their own
process-local tar handles.

## Optimizer

The default path uses a single flat optimizer config:

```yaml
optimizers:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01
```

## EMA

EMA is optional and disabled by default. When enabled, it targets runtime model
keys, not the outer YAML keys under `models`.

With the standard trainer, the single prepared model is exposed as `main`, so
the commented scaffold example uses:

```yaml
ema:
  enabled: true
  decay: 0.9999
  eval_with_ema: true
  models: [main]
```

Checkpoint behavior when `save_in_checkpoint: true`:

- `models_state_dict` keeps the normal training weights
- `ema_state_dict` keeps EMA resume metadata and shadow parameters
- `ema_models_state_dict` keeps a full drop-in model state dict with EMA
  parameters and the original buffers preserved

That lets trainer-managed evaluation use EMA during `validation_epoch()` and
`test_epoch()`, while standalone evaluator code can directly load
`checkpoint["ema_models_state_dict"]["main"]` when it wants EMA weights.

## Trainer

Generated repos default to a project-named trainer wrapper:

```yaml
trainer:
  my_exp:
    epochs: 3
```

That wrapper extends `dl_core.trainers.standard_trainer.StandardTrainer`,
which builds on the epoch-based `dl_core.core.EpochTrainer`.

After successful training, the trainer lifecycle calls `select_checkpoint()` and
passes that path to `post_training(checkpoint_path)`. The default selector
returns final `best.pth` when the checkpoint callback created it, falls back to
final `latest.pth`, and otherwise returns `None`. Override `select_checkpoint()`
for custom single- or multi-metric model selection, and override
`post_training()` for completed-run evaluation or export work.

For streaming or cyclic training, a local trainer can instead extend
`dl_core.core.IterationTrainer` and replace `epochs` with an iteration budget:

```yaml
trainer:
  my_streaming_exp:
    iterations: 100000
    log_frequency: 1000
    validation_frequency: 5000
    test_frequency: 10000
    checkpoint_frequency: 5000
```

One iteration consumes one training batch on every distributed rank. A zero
validation, test, or checkpoint frequency means final-only. Finite loaders are
cycled and their cycle/cursor state is checkpointed; infinite loaders continue
without requiring a length. The global batch sequence must still be partitioned
into complete per-rank batches by the sampler so ranks do not duplicate work.

There is no generic `BaseTrainer` export. Import `EpochTrainer`,
`IterationTrainer`, or `RLTrainer` explicitly according to the lifecycle.

## Reproducibility

Generated base configs expose reproducibility at the root level:

```yaml
seed: 2025
deterministic: true
```

`seed` drives trainer setup, dataset splitting, worker seeding, and any seed
values injected into downstream configs. `deterministic` is forwarded to the
trainer and dataset seed helpers so PyTorch deterministic mode can be disabled
explicitly when needed.

## Name Key Rules

`name` is only needed in sections where the config uses the flat single-item
shape and the loader needs an explicit selector inside the section itself.

Keep `name` in:

- `dataset.name`
- `optimizers.name`
- `schedulers.name`

Do not repeat `name` inside keyed mappings where the outer key already selects
the registered component:

```yaml
models:
  resnet_example:
    variant: resnet18

trainer:
  my_exp:
    epochs: 3

criterions:
  crossentropy:

metric_managers:
  standard:
    num_classes: 2
```

## Runtime

`runtime.name` is optional. If it is omitted, `dl-core` falls back to the
config filename stem for the run identifier used in artifact naming.

```yaml
runtime:
  # name: my_exp_baseline
  output_dir: artifacts
  log_level: INFO
  tags: []
```

## Sweep Templates

`configs/base_sweep.yaml` defines sweep defaults such as:

- `base_config`
- `fixed`
- `default_grid`
- `tracking`
- `seeds`

User sweep files typically extend the base template and only override `grid`
plus any description or tagging metadata.
