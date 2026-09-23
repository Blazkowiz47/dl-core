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

WebDataset-backed tar wrappers declare shard paths and required grouped members:

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
  persistent_workers: true
  batch_size: 32
  webdataset:
    shard_shuffle: 100
    sample_shuffle: 10000
    sample_shuffle_initial: 100
    resampled:
      train: true
      validation: false
      test: false
```

Install this optional path with `deep-learning-core[webdataset]`. WebDataset
groups members by sample key, shuffles shards and samples with bounded buffers,
and splits shards with `split_by_node` and `split_by_worker`. Training can use a
resampled stream with `IterationTrainer`; validation and test should normally
remain finite. `shard_shuffle`, `sample_shuffle`, `sample_shuffle_initial`,
`resampled`, and `empty_check` may be scalars or split-specific mappings.
`mix_longest` controls whether a finite weighted mix continues after one source
is exhausted and defaults to `true` for non-resampled streams.

`dataset.shards` and `shard_patterns` are default conveniences. A project
wrapper can instead override `build_shard_sources(split)` and return entries
with `name`, `weight`, and `shards`. Shards may be strings or dictionaries with
a `path` plus project metadata. Multiple positive-weight sources are mixed with
WebDataset `RandomMix`; zero-weight sources are skipped.

For map-style datasets, automatic validation and test partitions are created
from the raw training records before any configured sampler is applied. This
keeps oversampled identities out of held-out splits. DataLoader workers receive
deterministic seeds that change with the epoch; loading or rebuilding a split
does not reset the process-wide Python, NumPy, or PyTorch random streams.

## Accelerator

Optimization behavior is configured on the accelerator:

```yaml
accelerator:
  type: single_gpu
  mixed_precision: fp16
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
```

The standard trainer accumulates gradients across the configured number of
microbatches, clips once immediately before a real optimizer update, and steps
its scheduler only when that optimizer update occurs. FP16 gradients are
unscaled before clipping. A shorter final accumulation window is averaged over
the microbatches it actually contains and is not discarded.

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

When a training step returns a tensor under `probabilities_tensor` or
`probabilities`, the base trainer records class-neutral diagnostics:
`prob_entropy_mean`, `prob_confidence_mean`, and `prob_margin_mean` when two or
more class probabilities are available. A single-column sigmoid probability is
treated as a binary distribution for these diagnostics. If the batch also
contains valid integer labels, it records `prob_true_class_mean`. These metrics
do not depend on project-specific class names.

Checkpoint aliases such as `latest.pth` and `best.pth` are replaced atomically
after the new payload is fully written. Automatic local resume validates a
checkpoint before selecting it and falls back from an unreadable `latest.pth`
to the next loadable numbered checkpoint. An explicitly requested checkpoint
is strict: if it cannot be loaded, the run fails instead of silently restarting
from the beginning. Resume restores model, optimizer, scheduler, criterion,
accelerator, callback, and trainer progress state when those entries are
present.
Automatic local resume also searches an existing experiment-grouped run layout
when no suitable checkpoint is found in the flat run directory.

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
without requiring a length. WebDataset-backed loaders split shards between
ranks and workers before reading samples; a resampled training stream keeps
every rank available for the configured iteration budget.

With gradient accumulation, iteration checkpoints are deferred until the
current accumulation window has produced an optimizer step. This prevents a
resume point from silently dropping gradients held only in memory.
Logging, validation, and testing requested mid-window run at that same safe
boundary. The trainer completes a final partial window before reporting or
saving its final state.

There is no generic `BaseTrainer` export. Import `EpochTrainer`,
`IterationTrainer`, or `RLTrainer` explicitly according to the lifecycle.

## Reproducibility

Generated base configs expose reproducibility at the root level:

```yaml
seed: 2025
deterministic: true
```

`seed` drives trainer setup, deterministic dataset splitting, epoch-specific
DataLoader generators, and any seed values injected into downstream configs.
`deterministic` controls PyTorch deterministic mode during trainer setup.

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
