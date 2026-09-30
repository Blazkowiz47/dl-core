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
Set `runtime.log_level` to control worker logging. Configured callbacks must
use registered names and mapping-shaped options; invalid callbacks stop setup.

## Models

`models` is a mapping where the key is the registry name and the value is the
parameter block. Component names must match a registered name exactly.

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
`empty_check` defaults to `false`, allowing a rank or worker with no assigned
shards to finish its stream. An epoch with no shared training batches still
fails clearly. Set `empty_check: true` to treat an empty worker stream as an
error. `strict_pairs: false` skips missing-member samples with a warning;
the default raises instead. Transform errors still raise unless a project
wrapper handles them.
`mix_longest` controls whether a finite weighted mix continues after one source
is exhausted and defaults to `true` for non-resampled streams.

`EpochTrainer` supports finite iterable training loaders by stopping every rank
before the first unmatched batch. It finalizes a partial gradient-accumulation
window on the last shared batch; valid tail samples on longer ranks are not
trained on. Use `IterationTrainer` for infinite or resampled training streams.
Validation/test do not use that shortest-rank cutoff: each rank evaluates all
its valid samples, and globally empty splits raise. For uneven evaluation, use
the metric manager's default `gather` mode rather than `average`. Custom model
forwards and batch callbacks must avoid their own per-batch distributed
collectives in evaluation.

`dataset.shards` and `shard_patterns` are default conveniences. A project
wrapper can instead override `build_shard_sources(split)` and return entries
with `name`, `weight`, and `shards`. Shards may be strings or dictionaries with
a `path` plus project metadata. Multiple positive-weight sources are mixed with
WebDataset `RandomMix`; zero-weight sources are skipped.

### Indexed plain tar reading

`TarShardWrapper.build_indexed_dataset(data, split)` is an opt-in utility for
concrete wrappers. It accepts the same weighted source structure as streaming,
with local plain tar paths. The `members`, `key`, `path`, `shard_path`, and
source metadata passed to transforms retain their meaning. A storage adapter
must supply local paths and keep them reserved until the reader and workers
have finished. The default `build_dataset()` remains a WebDataset stream.

The index groups regular members by sample key and stores each extension's
offset and size. Sparse members, duplicate extensions within a sample, and
compressed tar files are rejected. Long member names are supported. Optional
`sample_keys` in a shard record limits its eligible keys before sampling.

`indexed_tar.index_dir` defaults to `~/.cache/dl-core/tar-indexes`; `null`
disables disk indexes. Index publication is atomic, and file identity, size,
and modification time determine reuse. `max_open_shards` defaults to eight
handles per process; handles are opened lazily and omitted from pickled state.
Call `dataset.close()` to close handles in the current process.

The wrapper's indexed batch sampler mixes source weights independently of
source size. A finite pass visits each selected sample once, with an optional
`num_samples` cutoff. `replacement: true` requires a positive `num_samples`
budget and allows repeated draws. These sampling options live in
`dataset.indexed_tar`, separately from `webdataset` stream settings. The
built-in sampler targets single-GPU use; custom samplers remain available
through `build_batch_sampler()`.

```python
loader = wrapper.get_split("train")  # Concrete wrapper selected indexed reading
totals = {shard: count for shard, count in loader.dataset.shard_totals.items() if count}
wrapper.reset_shard_progress(totals)
for batch in loader:
    if not batch:
        continue
    train_step(batch)
    wrapper.record_shard_consumption(batch["shard_id"])
    progress = wrapper.get_shard_progress()
```

Enable `dataset.track_shard_progress` to retain `shard_id` after the transform.
Azure adapters use container-relative `source_path` as the default ID; local
shards use their resolved path. A shard record may supply an explicit
`shard_id`. The progress utility also works with streaming when eligible counts
or budgets are supplied externally. `get_shard_progress(shard_id)` returns
`consumed`, `total`, and `fraction`; a `None` total yields a `None` fraction.
Reported fractions cap at one, while consumed counts retain all recorded draws.

Progress records completed training batches, so worker read-ahead does not
advance it. For repeated draws, filtering, or `drop_last`, supply appropriate
finite budgets or eligible counts instead of assuming the raw tar length.
Reset totals explicitly at each pass; never call progress methods from workers.

A bounded read/decode benchmark is available as
`uv run python scripts/benchmark_indexed_tar.py /path/to/shard.tar --samples 256`.
It uses one image-decoder thread per worker, warms the same selection for each
configuration, and reports index construction, first-pass time, and steady-pass
throughput for worker counts 0, 1, 2, 4, and 8. It reads the tar without changing
it, stores indexes temporarily, and removes them when finished.

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
    iterations: 100
```

That wrapper reuses the config-backed component steps from `StandardTrainer`
with the `IterationTrainer` lifecycle. One iteration consumes one training
batch per rank. Gradient accumulation does not rescale the configured budget.
`dl-core add trainer Name` also defaults to `IterationTrainer`; use
`--base epochtrainer` to request an epoch-based scaffold.

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
Automatic local resume selects the first run directory with checkpoint
artifacts: the flat run directory, an experiment-grouped directory, or a
standalone `artifacts/sweeps/<config_stem>/<run_name>/` directory. It keeps
writing checkpoints and final artifacts into that selected directory. A
`.yml` standalone config also recognizes its filename-with-extension layout.
For sweeps, `.yml` files use their stem as the artifact group name; automatic
resume also checks existing extension-named sweep directories.

After successful training, the trainer lifecycle calls `select_checkpoint()` and
passes that path to `post_training(checkpoint_path)`. The default selector
returns final `best.pth` when the checkpoint callback created it, falls back to
final `latest.pth`, and otherwise returns `None`. Override `select_checkpoint()`
for custom single- or multi-metric model selection, and override
`post_training()` for completed-run evaluation or export work.

The generated trainer already uses the `IterationTrainer` lifecycle. The same
iteration budget works for finite loaders and streaming or cyclic training:

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
    iterations: 100

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
