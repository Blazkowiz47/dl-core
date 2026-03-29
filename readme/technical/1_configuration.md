# Technical: 1. Configuration

`dl-core` uses YAML configs with dictionary-shaped component sections. The
generated experiment repository starts from `configs/base.yaml`, while sweep
templates build on top of that base config.

## Core Shape

The common top-level sections are:

- `runtime`
- `experiment`
- `accelerator`
- `models`
- `dataset`
- `optimizers`
- `trainer`
- `criterions`
- `metric_managers`
- `callbacks`

Generated sweep runs may also contain `executor` and `tracking`.

## Models

`models` is a mapping where the key is the registry name and the value is the
parameter block.

Generated experiment repos default to a local `resnet_example` wrapper:

```yaml
models:
  resnet_example:
    name: resnet_example
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

## Optimizer

The default path uses a single flat optimizer config:

```yaml
optimizers:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01
```

## Trainer

Generated repos default to a project-named trainer wrapper:

```yaml
trainer:
  my_exp:
    name: my_exp
    epochs: 3
```

That wrapper extends `dl_core.trainers.standard_trainer.StandardTrainer`,
which builds on the epoch-based `dl_core.core.EpochTrainer`.

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
