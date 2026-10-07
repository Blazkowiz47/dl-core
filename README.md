# deep-learning-core

Reusable deep learning framework core.

`deep-learning-core` contains the vendor-neutral training framework that can be reused
across many experiment repositories. It is intended to be the public base
package, while optional integrations such as Azure are layered on through
extras and companion extension packages.

Trainers own reusable optimization and rollout loops; experiment repositories
own and register neural model architectures. `deep-learning-core` deliberately
does not ship built-in neural networks.

Current public release: `deep-learning-core==0.1.16`.
Current development version: `0.1.16`.

Compatible companion package floors:

- `deep-learning-azure>=0.0.22,<0.1`
- `deep-learning-mlflow>=0.0.15,<0.1`
- `deep-learning-robotics>=0.0.6,<0.1`
- `deep-learning-wandb>=0.0.16,<0.1`

## What's New in 0.1.16?

- plain-tar index cache misses build in parallel, with four indexing workers
  by default and a separate `dataset.indexed_tar.index_workers` setting
- shared build slots and per-shard locks coordinate local ranks, while atomic
  publication preserves valid indexes and original sample ordering
- startup logs report cached/built shard counts and elapsed time, including
  while waiting for slow builds or locks and when indexing sequentially
- the indexed-tar benchmark compares cold and warm index construction with
  configurable indexing worker counts and distinguishes samples whose keys
  repeat across different shards

Previous versions are recorded in the [release history](RELEASES.md).

## Install

Install from PyPI:

```bash
pip install deep-learning-core
```

The package keeps PyTorch accelerator selection with the consuming project.
With uv, the Linux development/test extra resolves CPU-only PyTorch wheels so
CI does not install CUDA dependencies.

Install WebDataset-backed tar support only when needed:

```bash
pip install "deep-learning-core[webdataset]"
```

Install with Azure support:

```bash
pip install "deep-learning-core[azure]"
```

Install with local MLflow support:

```bash
pip install "deep-learning-core[mlflow]"
```

Install with W&B support:

```bash
pip install "deep-learning-core[wandb]"
```

Install with multiple variants:

```bash
pip install "deep-learning-core[azure,wandb]"
```

Install in a `uv` project:

```bash
uv add deep-learning-core
```

`deep-learning-core` intentionally ships with the full public runtime
dependencies, including `torch` and `opencv-python-headless`. Generated
classification projects declare `torchvision` directly because their example
ResNet architecture belongs to the experiment repository. The Azure
extra pulls in `deep-learning-azure`, which pins the Azure package versions
used by the validated Azure packaging stack. The MLflow extra pulls in
`deep-learning-mlflow` for local MLflow tracking. The W&B extra pulls in
`deep-learning-wandb` and leaves the `wandb` package itself unpinned.

## Package Variants

- `deep-learning-core`: local training, local sweeps, local sweep analysis, and the
  experiment scaffold
- `deep-learning-core[azure]`: adds the public
  [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
  package for Azure execution and Azure dataset foundations
- `deep-learning-core[mlflow]`: adds the public
  [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
  package for local MLflow integration
- `deep-learning-core[wandb]`: adds the public
  [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)
  package for Weights & Biases integration
- [`deep-learning-robotics`](https://github.com/Blazkowiz47/dl-robotics):
  adds fast scalar and vector 2D MAPF environments, metrics, and episode media
  as a separately installed companion package

The extension packages stay separate so the base package remains reusable and
vendor-neutral.

You can also install the companion packages directly when you want a specific
integration without using extras:

```bash
pip install deep-learning-azure
pip install deep-learning-mlflow
pip install deep-learning-robotics
pip install deep-learning-wandb
```

## Scope

- Base abstractions and registries
- Built-in accelerators, callbacks, criterions, metrics, and schedulers
- The standard trainer and standard dataset flow
- Episode-driven tabular, DQN, PPO, SAC, and Dreamer training
- Built-in augmentations
- Local execution and sweep orchestration
- Local sweep analysis from saved artifact summaries
- Experiment repository scaffolding via `dl-init`

## Out Of Scope

- Azure ML wiring unless the Azure extra is installed
- Workspace or datastore conventions
- Experiment-specific datasets, models, and trainers
- User-owned configs and private data

## Quick Start

```bash
uv run dl-core list
uv run dl-init --name my-exp --root-dir .
```

To initialize the current directory in place, omit `--name`:

```bash
uv run dl-init --root-dir .
```

The generated experiment repository is the normal consumer entry point. Inside
that repository, run `uv sync`, then run:

```bash
uv run dl-run --config configs/base.yaml --validate-only
uv run dl-inspect-dataset --config configs/base.yaml
uv run dl-smoke --config configs/base.yaml
cp configs/base.yaml experiments/debug.yaml
uv run dl-run --config experiments/debug.yaml --validate-only
uv run dl-run --config experiments/debug.yaml
uv run dl-sweep experiments/lr_sweep.yaml --preview
uv run dl-sweep experiments/lr_sweep.yaml --only "*seed_2025*"
uv run dl-sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml
uv run dl-sync --sweep experiments/lr_sweep.yaml --artifacts
uv run dl-analyze --sweep experiments/lr_sweep.yaml --name pareto_eer
uv run dl-analyze --sweep experiments/lr_sweep.yaml --compare latest
uv run dl-analyze --sweep experiments/lr_sweep.yaml --metric test/eer --mode min
```

New local runs use the flattened artifact layout:

- `dl-run`: `artifacts/runs/<run_name>/...`
- `artifacts/sweeps/<sweep_name>/<run_name>/...`

`dl-core` does not create a `latest` symlink for these run directories. Use the
concrete run directory names directly.

## First Run Workflow

If you are starting from scratch, the minimum path is:

```bash
pip install deep-learning-core
uv run dl-init --name my-exp --root-dir .
cd my-exp
uv sync
```

Then:

1. open these generated files first:
   - `src/datasets/my_exp.py`
   - `configs/base.yaml`
   - `scripts/temporary/test_dataset.py`
   - `scripts/temporary/preview_augmentations.py`
   - `scripts/temporary/test_model.py`
   - `experiments/lr_sweep.yaml`
   - `AGENTS.md`
   - `CLAUDE.md`
2. implement the generated dataset wrapper under `src/datasets/my_exp.py`
3. adjust `configs/base.yaml` so it points at the dataset/model/trainer you want
   and set the shared reproducibility defaults you need (`seed` and
   `deterministic`). Keep concrete single-run configs in `experiments/`,
   including debug and baseline runs. Reuse `experiments/debug.yaml` while
   prototyping instead of creating a new YAML for every check.
4. smoke-check the wrapper and, for image projects, inspect a few augmented
   training and validation samples before training:

```bash
cp configs/base.yaml experiments/debug.yaml
uv run python scripts/temporary/test_dataset.py --config experiments/debug.yaml
uv run python scripts/temporary/preview_augmentations.py --config experiments/debug.yaml --split train
uv run python scripts/temporary/preview_augmentations.py --config experiments/debug.yaml --split validation
uv run python scripts/temporary/test_model.py --config experiments/debug.yaml
```

The generated trainer uses `IterationTrainer`: `iterations` counts consumed
training batches per rank, even when gradient accumulation is enabled.

5. start with:

```bash
uv run dl-run --config configs/base.yaml --validate-only
uv run dl-inspect-dataset --config configs/base.yaml
uv run dl-smoke --config configs/base.yaml
uv run dl-run --config experiments/debug.yaml --validate-only
uv run dl-run --config experiments/debug.yaml
```

Create `experiments/debug.yaml` once and edit it in place for dataset, model,
or protocol checks. Save a named config when a run becomes a distinct experiment
worth keeping; repeated real runs named `debug` can share an artifact location.
To update an existing project's generated guidance, run
`uv run dl-init --refresh-agents --root-dir .`, review the diff, and confirm.
It replaces only `AGENTS.md`, so review any project-specific or extension notes.

Once that works, move on to:

```bash
uv run dl-sweep experiments/lr_sweep.yaml --preview
uv run dl-sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml
uv run dl-analyze --sweep experiments/lr_sweep.yaml \
  --metric test/eer --mode min \
  --metric test/accuracy --mode max \
  --rank-method rank-sum
```

`dl-sweep --preview` prints the expanded run matrix without creating configs or
starting runs. Use `--export sweep_preview.csv` or `--export sweep_preview.json`
when you want to save that expansion for review.
Use `--only` and `--skip` with glob patterns when you want to execute or
preview only a subset of generated run names. A later `--resume` retries only
the runs selected when that sweep started, even if you omit the filters. Run
names must be unique; include any fields whose changes should define a new run.
If existing sweep data would be overwritten, a fresh run asks for confirmation
before writing. Use `--overwrite` for an intentional non-interactive replacement;
`--resume` keeps the existing tracker. `--dry-run` still writes generated YAML,
while `--preview` does not.

Local sweeps support four mutually exclusive resume modes:

| Flag | Runs to execute |
| --- | --- |
| `--resume` | Pending runs |
| `--resume-failed` | Failed runs |
| `--resume-stopped` | Stopped runs |
| `--resume-all` | Pending, failed, and stopped runs |

All four modes keep the original selection and run names and reject
`--overwrite`. Completed, running, and unknown runs are excluded. Historical
failures or unknown rows remain visible in the sweep history and do not change
the exit code of a successful local resume. An empty selection reports, for
example, `No pending runs to resume.` and exits successfully. Azure retains its
existing `--resume` behavior; the three additional modes require a local executor.
Sweep grids must use one executor name. Mixed names are rejected before filtering
or launching runs; `--executor local` explicitly selects local execution for the
whole grid.

During a built-in local sweep, press Ctrl-C to show the running jobs. Enter
comma-separated row numbers such as `2,3` to stop those jobs, or press Enter to
continue. Rows keep their displayed numbers while the menu is open; invalid input
stops nothing.
Running jobs continue while you choose, and new launches pause. Press Ctrl-C again
or enter `all` to stop the whole local sweep. Confirmed stops are recorded as
`stopped`, separately from `failed`; jobs that never started remain `pending`.

Selected jobs receive a graceful interrupt, followed by termination and a forced
stop if they do not exit within the grace periods. Process cleanup includes
descendants such as distributed workers. Console output is suppressed while
prompting, while each run's output is retained in `final/logs/sweep.log` under its
artifact directory. EOF or Ctrl-C in noninteractive execution stops all owned
runs. `dl-run` stops its single local job directly on the first Ctrl-C.

To customize local commands with selective stopping, override
`LocalExecutor.build_command()`. Existing `execute_run()` overrides keep the
shared sequential or process-pool dispatch and their returned results. That
dispatch path does not provide the selective menu; custom hooks control their
execution and cleanup.

Local sweep exit codes describe this command's claimed runs: `0` for completion
or no eligible runs, `1` for a failure, `3` for an unconfirmed result, and `130`
when the whole command is stopped. Selective stopping alone is not a failure.

`dl-inspect-dataset` preserves the configured split behavior, but forces
single-process loading so you can quickly verify split sizes and inspect one
collated batch without starting a trainer.

`dl-sync --sweep ... --artifacts` syncs tracked run outputs into the local repo.
Backends that already write local artifacts simply refresh the tracker paths.
Remote-backed integrations can download the run bundle and patch
`sweep_tracking.json` with the resolved local artifact paths.

`dl-analyze` defaults to ranking by `test/accuracy` with `max`. You can make
that explicit or override it with one or more `--metric` / `--mode` pairs and
choose `lexicographic`, `rank-sum`, or `pareto` ranking.

For Azure-backed sweeps, `dl-analyze` fetches only the requested metric
histories instead of downloading every tracked metric history. Those fetched
histories are cached in `experiments/<sweep_name>/analysis_cache.json`. Use
`--force` to ignore and refresh that cache. Reports are written under
`experiments/<sweep_name>/analysis/` as `v1.md`, `v2.md`, and so on unless you
pass `--name`. A matching JSON report is always written next to each Markdown
report, and `--compare latest` or `--compare v1` compares the current ranking
against a saved report.

## EMA Checkpoints

When EMA is enabled with `save_in_checkpoint: true`, each checkpoint stores:

- `models_state_dict`: normal model weights for training resume
- `ema_state_dict`: EMA bookkeeping and shadow-parameter state for trainer-side
  resume
- `ema_models_state_dict`: a full drop-in model state dict with EMA parameters
  and the original model buffers preserved

That means evaluator-side code can load:

- `checkpoint["models_state_dict"]["main"]` for normal weights
- `checkpoint["ema_models_state_dict"]["main"]` for EMA weights

without needing to reconstruct EMA state manually.

## Post-Training Checkpoint Hooks

After a successful training loop, the trainer lifecycle calls
`select_checkpoint()` and passes the returned path into
`post_training(checkpoint_path)`. This hook runs before run-analysis artifacts
are persisted and before tracking callbacks upload finalized artifacts.

The default `select_checkpoint()` implementation keeps the existing checkpoint
callback behavior authoritative: it returns final `best.pth` when present,
falls back to final `latest.pth`, and returns `None` if no checkpoint exists.
Override `select_checkpoint()` when a project needs custom single- or
multi-metric model selection, and override `post_training()` for completed-run
evaluation, export, or report generation.

## Trainer Lifecycles

New projects and `dl-core add trainer` default to `IterationTrainer`, which
stops after an exact number of batches. Use `EpochTrainer` explicitly when a
complete pass over the training loader should define progress:

```yaml
trainer:
  stream_trainer:
    iterations: 100000
    log_frequency: 1000
    validation_frequency: 5000
    test_frequency: 10000
    checkpoint_frequency: 5000
```

```python
from dl_core.core import IterationTrainer


class StreamTrainer(IterationTrainer):
    ...
```

Every distributed rank consumes one local batch per iteration, so all ranks
perform the same number of synchronized model updates. Finite loaders restart
with a new deterministic data cycle after every rank has exhausted one pass.
Shorter ranks replay their current selection until the shared boundary; these
repeated batches count toward the iteration budget. A single GPU advances as
soon as its loader exhausts. Infinite loaders remain in their current cycle.
Cycle boundaries use actual yielded batches rather than estimated lengths.
Checkpoints retain the completed iteration, cycle number, cursor including
replays, world size, and wrapper selection state.
With gradient accumulation, reporting and checkpoint saves wait for a completed
optimizer step; a final partial window is stepped before the final report.

`BaseTrainer` is no longer part of the API. Existing epoch-based subclasses
should import `EpochTrainer`; switching a project to iteration-based training
also requires replacing `epochs` with `iterations` and choosing iteration
frequencies.

If Azure support is installed, `uv run dl-init --with-azure` will
also scaffold Azure-ready config placeholders and `azure-config.json`.

If local MLflow support is installed,
`uv run dl-init --with-mlflow` will also scaffold an `mlflow`
callback block and local tracking defaults.

If W&B support is installed, `uv run dl-init --with-wandb` will also
scaffold a `wandb` callback block, W&B tracking defaults, and `.env.example`.

Select at most one explicit sweep tracker; Azure can still provide the executor
alongside W&B or local MLflow tracking. In-place init patches
supported existing config and bootstrap files while preserving their other
content; it never replaces a project-owned component file.

## Companion Packages

- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-robotics`](https://github.com/Blazkowiz47/dl-robotics)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Scaffold Commands

Each `dl-core add ...` command creates the new module and updates the matching
local package `__init__.py` export list under `src/`.

`uv run dl-core describe ...` now also shows a minimal YAML snippet for common
config-backed component types such as datasets, models, callbacks, optimizers,
and trainers.

Common local component scaffolds:

```bash
uv run dl-core add model MyResNet
uv run dl-core add trainer MyTrainer
uv run dl-core add trainer EpochTrainer --base epochtrainer
uv run dl-core add trainer MyPolicy --base rltrainer
uv run dl-core add callback MyMetrics
uv run dl-core add metric_manager MyManager
uv run dl-core add episode_manager MyEpisodeManager
uv run dl-core add sampler MySampler
uv run dl-core add optimizer MyOptimizer
uv run dl-core add scheduler MyScheduler
uv run dl-core add criterion MyLoss
uv run dl-core add augmentation MyAugmentation
uv run dl-core add metric MyMetric
uv run dl-core add executor MyExecutor
```

Default-base scaffolds for augmentations, metrics, metric managers,
criterions, models, and executors now start with ready-to-edit method stubs
instead of empty wrapper subclasses.

See [Local Components and Sweeps](readme/guide/3_local_components_and_sweeps.md)
for component implementation rules and the recommended `compute_forward()`
structure.

Sweep scaffolds are supported too:

```bash
uv run dl-core add sweep DebugSweep
uv run dl-core add sweep AzureEval --tracking azure_mlflow
uv run dl-core add sweep MlflowBaseline --tracking mlflow
uv run dl-core add sweep WandbAblation --tracking wandb
```

Generated sweep files:

- live under `experiments/`
- extend `../configs/base_sweep.yaml`
- include runnable defaults in `fixed`
- start with `grid: {}`
- default the tracker experiment destination to the repository root name unless
  `tracking.experiment_name` overrides it
- let the tracker derive sweep grouping from the filename unless
  `tracking.sweep_name` overrides it

Project-specific criterions, optimizers, and schedulers can still be added
later with `uv run dl-core add ...` when they are actually needed.

You can inspect registered components and built-in base classes directly from
the CLI:

```bash
uv run dl-core list
uv run dl-core list sampler
uv run dl-core list metric_manager --json
uv run dl-core describe dataset my_dataset --root-dir .
uv run dl-core describe model my_resnet --root-dir .
uv run dl-core describe class dl_core.core.FrameWrapper
uv run dl-core describe class dl_azure.datasets.AzureComputeMultiFrameWrapper
uv run dl-core describe dataset my_dataset --root-dir . --json
```

The built-in sampler list now includes `label`, which balances samples by any
metadata key using either `undersample` or `oversample`.

Example sampler config:

```yaml
dataset:
  sampler:
    label:
      key: attack
      mode: undersample
```

The describe command shows:

- resolved class and registered names
- constructor signature
- inheritance chain
- docstring
- declared properties
- class-level attributes
- public methods defined on the class

It does not discover instance attributes created dynamically inside `__init__`
without constructing the class.

Scaffolds can target a specific base when you need one:

```bash
uv run dl-core add dataset MyDataset
uv run dl-core add dataset FrameSet --base frame
uv run dl-core add dataset TextSet --base text_sequence
uv run dl-core add dataset ActSet --base adaptive_computation
uv run dl-core add dataset TarSet --base tar_shard
uv run dl-core add callback EpochLogger --base metric_logger
uv run dl-core add metric_manager PadMetrics --base standard
uv run dl-core add optimizer AdamwWrapper --base adamw
uv run dl-core add scheduler CosineWrapper --base cosine
```

Built-in callbacks include `dataset_refresh`, which rebuilds selected dataset
splits at epoch boundaries by default. With `IterationTrainer`, opt into
rebuilding at data-cycle boundaries:

```yaml
callbacks:
  dataset_refresh:
    trigger: data_cycle
    refresh_frequency: 1
    splits: [train]
```

Use `trigger: epoch` for the existing epoch behavior. `refresh_frequency`
counts the selected boundary; a frequency of two retains a selection for two
cycles. The initial or resumed cycle rebuilds its active selection even when
its number is between scheduled refreshes. Other splits retain their loaders.

`on_data_cycle_start(cycle, logs)` and `on_data_cycle_end(cycle, logs)` run on
every rank. Start callbacks run after the wrapper's cycle/epoch is set and
before the prepared loader's iterator is created. Callbacks run in config
order: put prefetch preparation before `dataset_refresh`, and callbacks that
inspect the rebuilt loader after it. End logs include `completed` and `reason`,
distinguishing exhaustion from a partial cycle at the end of training. Cycle
callback failures stop all ranks rather than disable a required refresh.

`TarShardWrapper.refresh_dataset(split)` clears resolved and sampled source
lists and shard progress for that split. Its next `get_split()` calls the
concrete `build_shard_sources(split)` again. Quotas, frozen selections, and
bounded pools remain the concrete wrapper's responsibility. Workers finish
before retired datasets and their cache reservations are released. Refresh
preserves optimizer state and accumulated gradients.

Wrappers can override `get_data_cycle_state()` and
`restore_data_cycle_state(state)` to persist a fixed candidate pool or inventory
identity. State must be serializable, identical across ranks, and free of
credentials or process-local resources. Tar wrappers record logical train
shard identities and available ETags, hashes, and sample counts without signed
URL queries. Resume checks the rebuilt state and requires the same world size;
selection changes raise before training. Reproducing sample order also requires
the same seeds, reader settings, and deterministic selection/transform behavior.

When `dl-azure` is importable, the dataset scaffold also exposes Azure bases:

```bash
uv run dl-core add dataset AzureFrames --base azure_compute_frame
uv run dl-core add dataset AzureSeq --base azure_compute_multiframe
uv run dl-core add dataset AzureStream --base azure_streaming
uv run dl-core add dataset AzureStreamSeq --base azure_streaming_multiframe
uv run dl-core add dataset AzureTar --base azure_streaming_tar
```

Plain `deep-learning-core` currently exposes dataset bases for:

- `BaseWrapper`
- `FrameWrapper`
- `TextSequenceWrapper`
- `AdaptiveComputationDataset`
- `TarShardWrapper`

`TextSequenceWrapper` adds sequence-aware batch padding for tokenized inputs.
`AdaptiveComputationDataset` adds per-class sample stream helpers for
adaptive-time computation trainers. Multiframe dataset bases are still
provided through `dl-azure`.

`TarShardWrapper` uses the optional `webdataset` package to stream members such
as `sample.png` and `sample.json` as one grouped sample. Project wrappers
implement `transform()` and receive the grouped bytes in
`file_dict["members"]`. WebDataset performs shard/sample shuffling and splits
the shard stream between distributed ranks and DataLoader workers.

Explicit `dataset.shards` are optional. Project wrappers can construct paths and
weights together:

```python
from dl_core.datasets import TarShardWrapper


class MobaiTarWrapper(TarShardWrapper):
    def build_shard_sources(self, split: str) -> list[dict]:
        return [
            {
                "name": "bonafide",
                "weight": 0.5,
                "shards": self.find_bonafide_shards(split),
            },
            {
                "name": "replay",
                "weight": 0.3,
                "shards": self.find_replay_shards(split),
            },
            {
                "name": "print",
                "weight": 0.2,
                "shards": self.find_print_shards(split),
            },
        ]
```

Each shard may be a path string or a metadata dictionary containing `path`.
Positive-weight sources are passed to WebDataset `RandomMix`. The weights are
probabilistic rather than an exact within-batch composition and are most useful
with a resampled iteration-based training stream.

For finite iterable streams, `EpochTrainer` stops training at the shortest rank's
last shared batch by default; longer ranks leave their remaining tail unused.
`IterationTrainer` covers longer ranks by replaying shorter ranks within the
same cycle, and also supports resampled or endless training streams. A rank
with no valid training batches fails consistently across ranks. Empty collated
batches from skipped transforms do not advance the iteration or cycle cursor.
Validation and test instead process every valid shard sample, even when a rank
has no batches. Use metric-manager `gather` mode for uneven evaluation; a
rank-average metric is rejected when sample counts differ. Models and batch
callbacks should not perform their own distributed collectives during
evaluation.

Missing grouped members raise by default. With `strict_pairs: false`, those
samples are skipped with a warning, so realized sample counts may differ from
shard inventories or project-specific quotas. Other transform errors still
raise unless the project handles them explicitly.

### Indexed plain tar utilities

Concrete wrappers opt in by calling `build_indexed_dataset(data, split)` from
their `build_dataset()` implementation. It returns an `IndexedTarDataset` that
uses the existing `transform(file_dict, split)` contract. It supports plain
`.tar` files; members can have different sizes. Data stays in the tar, and the
reader seeks to indexed member offsets without extracting or decoding the
whole shard into memory.

```python
from dl_core.datasets import TarShardWrapper


class IndexedImages(TarShardWrapper):
    def build_dataset(self, data: list[dict], split: str):
        return self.build_indexed_dataset(data, split)

    def transform(self, file_dict: dict, split: str) -> dict:
        return {
            "image_bytes": file_dict["members"]["jpg"],
            "metadata_bytes": file_dict["members"]["json"],
        }
```

`num_workers` is the total worker pool shared across selected shards. With
four shards and eight workers, those eight workers can read samples from any
of the four shards. Each process opens its own bounded set of file handles.
Indexes are reused until the local tar's identity, size, or modification time
changes. Indexed utilities use PyTorch and the Python standard library; they
do not require the optional WebDataset package.

```yaml
dataset:
  num_workers: 8
  track_shard_progress: true
  indexed_tar:
    index_dir: /mnt/localssd/tar-indexes  # null disables persistent indexes
    index_workers: 4                   # Startup indexing; 1 runs sequentially
    max_open_shards: 8                 # Per worker
    replacement: false                # Visit each selected sample at most once
```

`index_workers` controls index construction before the DataLoader starts.
It is independent of `num_workers` and any consumer's `preparation_workers`.
Valid indexes are reused without starting a pool; cache misses use spawned
processes when more than one worker is available. `LOCAL_WORLD_SIZE` reduces
the pool per local rank, and file locks cap simultaneous builds across processes
sharing `index_dir`. Use the same directory and worker limit on each rank;
independent directories have independent limits. One worker scans sequentially
in a background thread so the parent can report progress and handle Ctrl-C.
Standalone Python scripts using multiple indexing workers need the usual
`if __name__ == "__main__":` entry-point guard.

At INFO level, indexing logs start, progress at roughly five-second intervals,
and completion with shard counts, cache hits, builds, and elapsed time. Direct
`IndexedTarDataset` users can pass `index_workers`, `logger`, and `index_label`;
`dataset.index_stats` exposes `shards`, `cached`, `built`, `workers` (per rank),
and `seconds`. Existing version-1 index caches remain compatible.

Finite indexed mixing visits all selected samples by default; source weights
affect their ordering. Set `replacement: true` and `num_samples: 10000` for a
bounded weighted repetition pass. Source probability is independent of how
many samples that source contains. `shuffle: false` gives ordered finite
reads. These options are independent of `webdataset` stream settings. The
built-in indexed sampler targets one GPU; concrete wrappers can supply their
own sampler through `build_batch_sampler()`.

Shard records can supply `sample_keys` to select an eligible subset before
sampling. Missing required members follow `strict_pairs`; a transform returning
`None` is omitted from its batch. An entirely skipped batch is `{}` and the
trainer should skip it. Use eligible counts when transforms can filter samples.

When `track_shard_progress` is enabled, the wrapper preserves a stable
`shard_id` in each transformed output. The trainer starts a pass with
`reset_shard_progress(totals)`, then calls
`record_shard_consumption(batch["shard_id"])` after completing each batch.
`get_shard_progress()` returns per-shard `consumed`, `total`, and `fraction`.
Unknown totals are `None`; finite budgets are required for repeated sampling.
Counters live in the owning training process and reset explicitly for a new
pass. Direct consumers can import `IndexedTarDataset`, `IndexedTarSampler`,
and `ShardProgress` from `dl_core.datasets`.

For storage integrations, hold reservations for the local files while their
dataset and workers are in use. See
[configuration and progress examples](readme/technical/1_configuration.md#indexed-plain-tar-reading).

## Releases

- `Publish TestPyPI` publishes to TestPyPI for release verification.
- `Publish` is the production workflow for PyPI.
- Trusted publishing is configured through GitHub Actions environments rather
  than long-lived API tokens.
- The publish action may upload digital attestations alongside the package.
  That is expected behavior from `pypa/gh-action-pypi-publish`.
- Package metadata keeps runtime dependencies unpinned, so the consuming
  environment resolves the latest compatible public releases.

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-core/tree/master/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-core)

## License

MIT. See [LICENSE](LICENSE).

## Development Validation

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests
uv run python -m compileall src/dl_core
uv build --no-sources
```
