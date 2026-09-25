# deep-learning-core Release History

The main README shows only the latest release. This page preserves the
release-by-release changes that were previously shown there.

## 0.1.9

- custom sweep name templates may choose their grid fields; final run names
  must still be unique
- `dl-sweep --overwrite` explicitly permits non-interactive replacement of
  existing sweep data and generated configs

## 0.1.8

- checkpoint writes are atomic, corrupt auto-resume candidates fail loudly,
  and trainer state restores consistently across epoch, iteration, and RL runs
- gradient accumulation, final reporting, early stopping, and probability
  diagnostics handle their default-path edge cases
- sweep execution preserves selected grid indices and run names, checks name
  uniqueness, rejects edited selected runs on resume, and confirms fresh
  tracker overwrites
- sweep defaults honor explicit accelerator and worker settings, and parallel
  interruption no longer pre-dispatches extra local runs
- runtime extensions load independently, scaffold edits are safer, and
  standalone artifacts use one run-name directory rather than a nested pair
- the PyTorch requirement is `torch>2.3` with no upper cap

## 0.1.7

- standard training preserves accumulated gradients, clips only when an
  optimizer update occurs, advances schedulers with real updates, and flushes
  partial final accumulation windows with the correct averaging factor
- FP16 gradients are unscaled before gradient clipping
- automatic validation and test partitions are created from raw records before
  sampling, preventing oversampled identities from leaking across splits
- dataset access no longer resets global RNG state; DataLoader workers receive
  deterministic epoch-varying seeds and dataset samplers receive epoch updates
- iteration-based training restores train mode after baseline evaluation

## 0.1.6

- repository and generated experiment guidance now asks authors to keep new
  components direct, avoid one-off helpers, and delay shared abstractions until
  multiple components need them
- model stubs and the generated ResNet example organize `compute_forward()` as
  input preparation, an ordered model pass, and final output construction
- generated guidance requires checking the official PyTorch releases before
  naming the latest stable version and distinguishes older compatibility pins

## 0.1.5

- tar datasets use optional WebDataset pipelines for grouped streaming,
  buffered shuffling, resampling, and distributed shard splitting
- project wrappers can build weighted shard sources dynamically through one
  override hook backed by WebDataset `RandomMix`
- the custom sidecar index, tar handle pool, and round-robin tar sampler were
  removed

## 0.1.4

- `IterationTrainer` provides fixed-batch training with deterministic loader
  cycling, iteration-based lifecycle frequencies, and resumable cursor state;
  the ambiguous `BaseTrainer` alias has been removed
- `TarShardDataset` indexes grouped samples in uncompressed tar files without
  extraction and reuses worker-local tar handles for direct sample reads
- `RoundRobinTarBatchSampler` builds deterministic, group-aware batches and
  partitions complete batches across distributed ranks without duplication
- dataset scaffolding includes local and Azure-compatible tar-shard bases

## 0.1.1

- `IterationTrainer` adds fixed-batch training with deterministic loader
  cycling, iteration-based lifecycle frequencies, and resumable cursor state
- the ambiguous `BaseTrainer` alias is removed in favor of explicit trainer
  selection
- indexed tar datasets and group-aware round-robin sampling support direct,
  distributed reads from uncompressed archives
- versions 0.1.2 and 0.1.3 were skipped in the repository release line because
  older TestPyPI artifacts already occupied those versions

## 0.1.0

- neural architectures belong exclusively to experiment repositories while
  `deep-learning-core` provides reusable training loops and registries
- DQN, PPO, SAC, and Dreamer require explicit project-owned model roles and
  validate their tensor or protocol contracts during setup and updates
- neutral Dreamer state/output types support arbitrary project world models
- generated projects include a self-contained ResNet example and declare
  `torchvision` in the experiment rather than the core runtime
- migration errors and docs identify older `dl_core.models` imports and their
  project-local replacements

## 0.0.35

- `DreamerTrainer` adds recurrent, discrete-action model-based RL through a
  categorical world model, latent imagination, and actor-critic learning
- episode-safe `SequenceReplayBuffer` sampling supports burn-in, vector lanes,
  ring overwrite, and exact checkpoint continuation
- recurrent policy state is carried between real environment steps and reset
  independently for completed vector lanes
- public observation, action-distribution, reward-prediction, and continuation
  hooks expose the main researcher customization boundaries
- a complete vectorized CartPole configuration demonstrates the world model,
  separate optimizers, sequence replay, evaluation, and checkpoint settings

## 0.0.34

- researcher extension hooks use public names across RL transition shaping,
  update schedules, augmentations, metric metadata, and epoch logs
- RL callback integrations override the public `on_*()` contract
- private methods are reserved for internal framework implementation
- the README and RL guide list the old-to-new hook migration names

## 0.0.33

- RL trainers expose scalar and vector transition-preparation hooks for
  research-specific reward shaping and replay transformations
- custom hooks receive isolated termination flags and metadata, must preserve
  vector-lane alignment, and leave episode bookkeeping faithful to the world
- trainers that do not override preparation retain the zero-copy collection
  path

## 0.0.32

- async vector environments expose split dispatch/wait operations, and DQN can
  overlap the next environment step with replay learning by default
- DQN reports collector and learner phase timings so environment, replay,
  model-update, and actor synchronization bottlenecks remain attributable
- the single-GPU accelerator can compile all models or selected runtime model
  keys in place, while DQN keeps variable inference outside learner graphs
- local auto-resume recognizes numbered step, episode, and legacy epoch
  checkpoints when `latest.pth` is unavailable

## 0.0.31

- DQN can shard vector-environment inference over configurable read-only actor
  copies and one CUDA stream per copy on a single GPU
- actor snapshots synchronize from the online policy at a configurable
  optimizer-step interval and expose policy-version and lag metrics
- deterministic evaluation uses the authoritative online policy, while
  checkpoints recreate derived actor copies without duplicating their weights

## 0.0.30

- Gymnasium vector environments now use separate asynchronous processes by
  default, while explicit synchronous collection remains available
- RL trainers can save numbered checkpoints by transition count with
  `checkpoint_frequency_steps`
- RL trainers can display step or episode progress with `show_progress`

## 0.0.29

- DQN and SAC can gate eligible replay updates through
  `should_update(global_step, transitions)`
- environments can append configurable Box or one-hot discrete action histories
  to scalar and vector observations while preserving terminal `final_obs`
- DQN and SAC support vector-safe n-step replay returns with shortened episode
  tails, matching bootstrap discounts, and backward-compatible checkpoints

## 0.0.28

- RL collection, replay insertion, episode persistence, environment creation,
  and RL component scaffolding now keep one-off logic inline for a more direct
  implementation
- public trainer, environment, episode-manager, and scaffold behavior remains
  unchanged

## 0.0.27

- episode managers provide generic RL summaries and selective, complete
  trajectory capture alongside the existing metric-manager system
- scalar and same-step vector environments share one collector contract with
  preserved terminal observations and per-lane episode identity
- Q-learning, DQN, PPO, and SAC now consume vector collection natively; neural
  policies perform batched inference and replay/rollout storage preserves the
  correct algorithm-specific scheduling and boundary semantics
- replay insertion and PPO rollout/GAE computation operate on real batches,
  while scalar custom trainers remain compatible through the original hooks
- extension packages can import and register environments without depending on
  dl-core import order

## 0.0.26

- Gymnasium-compatible environments can now be registered, discovered, and
  created through a first-class environment contract
- `RLTrainer` provides an episode-based lifecycle, deterministic evaluation,
  RL callback hooks, and resumable algorithm checkpoints alongside
  `EpochTrainer`
- `QLearningTrainer` adds tabular epsilon-greedy learning for finite discrete
  Gymnasium environments
- `DQNTrainer` adds replay-based discrete control with target networks,
  Double-DQN targets, and a built-in MLP Q-network
- `PPOTrainer` adds clipped on-policy optimization, GAE, and discrete or
  bounded-continuous actor-critic policies
- `SACTrainer` adds replay-based bounded-continuous control with twin critics,
  Polyak targets, and optional automatic entropy tuning
- `dl-run --validate-only` now resolves RL environments, models, optimizers,
  and callbacks without resetting or stepping an environment
- `dl-core add trainer MyPolicy --base rltrainer` scaffolds custom algorithms
  against the episode-oriented lifecycle
- the local metric callback records RL episode, update, and evaluation metrics
- local component loading now cleans up registrations and import paths between
  projects, while registry lookups prefer the most specific matching prefix
- configuration validation rejects malformed root and component structures
  more consistently

## 0.0.25

- `dl-init` is now the primary scaffold command
- `dl-core list` makes built-in and local registry discovery easier
- `dl-core add` defaults to plain base classes unless `--base` is given
- `dl-analyze` is the primary sweep-analysis CLI
- `dl-analyze` now supports explicit ranking metrics and rank methods
- `dl-analyze` now persists `analysis_cache.json` next to `sweep_tracking.json`
- `dl-analyze` now writes versioned reports under `analysis/vN.md`
- `dl-sync --sweep ... --artifacts` now syncs tracked remote artifacts into the
  local repository when the active backend supports it
- EMA checkpoints now include a drop-in `ema_models_state_dict` alongside the
  normal training weights and EMA resume metadata
- `dl-run --validate-only` now performs a real preflight by resolving the
  configured components without starting training
- dataset-driven trainers now expose `select_checkpoint()` and
  `post_training(checkpoint_path)` hooks for completed-run evaluation or export
  work; the default checkpoint selection uses `best.pth` then `latest.pth`
- `dl-inspect-dataset` now summarizes split sizes and one collated batch from
  the current config
- `dl-smoke` now checks one dataset batch and one model forward pass from a
  config file
- local artifacts now use:
  - `artifacts/runs/<run_name>/...`
  - `artifacts/sweeps/<sweep_name>/<run_name>/...`

Structured release notes begin with 0.0.25. Earlier package history remains
available through the repository's Git history.
