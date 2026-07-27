# deep-learning-core Release History

The main README shows only the latest release. This page preserves the
release-by-release changes that were previously shown there.

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
- `BaseTrainer` now exposes `select_checkpoint()` and
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
