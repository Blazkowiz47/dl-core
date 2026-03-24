# Technical: 4. Local Component Loading

`dl-core` supports two kinds of component loading.

## Built-In Components

`load_builtin_components()` imports the package modules that own built-in
registrations:

- accelerators
- augmentations
- callbacks
- criterions
- datasets
- executors
- metric managers
- metrics
- models
- optimizers
- schedulers
- trainers

## Local Experiment Components

`load_local_components()` looks for the nearest parent directory that contains:

- `pyproject.toml`
- `src/`

If it finds that project root, it:

1. adds `src/` to `sys.path`
2. imports each top-level package in `src/`
3. skips `dl_core`

That is why the experiment package wrappers in `src/my_exp/` register
automatically when you run `dl-run` or `dl-sweep` from inside the experiment
repository.

## Practical Implication

The scaffolded experiment package is the registration boundary. Keep
project-specific components there, not inside `dl-core`.
