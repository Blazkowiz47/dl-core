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
2. imports `bootstrap.py` when present
3. imports each known component package in `src/`

That is why local wrappers in `src/datasets/`, `src/models/`, `src/trainers/`,
and other component folders register automatically when you run `dl-run` or
`dl-sweep` from inside the experiment repository.

## Practical Implication

The scaffolded `src/` tree is the registration boundary. Keep project-specific
components there, not inside `dl-core`.
