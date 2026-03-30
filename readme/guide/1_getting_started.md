# Guide: 1. Getting Started

`dl-core` is not meant to be the place where you keep concrete experiment
configs and datasets. It is the reusable framework package. The normal workflow
is:

1. develop and version `dl-core`
2. create a separate experiment repository with `dl-init-experiment`
3. install `dl-core` into that repository
4. keep your configs and local wrappers in the experiment repository

## Step 1: Install the Package Repo

```bash
cd dl-core
uv sync
```

## Step 2: Verify the Built-In Components

```bash
uv run dl-core list
uv run dl-core list sampler
```

This confirms the package entrypoints work and the built-in registries are
populated. The sampler listing should include the built-in `attack` sampler.

## Step 3: Run the Package Tests

```bash
uv run --extra dev pytest
```

## Step 4: Scaffold a Consumer Repository

```bash
uv run dl-init-experiment --name my-exp --root-dir .
```

To initialize the current directory in place:

```bash
uv run dl-init-experiment --root-dir .
```

The generated repository becomes your working area for experiments. That is
where you edit configs, add local wrappers, and run training jobs.

## Step 5: Validate the Generated Repo

Inside the generated repository:

```bash
uv add --editable ../dl-core
uv sync
uv run dl-run --config configs/base.yaml
```

If that succeeds, your local package plus experiment repo integration is
working.
