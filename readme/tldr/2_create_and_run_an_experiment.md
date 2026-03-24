# TLDR: Create and Run an Experiment

The default path is:

- scaffold a new experiment repository
- use the generated project-named dataset and trainer wrappers
- keep the default `ResNetExample` model wrapper
- run locally first

## 1. Create the Experiment Repository

```bash
uv run dl-init-experiment --name my-exp --root-dir .
cd my-exp
```

Or, if you are already inside the empty target directory:

```bash
uv run dl-init-experiment --root-dir .
```

## 2. Install the Experiment Repository

For local development against a sibling checkout of `dl-core`:

```bash
uv add --editable ../dl-core
uv sync
```

## 3. Inspect the Generated Defaults

The scaffold gives you:

- `configs/base.yaml`
- `configs/base_sweep.yaml`
- `configs/sweeps/example_sweep.yaml`
- `src/my_exp/datasets/my_exp.py`
- `src/my_exp/trainers/my_exp.py`
- `src/my_exp/models/resnet_example.py`

## 4. Run Locally

```bash
uv run dl-run --config configs/base.yaml
uv run dl-sweep --sweep configs/sweeps/example_sweep.yaml
```

The generated base config defaults to a synthetic dataset path so the scaffold
can run without a real dataset while you validate the pipeline.
