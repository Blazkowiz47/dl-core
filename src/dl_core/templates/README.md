# dl-core Templates

These files back the `dl-init` command.

They are intentionally generic:

- standard trainer
- standard dataset
- project-owned ResNet baseline
- local executor defaults
- root-level reproducibility defaults in `base.yaml` via `seed` and `deterministic`

Azure-specific execution should be layered on top through `dl-azure`.
