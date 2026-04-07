# TODO

Agreed next improvements for `deep-learning-core`, in working order:

- [x] Strengthen `dl-run --validate-only` into a real preflight that resolves
  configured components and prints an effective summary without starting
  training.
- [ ] Add a clean matrix preview/export mode to `dl-sweep`.
- [ ] Add report comparison support to `dl-analyze`.
- [ ] Add a separate artifact sync/download command for remote-backed sweeps.
- [ ] Show minimal example config snippets in `dl-core describe`.
- [ ] Expand `dl-core add` with more ready-to-edit scaffold variants.
- [ ] Add a lightweight smoke runner helper for one-batch checks.
- [ ] Harden interrupt/finalization handling, especially for multi-worker runs.
- [ ] Add a dataset inspection command for split sizes and sample structure.
- [ ] Add `dl-sweep` filters such as `--only` and `--skip`.
