# Dynamic Extension Plan

## Goal

Make `dl-core` the base package and support these install forms cleanly:

- `dl-core`
- `dl-core[azure]`
- `dl-core[wandb]`
- `dl-core[azure,wandb]`

The key requirement is to keep one shared scaffold engine in `dl-core` and let
extension packages contribute small, targeted changes without duplicating the
base initialization flow.

## Target Design

### Package Layout

- `dl-core`
  - owns base scaffold generation
  - owns component registries, local execution, sweeps, analysis, docs
  - discovers installed scaffold/runtime extensions
- `dl-azure`
  - depends on `dl-core`
  - provides Azure executor integration and Azure scaffold additions
- `dl-wandb`
  - depends on `dl-core`
  - provides W&B tracking integration and W&B scaffold additions

### Extras

`dl-core` should expose install extras that pull in extension packages:

- `azure = ["dl-azure>=0.1.0,<0.2"]`
- `wandb = ["dl-wandb>=0.1.0,<0.2"]`

That gives the public UX:

- `uv add dl-core`
- `uv add "dl-core[azure]"`
- `uv add "dl-core[wandb]"`
- `uv add "dl-core[azure,wandb]"`

## Main Principle

Do not fork the scaffold.

There should be:

- one `dl-init`
- one base experiment template
- one extension hook system

Azure and W&B should only provide deltas on top of the base scaffold.

## Plugin Model

### Discovery

Use Python entry points for extension discovery.

Suggested entry point group:

- `dl_core.init_extensions`

Each installed extension package registers one plugin object under that group.

### Plugin Responsibilities

An init extension should be able to:

- declare its feature name, for example `azure` or `wandb`
- expose CLI flags to `dl-init`
- patch generated config files
- add extra scaffold files
- add optional README sections or notes
- register bootstrap imports if needed

### Plugin Boundaries

Keep plugin scope narrow.

Plugins should not:

- replace the full scaffold engine
- own generic project layout
- duplicate base templates

Plugins should only modify:

- generated files
- generated config content
- generated optional metadata

## `dl-init` Behavior

### Core Behavior

`dl-init` remains the scaffold entrypoint.

It should:

1. generate the base repository layout
2. discover installed extensions
3. expose extension flags in `--help`
4. apply selected extensions during scaffold generation

### Default Selection Rules

To avoid surprising behavior:

- if only `dl-core` is installed, scaffold plain mode
- if exactly one extension is installed, allow that extension to be auto-enabled
- if multiple extensions are installed, default to plain mode unless flags are
  passed explicitly

Suggested explicit control flags:

- `--with-azure`
- `--with-wandb`
- `--plain`

This avoids ambiguous auto-selection when both Azure and W&B are available.

## Azure Extension Scope

`dl-azure` should contribute:

- `--with-azure`
- sample `azure-config.json` with placeholder values only
- Azure executor config patch
- optional Azure-specific bootstrap import
- optional Azure-specific README note

The generated `azure-config.json` must contain only generic keys and dummy
values. No secrets, no company defaults.

Recommended placeholder structure:

```json
{
  "subscription_id": "<subscription-id>",
  "resource_group": "<resource-group>",
  "workspace_name": "<workspace-name>"
}
```

Keep run-specific or experiment-specific settings in YAML, not in
`azure-config.json`.

## W&B Extension Scope

`dl-wandb` should contribute:

- `--with-wandb`
- W&B tracking config patch
- optional sample environment guidance
- optional README note for grouping, project, and entity configuration

Do not put secrets in generated files.

If environment variables are needed, prefer:

- `.env.example`
- README guidance

over committed secrets or committed machine-specific values.

## Runtime Reuse Strategy

To minimize tech debt:

- keep all generic execution and analysis logic in `dl-core`
- make extension packages register only what they own
- keep executor and tracking concerns separate

Examples:

- Azure should remain an executor/infrastructure concern
- W&B should remain a tracking/logging concern

This matters for combinations such as:

- local executor + W&B tracking
- Azure executor + local tracking
- Azure executor + W&B tracking

The architecture should support those combinations without special-case code.

## Proposed Interface

### In `dl-core`

Add a small extension contract, for example:

- `InitExtension.name`
- `InitExtension.add_arguments(parser)`
- `InitExtension.is_enabled(args, discovered_extensions)`
- `InitExtension.apply(scaffold_context)`

Where `scaffold_context` holds:

- target paths
- rendered base config content
- project metadata
- selected feature flags

This lets `dl-core` own rendering while extensions mutate a controlled context.

### In Extension Packages

Each extension package should export one plugin implementation through entry
points.

That implementation should remain small and declarative.

## Implementation Phases

### Phase 1: Introduce Extension Discovery

- add entry point discovery helper in `dl-core`
- add a minimal extension interface
- keep current scaffold behavior unchanged when no extension is installed

### Phase 2: Make `dl-init` Extension-Aware

- allow extensions to register CLI options
- add explicit feature selection flow
- preserve current plain initialization path

### Phase 3: Extract Azure Scaffold Logic

- move Azure-specific scaffold changes out of `dl-core`
- create `dl-azure`
- implement Azure init plugin
- generate placeholder `azure-config.json`

### Phase 4: Add W&B Extension

- create `dl-wandb`
- implement W&B init plugin
- add tracking config patching

### Phase 5: Add Runtime Plugin Discovery

- auto-import installed extension packages so their runtime components register
- avoid requiring manual bootstrap imports where possible

### Phase 6: Documentation and Matrix Testing

- document install variants
- document scaffold outcomes
- test these combinations:
  - `dl-core`
  - `dl-core[azure]`
  - `dl-core[wandb]`
  - `dl-core[azure,wandb]`

## Testing Plan

Add smoke tests for:

- plain init with no extensions
- init with only Azure extension installed
- init with only W&B extension installed
- init with both installed and explicit flags
- local run path still working after extension discovery is introduced

Test expectations should cover:

- generated files
- generated config deltas
- CLI help content
- no duplicate scaffold logic

## Risks

### Risk 1: Hidden Import Coupling

If extensions rely on import side effects without a clear plugin discovery path,
the system will become fragile.

Mitigation:

- use explicit entry point discovery for scaffold plugins
- later add explicit runtime plugin loading for component registration

### Risk 2: Too Much Auto-Magic

If installed extras silently change the scaffold with no user control, the UX
becomes confusing.

Mitigation:

- keep explicit flags
- add `--plain`
- only auto-enable when exactly one extension is installed

### Risk 3: Public Extras Depending on Private Packages

If `dl-core[azure]` references a private package, public installs break.

Mitigation:

- make `dl-azure` public and generic if `dl-core[azure]` is public

## Recommended Execution Order

1. implement extension discovery in `dl-core`
2. refactor `dl-init` around a scaffold context
3. add the Azure plugin first
4. create the W&B plugin second
5. add runtime plugin auto-loading
6. expand docs and install examples

## Non-Goals For The First Iteration

- full secret management
- company-specific defaults
- automatic cloud credential setup
- install-time file generation

Those should stay out of scope for the initial plugin architecture.
