#!/usr/bin/env python3
"""
Entry point for single training runs (orchestration).

This file handles orchestration and dispatches work through an executor.
For sweep orchestration, use `dl-sweep` instead.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dl_core import load_builtin_components, load_local_components
from dl_core.core import (
    ACCELERATOR_REGISTRY,
    CALLBACK_REGISTRY,
    CRITERION_REGISTRY,
    DATASET_REGISTRY,
    EXECUTOR_REGISTRY,
    METRIC_MANAGER_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
)
from dl_core.core.registry import print_registry_info
from dl_core.sweep.template import ensure_tracking_experiment_name
from dl_core.utils.config_validator import validate_config
from dl_core.utils.logging import setup_logging


def _load_run_config(config_path: Path) -> dict[str, Any]:
    """Load one run config from YAML and attach the source path."""
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a mapping")

    config["_config_path"] = str(config_path)
    return config


def _resolve_trainer_name(config: dict[str, Any]) -> str:
    """Return the configured trainer name or the standard default."""
    trainer_config = config.get("trainer", {})
    if isinstance(trainer_config, dict) and trainer_config:
        return str(next(iter(trainer_config)))
    return "standard"


def _resolve_single_component_config(
    config: dict[str, Any],
    section_name: str,
    *,
    default_name: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Normalize a single-component config section."""
    section_config = config.get(section_name, {})
    if not section_config:
        return None
    if not isinstance(section_config, dict):
        raise ValueError(f"'{section_name}' must be a dict")

    if "name" in section_config:
        component_name = str(section_config.get("name") or default_name or "main")
        component_config = {
            key: value for key, value in section_config.items() if key != "name"
        }
        return component_name, component_config

    items = list(section_config.items())
    if len(items) != 1:
        raise ValueError(
            f"'{section_name}' must define exactly one component for preflight"
        )

    component_key, component_config = items[0]
    if not isinstance(component_config, dict):
        raise ValueError(f"'{section_name}.{component_key}' must be a dict")

    normalized_config = dict(component_config)
    component_name = str(normalized_config.get("name") or component_key)
    normalized_config.pop("name", None)
    return component_name, normalized_config


def _component_path(component_class: type[Any]) -> str:
    """Return a fully qualified class path."""
    return f"{component_class.__module__}.{component_class.__name__}"


def _print_component_list(title: str, entries: list[str]) -> None:
    """Print a component list when entries are present."""
    if not entries:
        return

    print(f"   {title}:")
    for entry in entries:
        print(f"     - {entry}")


def _run_preflight(
    config: dict[str, Any],
    *,
    config_path: Path,
    mode: str,
    run_name: str,
) -> None:
    """Resolve and safely instantiate configured components."""
    tracking_config = config.get("tracking", {})
    if not isinstance(tracking_config, dict):
        tracking_config = {}

    experiment_name = ensure_tracking_experiment_name(
        {
            "tracking": dict(tracking_config),
            "experiment": config.get("experiment", {}),
            "sweep_file": str(config_path),
        },
        config_path=config_path,
    )
    output_dir = config.get("runtime", {}).get("output_dir", "artifacts")
    trainer_name = _resolve_trainer_name(config)

    trainer_class = TRAINER_REGISTRY.get_class(trainer_name)
    executor_class = EXECUTOR_REGISTRY.get_class(mode)
    accelerator_type = str(config.get("accelerator", {}).get("type", "cpu"))
    accelerator_class = ACCELERATOR_REGISTRY.get_class(accelerator_type)

    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict) or not dataset_config.get("name"):
        raise ValueError("dataset.name is required for preflight")
    dataset_name = str(dataset_config["name"])
    dataset_class = DATASET_REGISTRY.get_class(dataset_name)
    DATASET_REGISTRY.get(dataset_name, dataset_config)

    models_config = config.get("models", {})
    if not isinstance(models_config, dict) or not models_config:
        raise ValueError("At least one model must be configured")

    model_entries: list[str] = []
    model_instances = []
    for model_name, model_config in models_config.items():
        resolved_config = dict(model_config or {})
        model_class = MODEL_REGISTRY.get_class(model_name)
        model = MODEL_REGISTRY.get(model_name, resolved_config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_entries.append(
            f"{model_name} -> {_component_path(model_class)} ({parameter_count:,} params)"
        )
        model_instances.append(model)

    criterion_entries: list[str] = []
    criterions_config = config.get("criterions", {})
    if not isinstance(criterions_config, dict):
        raise ValueError("'criterions' must be a mapping")
    for criterion_name, criterion_config in criterions_config.items():
        resolved_config = dict(criterion_config or {})
        resolved_config.setdefault("name", criterion_name)
        criterion_class = CRITERION_REGISTRY.get_class(criterion_name)
        CRITERION_REGISTRY.get(criterion_name, resolved_config)
        criterion_entries.append(
            f"{criterion_name} -> {_component_path(criterion_class)}"
        )

    optimizer_entry = None
    optimizer_spec = _resolve_single_component_config(
        config,
        "optimizers",
        default_name="adam",
    )
    optimizer = None
    if optimizer_spec is not None:
        optimizer_name, optimizer_config = optimizer_spec
        optimizer_class = OPTIMIZER_REGISTRY.get_class(optimizer_name)
        parameters = [
            parameter
            for model in model_instances
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("Configured models expose no trainable parameters")
        optimizer = OPTIMIZER_REGISTRY.get(
            optimizer_name,
            parameters,
            **optimizer_config,
        )
        optimizer_entry = f"{optimizer_name} -> {_component_path(optimizer_class)}"

    scheduler_entry = None
    scheduler_spec = _resolve_single_component_config(config, "schedulers")
    if scheduler_spec is not None:
        if optimizer is None:
            raise ValueError("A scheduler requires an optimizer in preflight")
        scheduler_name, scheduler_config = scheduler_spec
        scheduler_class = SCHEDULER_REGISTRY.get_class(scheduler_name)
        SCHEDULER_REGISTRY.get(scheduler_name, optimizer, **scheduler_config)
        scheduler_entry = f"{scheduler_name} -> {_component_path(scheduler_class)}"

    metric_manager_entries: list[str] = []
    metric_managers_config = config.get("metric_managers", {})
    if not isinstance(metric_managers_config, dict):
        raise ValueError("'metric_managers' must be a mapping")
    for manager_name in metric_managers_config:
        manager_class = METRIC_MANAGER_REGISTRY.get_class(manager_name)
        metric_manager_entries.append(
            f"{manager_name} -> {_component_path(manager_class)}"
        )

    callback_entries: list[str] = []
    callbacks_config = config.get("callbacks", {})
    if not isinstance(callbacks_config, dict):
        raise ValueError("'callbacks' must be a mapping")
    for callback_name in callbacks_config:
        callback_class = CALLBACK_REGISTRY.get_class(callback_name)
        callback_entries.append(
            f"{callback_name} -> {_component_path(callback_class)}"
        )

    print("\n✓ Preflight complete")
    print(f"   Config: {config_path}")
    print(f"   Mode: {mode}")
    print(f"   Run name: {run_name}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Output dir: {output_dir}")
    print(f"   Executor: {mode} -> {_component_path(executor_class)}")
    print(f"   Accelerator: {accelerator_type} -> {_component_path(accelerator_class)}")
    print(f"   Trainer: {trainer_name} -> {_component_path(trainer_class)}")
    print(f"   Dataset: {dataset_name} -> {_component_path(dataset_class)}")
    _print_component_list("Models", model_entries)
    _print_component_list("Criterions", criterion_entries)
    if optimizer_entry is not None:
        print(f"   Optimizer: {optimizer_entry}")
    if scheduler_entry is not None:
        print(f"   Scheduler: {scheduler_entry}")
    _print_component_list("Metric managers", metric_manager_entries)
    _print_component_list("Callbacks", callback_entries)
    print("   No training was started.")


def main():
    """Main orchestration function for single runs."""
    parser = argparse.ArgumentParser(
        description="Run one training configuration through the configured executor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the scaffolded base config
  dl-run --config configs/base.yaml

  # Run with a custom runtime name
  dl-run --config configs/base.yaml --name baseline_debug

  # Validate config and resolve components without starting training
  dl-run --config configs/base.yaml --validate-only

  # Dry-run (show what would be executed)
  dl-run --config configs/base.yaml --dry-run

  # Show registered components
  dl-run --show-registry

Typical first use:
  1. Run dl-init
  2. Implement your dataset wrapper under src/datasets/
  3. Update configs/base.yaml
  4. Run dl-run --config configs/base.yaml --validate-only
  5. Run dl-run --config configs/base.yaml
        """,
    )

    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file")

    parser.add_argument(
        "--name",
        type=str,
        help="Run name (default: config filename without extension)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["local"],
        default="local",
        help="Execution mode (default: local).",
    )

    # Azure-specific args
    parser.add_argument(
        "--compute",
        type=str,
        help="Executor-specific compute target override when supported.",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="dl_lab",
        help="Executor-specific environment override (default: dl_lab).",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, resolve components, and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--show-registry",
        action="store_true",
        help="Show registered components and exit",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    load_builtin_components()
    load_local_components(args.config)

    # Show registry if requested
    if args.show_registry:
        print_registry_info()
        return 0

    # Config required for all other operations
    if not args.config:
        parser.print_help()
        print("\nError: --config is required (unless using --show-registry)")
        return 1

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Validate config
    print("Validating configuration...")
    is_valid = validate_config(str(config_path), verbose=True)

    if not is_valid:
        print("✗ Config validation failed. Aborting.")
        return 1

    run_config = _load_run_config(config_path)

    if args.validate_only:
        try:
            _run_preflight(
                run_config,
                config_path=config_path,
                mode=args.mode,
                run_name=args.name or config_path.stem,
            )
            return 0
        except Exception as exc:
            print(f"✗ Preflight failed: {exc}")
            return 1

    # Config log level takes priority over default CLI arg
    # Explicit CLI arg still overrides config
    config_log_level = run_config.get("runtime", {}).get("log_level")
    effective_log_level = config_log_level or args.log_level

    setup_logging(effective_log_level)
    if config_log_level:
        print(f"   Log level: {effective_log_level} (from config)")
    else:
        print(f"   Log level: {effective_log_level} (from CLI)")

    # Determine run name (default to config filename without extension)
    run_name = args.name or config_path.stem

    print("\n🚀 Training Run")
    print(f"   Config: {config_path}")
    print(f"   Mode: {args.mode}")
    print(f"   Run name: {run_name}")

    tracking_config = run_config.get("tracking", {})
    if not isinstance(tracking_config, dict):
        tracking_config = {}
    tracking_payload = dict(tracking_config)
    single_run_tracking = {
        "tracking": tracking_payload,
        "experiment": run_config.get("experiment", {}),
        "sweep_file": str(config_path),
    }
    experiment_name = ensure_tracking_experiment_name(
        single_run_tracking,
        config_path=config_path,
    )

    # Create minimal sweep config for executor initialization
    # (executors expect sweep_config even for single runs)
    sweep_config = {
        "sweep_file": str(config_path),
        "tracking": tracking_payload,
        "executor": {"mode": args.mode},
        "accelerator": run_config.get("accelerator", {}),
        "grid": {},  # Empty grid for single run
    }

    # Create executor
    run_id = f"run_{int(time.time())}"

    executor = EXECUTOR_REGISTRY.get(
        args.mode,
        sweep_config,
        experiment_name,
        run_id,
        dry_run=args.dry_run,
        max_workers=1,
        compute_target=args.compute,
        environment_name=args.environment,
    )

    # Execute run
    try:
        success = executor.run(str(config_path), run_name=run_name)

        if success:
            print(f"\n✓ Run '{run_name}' completed successfully")
            return 0
        else:
            print(f"\n✗ Run '{run_name}' failed")
            return 1

    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
