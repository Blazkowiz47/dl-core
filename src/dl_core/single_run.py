#!/usr/bin/env python3
"""
Entry point for single training runs (orchestration).

This file handles orchestration and dispatches work through an executor.
For sweep orchestration, use `dl-sweep` instead.
"""

import argparse
import sys
import time
import yaml
from pathlib import Path

from dl_core import load_builtin_components, load_local_components
from dl_core.core import EXECUTOR_REGISTRY
from dl_core.core.registry import print_registry_info
from dl_core.utils.config_validator import validate_config
from dl_core.utils.logging import setup_logging


def main():
    """Main orchestration function for single runs."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Lab - Single Run Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config (local execution)
  dl-run --config configs/baseline.yaml
  
  # Train with custom name
  dl-run --config configs/baseline.yaml --name test_run
  
  # Validate config only
  dl-run --config configs/baseline.yaml --validate-only
  
  # Dry-run (show what would be executed)
  dl-run --config configs/baseline.yaml --dry-run
  
  # Show registered components
  dl-run --show-registry
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
        help="Execution mode (default: local)",
    )

    # Azure-specific args
    parser.add_argument(
        "--compute",
        type=str,
        help="Azure compute target (required for azure mode)",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="dl_lab",
        help="Azure environment name (default: dl_lab)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config, do not train",
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

    if args.validate_only:
        print("✓ Validation complete. Exiting (--validate-only)")
        return 0

    # Load config to extract accelerator info for executor
    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)

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

    # Create minimal sweep config for executor initialization
    # (executors expect sweep_config even for single runs)
    sweep_config = {
        "sweep_file": str(config_path),
        "executor": {"mode": args.mode},
        "accelerator": run_config.get("accelerator", {}),
        "grid": {},  # Empty grid for single run
    }

    # Create executor
    experiment_name = f"run_{run_name}"
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
