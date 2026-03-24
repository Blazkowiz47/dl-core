"""Main sweep runner with pluggable executors."""

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dl_core import load_builtin_components, load_local_components
from dl_core.core import EXECUTOR_REGISTRY
from dl_core.utils.logging import setup_logging
from dl_core.utils.sweep_tracker import SweepTracker
from .config import ConfigBuilder
from .template import load_user_sweep, generate_experiment_name


def generate_all_run_configs(
    sweep_config: Dict[str, Any], base_config: Dict[str, Any]
) -> Tuple[ConfigBuilder, List[Dict[str, Any]]]:
    """
    Generate all run configs from sweep config.

    Args:
        sweep_config: Sweep configuration
        base_config: Base training configuration

    Returns:
        Tuple of ConfigBuilder and list of complete run configurations
    """
    # Use ConfigBuilder to generate run configs
    builder = ConfigBuilder(sweep_config)
    seeds = sweep_config.get("seeds", [base_config.get("seed", 42)])
    return builder, builder.generate_run_configs(base_config, seeds)


def get_config_output_dir(sweep_config: Dict[str, Any], sweep_id: str) -> Path:
    """Determine directory for storing generated run configs."""
    sweep_file = sweep_config.get("sweep_file", "")
    if sweep_file:
        sweep_path = Path(sweep_file)
        return sweep_path.parent / sweep_path.stem
    return Path("sweep_configs") / sweep_id


def main():
    """Main sweep runner entry point."""
    parser = argparse.ArgumentParser(description="Sweep Runner")

    parser.add_argument(
        "--sweep", type=str, required=True, help="Path to sweep YAML file"
    )

    parser.add_argument(
        "--executor",
        type=str,
        choices=["local"],
        help="Execution executor (overrides executor.name in sweep config)",
    )

    # Azure-specific args (override executor config)
    parser.add_argument(
        "--compute",
        type=str,
        help="Azure compute target (overrides executor.compute_target)",
    )

    parser.add_argument(
        "--environment",
        type=str,
        help="Azure environment name (overrides executor.environment_name)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for local executor (default: 1, sequential)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume sweep: only run failed and pending runs (skip completed)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    load_builtin_components()
    load_local_components(args.sweep)

    # Load sweep config
    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"Error: Sweep file not found: {sweep_path}")
        return 1

    sweep_config = load_user_sweep(sweep_path)

    # Add sweep file path to config for sweep naming
    sweep_config["sweep_file"] = str(sweep_path)

    # Determine experiment name (used for tracker and executors)
    experiment_name = generate_experiment_name(sweep_config, timestamp="")

    # Load base config
    base_config_path = sweep_config["base_config"]
    if not Path(base_config_path).exists():
        print(f"Error: Base config not found: {base_config_path}")
        return 1

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Update logging level from config if specified
    config_log_level = base_config.get("runtime", {}).get("log_level")
    if config_log_level:
        setup_logging(config_log_level)
        print(f"   Log level: {config_log_level} (from config)")

    # Generate all run configs and builder
    builder, all_configs = generate_all_run_configs(sweep_config, base_config)
    total_runs = len(all_configs)

    # Handle resume mode: filter to only failed and pending runs
    resume_tracking_context = None
    if args.resume:
        # Create temporary tracker to check for existing sweep
        temp_tracker = SweepTracker(sweep_path, experiment_name, "temp")

        if not temp_tracker.json_path.exists():
            print(
                f"Error: Cannot resume - sweep tracking file not found: {temp_tracker.json_path}"
            )
            return 1

        # Get sweep data to extract tracker context
        sweep_data = temp_tracker.get_sweep_data()
        if not sweep_data:
            print("Error: Cannot resume - sweep tracking data is empty")
            return 1

        # Extract tracking context from existing sweep
        resume_tracking_context = sweep_data.get("tracking_context")
        if resume_tracking_context:
            print(
                f"   Resuming under existing tracking context: "
                f"{resume_tracking_context}"
            )

        # Get failed and pending runs
        failed_runs = temp_tracker.get_failed_runs()
        pending_runs = temp_tracker.get_pending_runs(expected_total_runs=total_runs)
        resume_runs = sorted(failed_runs + pending_runs)

        # Safety check: warn if tracking file has unexpected total_runs
        tracked_total = sweep_data.get("total_runs")
        if tracked_total is not None and tracked_total != total_runs:
            print(
                f"⚠️  WARNING: Sweep config has {total_runs} runs but tracking file shows {tracked_total}"
            )
            print(f"   Using sweep config total ({total_runs}) to detect missing runs")

        if not resume_runs:
            print("No failed or pending runs to resume. All runs are completed!")
            return 0

        # Filter configs to only include resume runs and store original indices
        filtered_configs = []
        for idx, cfg in enumerate(all_configs):
            if idx in resume_runs:
                # Store original index in config for tracker
                cfg["_sweep_run_index"] = idx
                filtered_configs.append(cfg)
        all_configs = filtered_configs

        print(f"\n🔄 Resuming Sweep: {sweep_path.name}")
        print(f"   Failed runs: {len(failed_runs)} {failed_runs}")
        print(f"   Pending runs: {len(pending_runs)} {pending_runs}")
        print(f"   Total to run: {len(resume_runs)}")
    else:
        print(f"\n🚀 Sweep: {sweep_path.name}")
        print(f"   Total runs: {total_runs}")

    # Determine sweep identifier for this execution (used for fallbacks)
    sweep_id = f"sweep_{int(time.time())}"

    # Save configurations to disk once (before executors run)
    config_output_dir = get_config_output_dir(sweep_config, sweep_id)
    saved_config_descriptors = builder.save_configs(all_configs, config_output_dir)
    print(f"   Generated configs in: {config_output_dir}")

    # Extract just the paths from the saved config descriptors
    # Format: List of (run_index, config_path) tuples
    config_paths = [
        (run_index, config_path)
        for run_index, _, config_path in saved_config_descriptors
    ]

    # Determine executor from first generated run config
    if not all_configs:
        print("Error: No run configs generated")
        return 1

    run_executor_config = all_configs[0].get("executor", {})
    if not run_executor_config:
        print("Error: Executor config not found in generated run configurations")
        print("Ensure your sweep config includes executor configuration")
        return 1

    # Apply CLI overrides (CLI takes precedence)
    executor_name = args.executor or run_executor_config.get("name")
    if not executor_name:
        print(
            "Error: Executor name not specified (use --executor or include in sweep config)"
        )
        return 1

    compute_target = args.compute or run_executor_config.get("compute_target")
    environment_name = args.environment or run_executor_config.get(
        "environment_name", "dl_lab"
    )

    # Display executor info
    print(f"   executor: {executor_name}")

    # Update sweep_config with expanded executor config from generated run configs
    # This ensures the executor receives the full expanded config (e.g., datastore_name)
    # rather than unexpanded preset references
    sweep_config["executor"] = run_executor_config

    # Instantiate executor with determined config
    executor = EXECUTOR_REGISTRY.get(
        executor_name,
        sweep_config,
        experiment_name,
        sweep_id,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        compute_target=compute_target,
        environment_name=environment_name,
        tracking_context=resume_tracking_context,
        resume=args.resume,
    )

    # Run sweep using executor - now passing only config paths
    try:
        progress = executor.run_sweep(config_paths, max_workers=args.max_workers)
        print(f"\n✓ Sweep complete: {progress['completed']}/{progress['total']} runs")
        return 0

    except Exception as e:
        print(f"\n✗ Sweep failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
