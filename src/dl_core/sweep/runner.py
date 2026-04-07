"""Main sweep runner with pluggable executors."""

import argparse
import csv
import fnmatch
import json
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
from .config.config_utils import deep_get
from .template import (
    ensure_tracking_experiment_name,
    generate_experiment_name,
    load_user_sweep,
)


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


def _get_preview_columns(builder: ConfigBuilder) -> list[str]:
    """Return the sweep grid keys that should appear in preview output."""
    resolved_grid = builder._resolve_preset_references(builder.grid)
    return list(resolved_grid.keys())


def _build_preview_rows(
    builder: ConfigBuilder,
    prepared_configs: List[Tuple[int, Dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Build preview rows from expanded run configs."""
    preview_columns = _get_preview_columns(builder)
    rows: list[dict[str, Any]] = []

    for run_index, run_config, run_name in prepared_configs:
        row: dict[str, Any] = {
            "index": run_index,
            "run_name": run_name,
            "seed": run_config.get("seed"),
        }
        for column in preview_columns:
            try:
                row[column] = deep_get(run_config, column)
            except KeyError:
                row[column] = None
        rows.append(row)

    return rows


def _stringify_preview_value(value: Any) -> str:
    """Return a compact string form for one preview value."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _render_preview_table(rows: list[dict[str, Any]]) -> str:
    """Render preview rows as a simple text table."""
    if not rows:
        return "No sweep runs were generated."

    columns = list(rows[0].keys())
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(
                widths[column],
                len(_stringify_preview_value(row[column])),
            )

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(
            _stringify_preview_value(row[column]).ljust(widths[column])
            for column in columns
        )
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _export_preview_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write preview rows to CSV or JSON based on file extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        return

    if output_path.suffix != ".csv":
        raise ValueError("Preview export path must end with .csv or .json")

    fieldnames = list(rows[0].keys()) if rows else ["index", "run_name", "seed"]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _stringify_preview_value(value)
                    for key, value in row.items()
                }
            )


def _filter_prepared_configs(
    prepared_configs: List[Tuple[int, Dict[str, Any], str]],
    only_patterns: list[str],
    skip_patterns: list[str],
) -> List[Tuple[int, Dict[str, Any], str]]:
    """Filter prepared configs by run-name glob patterns."""
    filtered_configs = []
    for prepared_config in prepared_configs:
        _, _, run_name = prepared_config
        if only_patterns and not any(
            fnmatch.fnmatch(run_name, pattern) for pattern in only_patterns
        ):
            continue
        if skip_patterns and any(
            fnmatch.fnmatch(run_name, pattern) for pattern in skip_patterns
        ):
            continue
        filtered_configs.append(prepared_config)
    return filtered_configs


def main():
    """Main sweep runner entry point."""
    parser = argparse.ArgumentParser(
        description="Generate run configs from a sweep file and execute them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dl-sweep experiments/lr_sweep.yaml\n"
            "  dl-sweep experiments/lr_sweep.yaml --preview\n"
            "  dl-sweep experiments/lr_sweep.yaml --export sweep_preview.csv\n"
            "  dl-sweep experiments/lr_sweep.yaml --only '*seed_2025*'\n"
            "  dl-sweep experiments/lr_sweep.yaml --dry-run\n"
            "  dl-sweep experiments/lr_sweep.yaml --resume\n"
            "  dl-sweep --sweep experiments/lr_sweep.yaml  # compatibility alias\n\n"
            "The sweep file normally lives under experiments/ and points at\n"
            "configs/base.yaml via base_config."
        ),
    )

    parser.add_argument(
        "sweep_path",
        nargs="?",
        help="Path to sweep YAML file",
    )
    parser.add_argument(
        "--sweep",
        dest="sweep_flag",
        type=str,
        help="Path to sweep YAML file (backward-compatible alias)",
    )

    parser.add_argument(
        "--executor",
        type=str,
        choices=["local"],
        help="Execution executor (overrides executor.name in the sweep config).",
    )

    # Azure-specific args (override executor config)
    parser.add_argument(
        "--compute",
        type=str,
        help="Executor-specific compute target override when supported.",
    )

    parser.add_argument(
        "--environment",
        type=str,
        help="Executor-specific environment override when supported.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the expanded sweep matrix and exit without executing",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Write the expanded sweep matrix to a .csv or .json file and exit",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Keep only run names matching this glob pattern. Repeatable.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip run names matching this glob pattern. Repeatable.",
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
    if args.sweep_path and args.sweep_flag and args.sweep_path != args.sweep_flag:
        parser.error("Pass the sweep file either positionally or with --sweep.")
    sweep_arg = args.sweep_flag or args.sweep_path
    if not sweep_arg:
        parser.error("Provide a sweep file as a positional argument or with --sweep.")

    setup_logging(args.log_level)
    load_builtin_components()
    load_local_components(sweep_arg)

    # Load sweep config
    sweep_path = Path(sweep_arg)
    if not sweep_path.exists():
        print(f"Error: Sweep file not found: {sweep_path}")
        return 1

    sweep_config = load_user_sweep(sweep_path)

    # Add sweep file path to config for sweep naming
    sweep_config["sweep_file"] = str(sweep_path)
    ensure_tracking_experiment_name(sweep_config, config_path=sweep_path)

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
    prepared_configs = builder.prepare_configs(all_configs)
    prepared_configs = _filter_prepared_configs(
        prepared_configs,
        args.only,
        args.skip,
    )
    if not prepared_configs:
        print("No sweep runs matched the requested filters.")
        return 0

    if args.only or args.skip:
        print(f"   Filtered runs: {len(prepared_configs)}/{len(all_configs)}")
    all_configs = [run_config for _, run_config, _ in prepared_configs]

    if args.preview or args.export:
        preview_rows = _build_preview_rows(builder, prepared_configs)
        if args.preview:
            print("\n📋 Sweep Preview")
            print(_render_preview_table(preview_rows))
        if args.export:
            export_path = Path(args.export)
            _export_preview_rows(export_path, preview_rows)
            print(f"   Exported preview: {export_path}")
        return 0

    # Save configurations to disk once (before executors run)
    config_output_dir = get_config_output_dir(sweep_config, sweep_id)
    saved_config_descriptors = builder.save_configs(
        all_configs,
        config_output_dir,
    )
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
