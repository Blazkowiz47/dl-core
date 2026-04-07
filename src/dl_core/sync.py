"""Sync remote-backed sweep artifacts into the local repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dl_core import load_builtin_components, load_local_components
from dl_core.analysis.sweep_analyzer import get_sweep_tracking_path
from dl_core.core import METRICS_SOURCE_REGISTRY
from dl_core.utils.sweep_tracker import SweepTracker


def _resolve_metrics_source_backend(
    sweep_data: dict[str, Any],
    run_data: dict[str, Any],
) -> str:
    """Resolve the metrics backend for one tracked run."""
    for candidate in (
        run_data.get("metrics_source_backend"),
        run_data.get("tracking_backend"),
        sweep_data.get("metrics_source_backend"),
        sweep_data.get("tracking_backend"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "local"


def _should_sync_run(run_data: dict[str, Any]) -> bool:
    """Return whether a run should participate in artifact sync."""
    return run_data.get("status") not in {"pending", "running"}


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for remote artifact sync."""
    parser = argparse.ArgumentParser(
        prog="dl-sync",
        description="Sync remote-backed sweep artifacts into the local repo.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to the user-facing sweep YAML file.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Sync run artifacts for the tracked sweep runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh local artifacts even when matching paths already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run artifact sync for one tracked sweep."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.artifacts:
        parser.error("Pass --artifacts to sync remote run outputs.")

    sweep_path = Path(args.sweep).resolve()
    tracking_path = get_sweep_tracking_path(sweep_path)
    if not tracking_path.exists():
        parser.error(f"Sweep tracking file not found: {tracking_path}")

    load_builtin_components()
    load_local_components(sweep_path)

    tracker = SweepTracker(sweep_path, sweep_path.stem, "sync")
    sweep_data = tracker.get_sweep_data()
    if not sweep_data:
        parser.error(f"Sweep tracking file is empty: {tracking_path}")
    sweep_data["_tracking_dir"] = str(tracking_path.parent)
    sweep_data["_sweep_path"] = str(sweep_path)

    run_items = [
        (int(run_index), run_data)
        for run_index, run_data in sweep_data.get("runs", {}).items()
        if _should_sync_run(run_data)
    ]
    run_items.sort(key=lambda item: item[0])

    print(f"Syncing artifacts for {len(run_items)} tracked runs")
    synced = 0
    backends: dict[str, Any] = {}
    with tqdm(total=len(run_items), desc="Sync artifacts", unit="run") as pbar:
        for run_index, run_data in run_items:
            backend_name = _resolve_metrics_source_backend(sweep_data, run_data)
            metrics_source = backends.get(backend_name)
            if metrics_source is None:
                metrics_source = METRICS_SOURCE_REGISTRY.get(backend_name)
                backends[backend_name] = metrics_source

            updates = metrics_source.sync_run_artifacts(
                run_index,
                run_data,
                sweep_data,
                force=args.force,
            )
            if updates:
                tracker.update_run_status(
                    run_index,
                    run_data.get("status", "unknown"),
                    config_path=updates.get("config_path"),
                    artifact_dir=updates.get("artifact_dir"),
                    metrics_summary_path=updates.get("metrics_summary_path"),
                    metrics_history_path=updates.get("metrics_history_path"),
                )
                if updates.get("artifact_dir"):
                    synced += 1
            pbar.update(1)

    print(f"Synced artifact paths for {synced}/{len(run_items)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
