"""Local-first sweep analysis for ``dl-core``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dl_core import load_builtin_components
from dl_core.core import METRICS_SOURCE_REGISTRY


def get_sweep_tracking_path(sweep_path: Path) -> Path:
    """
    Resolve the tracker JSON for a generated sweep.

    Args:
        sweep_path: Path to the user-facing sweep YAML file

    Returns:
        Path to the generated ``sweep_tracking.json`` file.
    """
    return sweep_path.parent / sweep_path.stem / "sweep_tracking.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def _resolve_metrics_source_backend(
    sweep_data: dict[str, Any], run_data: dict[str, Any]
) -> str:
    """
    Resolve the metrics source backend for one tracked run.

    Args:
        sweep_data: Full sweep tracking payload
        run_data: Sweep tracker entry for one run

    Returns:
        Metrics source backend name.
    """
    backend_candidates = (
        run_data.get("metrics_source_backend"),
        run_data.get("tracking_backend"),
        sweep_data.get("metrics_source_backend"),
        sweep_data.get("tracking_backend"),
    )
    for candidate in backend_candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return "local"


def collect_sweep_runs(sweep_path: str | Path) -> list[dict[str, Any]]:
    """
    Collect normalized run analysis records for a sweep.

    Args:
        sweep_path: Path to the user-facing sweep YAML file

    Returns:
        List of normalized run records.
    """
    resolved_sweep_path = Path(sweep_path).resolve()
    tracking_path = get_sweep_tracking_path(resolved_sweep_path)
    if not tracking_path.exists():
        raise FileNotFoundError(f"Sweep tracking file not found: {tracking_path}")

    load_builtin_components()
    sweep_data = _load_json(tracking_path)
    runs = sweep_data.get("runs", {})
    collected_runs: list[dict[str, Any]] = []

    for run_index_str, run_data in sorted(runs.items(), key=lambda item: int(item[0])):
        run_index = int(run_index_str)
        metrics_source_backend = _resolve_metrics_source_backend(sweep_data, run_data)
        metrics_source = METRICS_SOURCE_REGISTRY.get(metrics_source_backend)
        collected_runs.append(
            metrics_source.collect_run(
                run_index=run_index,
                run_data=run_data,
                sweep_data=sweep_data,
            )
        )

    return collected_runs


def _format_metric_value(value: Any) -> str:
    """Format a metric value for terminal output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def _sort_runs_for_display(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort runs by selection value when available, otherwise by run index."""
    runs_with_metric = [
        run for run in runs if isinstance(run.get("selection_value"), (int, float))
    ]
    if not runs_with_metric:
        return sorted(runs, key=lambda run: run["run_index"])

    first_mode = runs_with_metric[0].get("selection_mode")
    reverse = first_mode == "max"
    metric_sorted = sorted(
        runs_with_metric,
        key=lambda run: float(run["selection_value"]),
        reverse=reverse,
    )
    runs_without_metric = [
        run for run in runs if not isinstance(run.get("selection_value"), (int, float))
    ]
    runs_without_metric.sort(key=lambda run: run["run_index"])
    return metric_sorted + runs_without_metric


def _render_text_report(sweep_path: Path, runs: list[dict[str, Any]]) -> str:
    """Render a human-readable text report for the collected sweep runs."""
    lines = [
        f"Sweep: {sweep_path.name}",
        f"Tracked runs: {len(runs)}",
        "",
    ]

    for run in _sort_runs_for_display(runs):
        metric_name = run.get("selection_metric") or "-"
        metric_value = _format_metric_value(run.get("selection_value"))
        lines.append(
            " | ".join(
                [
                    f"[{run['run_index']}]",
                    f"status={run['status']}",
                    f"name={run['run_name']}",
                    f"metric={metric_name}",
                    f"value={metric_value}",
                    f"best_epoch={run.get('best_epoch', '-')}",
                ]
            )
        )
        if run.get("artifact_dir"):
            lines.append(f"  artifact_dir: {run['artifact_dir']}")
        if run.get("metrics_summary_path"):
            lines.append(f"  summary: {run['metrics_summary_path']}")
        if run.get("error_message"):
            lines.append(f"  error: {run['error_message']}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the local sweep analyzer CLI."""
    parser = argparse.ArgumentParser(
        prog="dl-analyze-sweep",
        description="Inspect local sweep results from saved artifact summaries.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to the user-facing sweep YAML file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the normalized run records as JSON.",
    )
    args = parser.parse_args(argv)

    sweep_path = Path(args.sweep).resolve()
    runs = collect_sweep_runs(sweep_path)

    if args.json:
        print(json.dumps(runs, indent=2))
    else:
        print(_render_text_report(sweep_path, runs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
