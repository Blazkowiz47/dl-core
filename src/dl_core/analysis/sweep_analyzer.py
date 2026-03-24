"""Local-first sweep analysis for ``dl-core``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from dl_core.utils.artifact_manager import get_run_artifact_dir


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


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _infer_artifact_dir(run_data: dict[str, Any]) -> Path | None:
    """
    Resolve the local artifact directory for a tracked run.

    Args:
        run_data: Sweep tracker entry for one run

    Returns:
        Path to the artifact directory, if it can be resolved.
    """
    artifact_dir = run_data.get("artifact_dir")
    if isinstance(artifact_dir, str) and artifact_dir:
        return Path(artifact_dir)

    config_path_value = run_data.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        return None

    config_path = Path(config_path_value)
    if not config_path.exists():
        return None

    config = _load_yaml(config_path)
    runtime_config = config.get("runtime", {})
    experiment_config = config.get("experiment", {})
    run_name = runtime_config.get("name", config_path.stem)
    output_dir = runtime_config.get("output_dir", "artifacts")

    sweep_name = None
    sweep_file = config.get("sweep_file")
    if isinstance(sweep_file, str) and sweep_file:
        sweep_name = Path(sweep_file).stem

    return Path(
        get_run_artifact_dir(
            run_name=run_name,
            output_dir=output_dir,
            experiment_name=experiment_config.get("name"),
            sweep_name=sweep_name,
        )
    )


def _resolve_metrics_path(
    run_data: dict[str, Any],
    artifact_dir: Path | None,
    filename: str,
    tracker_key: str,
) -> Path | None:
    """
    Resolve a metrics artifact path from tracker data or artifact structure.

    Args:
        run_data: Sweep tracker entry for one run
        artifact_dir: Resolved artifact directory, if available
        filename: File expected under ``metrics/``
        tracker_key: Explicit tracker key for the artifact

    Returns:
        Resolved path, if available.
    """
    tracker_value = run_data.get(tracker_key)
    if isinstance(tracker_value, str) and tracker_value:
        return Path(tracker_value)

    if artifact_dir is None:
        return None

    return artifact_dir / "metrics" / filename


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

    sweep_data = _load_json(tracking_path)
    runs = sweep_data.get("runs", {})
    collected_runs: list[dict[str, Any]] = []

    for run_index_str, run_data in sorted(runs.items(), key=lambda item: int(item[0])):
        artifact_dir = _infer_artifact_dir(run_data)
        summary_path = _resolve_metrics_path(
            run_data,
            artifact_dir,
            filename="summary.json",
            tracker_key="metrics_summary_path",
        )
        history_path = _resolve_metrics_path(
            run_data,
            artifact_dir,
            filename="history.json",
            tracker_key="metrics_history_path",
        )

        summary: dict[str, Any] = {}
        if summary_path is not None and summary_path.exists():
            summary = _load_json(summary_path)

        run_name = (
            run_data.get("tracking_run_name")
            or summary.get("run_name")
            or Path(run_data.get("config_path", f"run_{run_index_str}.yaml")).stem
        )

        collected_runs.append(
            {
                "run_index": int(run_index_str),
                "run_name": run_name,
                "status": run_data.get("status", "unknown"),
                "error_message": (
                    run_data.get("error_message") or summary.get("error_message")
                ),
                "artifact_dir": str(artifact_dir) if artifact_dir else None,
                "config_path": run_data.get("config_path"),
                "metrics_summary_path": (
                    str(summary_path) if summary_path is not None else None
                ),
                "metrics_history_path": (
                    str(history_path) if history_path is not None else None
                ),
                "summary_available": bool(summary),
                "selection_metric": summary.get("selection_metric"),
                "selection_mode": summary.get("selection_mode"),
                "selection_value": summary.get("selection_value"),
                "best_epoch": summary.get("best_epoch"),
                "final_epoch": summary.get("final_epoch"),
                "best_metrics": summary.get("best_metrics", {}),
                "final_metrics": summary.get("final_metrics", {}),
            }
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
