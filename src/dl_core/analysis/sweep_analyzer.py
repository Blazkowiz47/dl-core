"""Local-first sweep analysis for ``dl-core``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dl_core import load_builtin_components, load_local_components
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


def get_sweep_analysis_markdown_path(sweep_path: Path) -> Path:
    """Resolve the default Markdown analysis report path for a sweep."""
    return sweep_path.parent / sweep_path.stem / "analysis.md"


def get_sweep_analysis_json_path(sweep_path: Path) -> Path:
    """Resolve the optional JSON analysis report path for a sweep."""
    return sweep_path.parent / sweep_path.stem / "analysis.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def _emit_progress(message: str) -> None:
    """Write one short analyzer progress message to stderr."""
    print(message, file=sys.stderr)


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

    _emit_progress(f"Loading sweep tracker: {tracking_path}")
    load_builtin_components()
    load_local_components(resolved_sweep_path)
    sweep_data = _load_json(tracking_path)
    sweep_data["_tracking_dir"] = str(tracking_path.parent)
    sweep_data["_sweep_path"] = str(resolved_sweep_path)
    runs = sweep_data.get("runs", {})
    collected_runs: list[dict[str, Any]] = []
    _emit_progress(
        f"Collecting {len(runs)} tracked runs from {resolved_sweep_path.name}"
    )

    total_runs = len(runs)
    for run_index_str, run_data in sorted(runs.items(), key=lambda item: int(item[0])):
        run_index = int(run_index_str)
        metrics_source_backend = _resolve_metrics_source_backend(sweep_data, run_data)
        metrics_source = METRICS_SOURCE_REGISTRY.get(metrics_source_backend)
        run_name = run_data.get("tracking_run_name") or f"run_{run_index}"
        _emit_progress(
            f"[{run_index + 1}/{total_runs}] Collecting {run_name} "
            f"via {metrics_source_backend}"
        )
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


def _summarize_status_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Count runs by analyzer status."""
    counts: dict[str, int] = {}
    for run in runs:
        status = str(run.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _resolve_primary_metric(runs: list[dict[str, Any]]) -> tuple[str, str]:
    """Resolve the main selection metric and mode for one sweep report."""
    for run in runs:
        metric = run.get("selection_metric")
        mode = run.get("selection_mode")
        if isinstance(metric, str) and metric:
            return metric, mode if isinstance(mode, str) and mode else "-"
    return "-", "-"


def _select_common_best_metric_columns(
    runs: list[dict[str, Any]],
    *,
    primary_metric: str,
    max_columns: int = 4,
) -> list[str]:
    """Select a small set of common best-epoch metrics for report tables."""
    metric_sets: list[set[str]] = []
    for run in runs:
        best_metrics = run.get("best_metrics")
        if not isinstance(best_metrics, dict) or not best_metrics:
            continue
        run_metric_names = {
            metric_name
            for metric_name, value in best_metrics.items()
            if isinstance(metric_name, str)
            and metric_name
            and isinstance(value, (int, float))
        }
        if run_metric_names:
            metric_sets.append(run_metric_names)

    if not metric_sets:
        return []

    common_metrics = set.intersection(*metric_sets)
    common_metrics.discard(primary_metric)
    ordered_metrics = sorted(common_metrics)
    return ordered_metrics[:max_columns]


def _get_top_runs(
    runs: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Return the top-ranked runs for compact report summaries."""
    return _sort_runs_for_display(runs)[:top_n]


def _collect_metric_statistics(
    runs: list[dict[str, Any]],
    *,
    primary_metric: str,
    metric_columns: list[str],
) -> list[dict[str, str]]:
    """Collect basic metric statistics from selection and best-metric values."""
    metric_values: dict[str, list[float]] = {}
    if primary_metric != "-":
        metric_values[primary_metric] = [
            float(run["selection_value"])
            for run in runs
            if isinstance(run.get("selection_value"), (int, float))
        ]

    for metric_name in metric_columns:
        values: list[float] = []
        for run in runs:
            best_metrics = run.get("best_metrics")
            if not isinstance(best_metrics, dict):
                continue
            metric_value = best_metrics.get(metric_name)
            if isinstance(metric_value, (int, float)):
                values.append(float(metric_value))
        metric_values[metric_name] = values

    statistics: list[dict[str, str]] = []
    for metric_name, values in metric_values.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        statistics.append(
            {
                "metric": metric_name,
                "mean": f"{mean:.6f}",
                "std": f"{variance ** 0.5:.6f}",
                "min": f"{min(values):.6f}",
                "max": f"{max(values):.6f}",
            }
        )
    return statistics


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render one compact monospace table as plain text lines."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _format_row(row: list[str]) -> str:
        return " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return [_format_row(headers), separator, *(_format_row(row) for row in rows)]


def _render_text_report(sweep_path: Path, runs: list[dict[str, Any]]) -> str:
    """Render a human-readable text report for the collected sweep runs."""
    sorted_runs = _sort_runs_for_display(runs)
    primary_metric, primary_mode = _resolve_primary_metric(sorted_runs)
    status_counts = _summarize_status_counts(sorted_runs)
    top_runs = _get_top_runs(sorted_runs)
    lines = [
        f"Sweep: {sweep_path.name}",
        f"Tracked runs: {len(sorted_runs)}",
        f"Primary metric: {primary_metric}",
        f"Mode: {primary_mode}",
        (
            "Status summary: "
            + ", ".join(
                f"{status}={count}"
                for status, count in sorted(status_counts.items())
            )
        ),
        "",
    ]

    ranking_rows = [
        [
            str(rank),
            run["run_name"],
            str(run.get("status", "-")),
            _format_metric_value(run.get("selection_value")),
            str(run.get("best_epoch", "-")),
        ]
        for rank, run in enumerate(sorted_runs, start=1)
    ]
    lines.extend(
        [
            "Ranking",
            *(
                _render_table(
                    ["Rank", "Run", "Status", "Value", "Best Epoch"],
                    ranking_rows,
                )
            ),
        ]
    )

    metric_columns = _select_common_best_metric_columns(
        sorted_runs,
        primary_metric=primary_metric,
    )
    if top_runs:
        lines.extend(["", "Top Performers"])
        for rank, run in enumerate(top_runs, start=1):
            lines.append(
                f"{rank}. {run['run_name']} | "
                f"value={_format_metric_value(run.get('selection_value'))} | "
                f"best_epoch={run.get('best_epoch', '-')}"
            )

    if metric_columns:
        metric_rows = []
        for run in sorted_runs:
            best_metrics = run.get("best_metrics")
            if not isinstance(best_metrics, dict):
                continue
            metric_rows.append(
                [
                    run["run_name"],
                    *[
                        _format_metric_value(best_metrics.get(metric_name))
                        for metric_name in metric_columns
                    ],
                ]
            )
        if metric_rows:
            lines.extend(
                [
                    "",
                    "Common Best-Epoch Metrics",
                    *(_render_table(["Run", *metric_columns], metric_rows)),
                ]
            )

    metric_statistics = _collect_metric_statistics(
        sorted_runs,
        primary_metric=primary_metric,
        metric_columns=metric_columns,
    )
    if metric_statistics:
        stats_rows = [
            [
                row["metric"],
                row["mean"],
                row["std"],
                row["min"],
                row["max"],
            ]
            for row in metric_statistics
        ]
        lines.extend(
            [
                "",
                "Metric Statistics",
                *(
                    _render_table(
                        ["Metric", "Mean", "Std", "Min", "Max"],
                        stats_rows,
                    )
                ),
            ]
        )

    failures = [run for run in sorted_runs if run.get("error_message")]
    if failures:
        lines.extend(["", "Failures"])
        for run in failures:
            lines.append(f"- {run['run_name']}: {run['error_message']}")

    warnings = [run for run in sorted_runs if run.get("metrics_source_warning")]
    if warnings:
        lines.extend(["", "Warnings"])
        for run in warnings:
            lines.append(
                f"- {run['run_name']}: {run['metrics_source_warning']}"
            )

    return "\n".join(lines)


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render one Markdown table."""
    separator = ["---"] * len(headers)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_rows.extend("| " + " | ".join(row) + " |" for row in rows)
    return markdown_rows


def _render_markdown_report(sweep_path: Path, runs: list[dict[str, Any]]) -> str:
    """Render a Markdown report for one collected sweep."""
    sorted_runs = _sort_runs_for_display(runs)
    primary_metric, primary_mode = _resolve_primary_metric(sorted_runs)
    status_counts = _summarize_status_counts(sorted_runs)
    top_runs = _get_top_runs(sorted_runs)
    lines = [
        "# Sweep Analysis",
        "",
        f"- Sweep: `{sweep_path.name}`",
        f"- Tracked runs: {len(sorted_runs)}",
        f"- Primary metric: `{primary_metric}`",
        f"- Mode: `{primary_mode}`",
        f"- Status summary: "
        + ", ".join(
            f"`{status}={count}`" for status, count in sorted(status_counts.items())
        ),
        "",
        "## Ranking",
        "",
    ]

    ranking_rows = [
        [
            str(rank),
            run["run_name"],
            str(run.get("status", "-")),
            _format_metric_value(run.get("selection_value")),
            str(run.get("best_epoch", "-")),
        ]
        for rank, run in enumerate(sorted_runs, start=1)
    ]
    lines.extend(
        _render_markdown_table(
            ["Rank", "Run", "Status", "Value", "Best Epoch"],
            ranking_rows,
        )
    )
    lines.append("")

    metric_columns = _select_common_best_metric_columns(
        sorted_runs,
        primary_metric=primary_metric,
    )
    if top_runs:
        lines.extend(["## Top Performers", ""])
        for rank, run in enumerate(top_runs, start=1):
            lines.extend(
                [
                    f"### {rank}. {run['run_name']}",
                    "",
                    f"- Value: `{_format_metric_value(run.get('selection_value'))}`",
                    f"- Best epoch: `{run.get('best_epoch', '-')}`",
                ]
            )
            best_metrics = run.get("best_metrics")
            if isinstance(best_metrics, dict):
                for metric_name in metric_columns[:3]:
                    if metric_name in best_metrics:
                        lines.append(
                            f"- {metric_name}: "
                            f"`{_format_metric_value(best_metrics.get(metric_name))}`"
                        )
            lines.append("")

    if metric_columns:
        metric_rows = []
        for run in sorted_runs:
            best_metrics = run.get("best_metrics")
            if not isinstance(best_metrics, dict):
                continue
            metric_rows.append(
                [
                    run["run_name"],
                    *[
                        _format_metric_value(best_metrics.get(metric_name))
                        for metric_name in metric_columns
                    ],
                ]
            )
        if metric_rows:
            lines.extend(["## Common Best-Epoch Metrics", ""])
            lines.extend(
                _render_markdown_table(["Run", *metric_columns], metric_rows)
            )
            lines.append("")

    metric_statistics = _collect_metric_statistics(
        sorted_runs,
        primary_metric=primary_metric,
        metric_columns=metric_columns,
    )
    if metric_statistics:
        lines.extend(["## Metric Statistics", ""])
        lines.extend(
            _render_markdown_table(
                ["Metric", "Mean", "Std", "Min", "Max"],
                [
                    [
                        row["metric"],
                        row["mean"],
                        row["std"],
                        row["min"],
                        row["max"],
                    ]
                    for row in metric_statistics
                ],
            )
        )
        lines.append("")

    failures = [run for run in sorted_runs if run.get("error_message")]
    if failures:
        lines.extend(["## Failures", ""])
        for run in failures:
            lines.append(f"- `{run['run_name']}`: {run['error_message']}")
        lines.append("")

    warnings = [run for run in sorted_runs if run.get("metrics_source_warning")]
    if warnings:
        lines.extend(["## Warnings", ""])
        for run in warnings:
            lines.append(
                f"- `{run['run_name']}`: {run['metrics_source_warning']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_analysis_reports(
    sweep_path: Path,
    runs: list[dict[str, Any]],
    *,
    write_json_report: bool,
) -> tuple[Path, Path | None]:
    """Write Markdown and optional JSON analysis reports next to the sweep."""
    markdown_path = get_sweep_analysis_markdown_path(sweep_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _render_markdown_report(sweep_path, runs),
        encoding="utf-8",
    )

    json_path: Path | None = None
    if write_json_report:
        json_path = get_sweep_analysis_json_path(sweep_path)
        json_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    return markdown_path, json_path


def main(argv: list[str] | None = None) -> int:
    """Run the local sweep analyzer CLI."""
    parser = argparse.ArgumentParser(
        prog="dl-analyze",
        description=(
            "Inspect sweep results and write experiments/<sweep>/analysis.md "
            "by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dl-analyze --sweep experiments/lr_sweep.yaml\n"
            "  dl-analyze --sweep experiments/lr_sweep.yaml --json\n\n"
            "This command reads the generated experiments/<sweep_name>/"
            "sweep_tracking.json and writes experiments/<sweep_name>/"
            "analysis.md. Use --json to also write analysis.json."
        ),
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to the user-facing sweep YAML file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write analysis.json and print the normalized run records.",
    )
    args = parser.parse_args(argv)

    sweep_path = Path(args.sweep).resolve()
    runs = collect_sweep_runs(sweep_path)
    markdown_path, json_path = _write_analysis_reports(
        sweep_path,
        runs,
        write_json_report=args.json,
    )
    _emit_progress(f"Wrote Markdown report to {markdown_path}")
    if json_path is not None:
        _emit_progress(f"Wrote JSON report to {json_path}")

    if args.json:
        print(json.dumps(runs, indent=2))
    else:
        print(_render_text_report(sweep_path, runs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
