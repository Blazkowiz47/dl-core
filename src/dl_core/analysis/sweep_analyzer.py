"""Local-first sweep analysis for ``dl-core``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dl_core import load_builtin_components, load_local_components
from dl_core.core import METRICS_SOURCE_REGISTRY
from tqdm import tqdm


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


def collect_sweep_runs(
    sweep_path: str | Path,
    *,
    ranking_metrics: list[dict[str, str]] | None = None,
    rank_method: str = "lexicographic",
) -> list[dict[str, Any]]:
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
    sweep_data["_ranking_metrics"] = ranking_metrics or [
        {"metric": "test/accuracy", "mode": "max"}
    ]
    sweep_data["_rank_method"] = rank_method
    runs = sweep_data.get("runs", {})
    collected_runs: list[dict[str, Any]] = []
    run_items = sorted(
        ((int(run_index_str), run_data) for run_index_str, run_data in runs.items()),
        key=lambda item: item[0],
    )
    _emit_progress(
        f"Collecting {len(runs)} tracked runs from {resolved_sweep_path.name}"
    )

    prepared_backends: dict[str, Any] = {}
    for run_index, run_data in run_items:
        metrics_source_backend = _resolve_metrics_source_backend(sweep_data, run_data)
        if metrics_source_backend in prepared_backends:
            continue

        metrics_source = METRICS_SOURCE_REGISTRY.get(metrics_source_backend)
        backend_run_items = [
            (item_run_index, item_run_data)
            for item_run_index, item_run_data in run_items
            if _resolve_metrics_source_backend(sweep_data, item_run_data)
            == metrics_source_backend
        ]
        if backend_run_items:
            _emit_progress(
                f"Preparing {len(backend_run_items)} runs via "
                f"{metrics_source_backend}"
            )
            with tqdm(
                total=len(backend_run_items),
                desc=f"Prefetch {metrics_source_backend}",
                unit="run",
                file=sys.stderr,
                leave=False,
            ) as pbar:
                metrics_source.prepare_sweep(
                    run_items=backend_run_items,
                    sweep_data=sweep_data,
                    progress_callback=lambda: pbar.update(1),
                )
        prepared_backends[metrics_source_backend] = metrics_source

    total_runs = len(run_items)
    with tqdm(
        total=total_runs,
        desc="Collect runs",
        unit="run",
        file=sys.stderr,
        leave=False,
    ) as pbar:
        for run_index, run_data in run_items:
            metrics_source_backend = _resolve_metrics_source_backend(
                sweep_data,
                run_data,
            )
            metrics_source = prepared_backends[metrics_source_backend]
            run_name = run_data.get("tracking_run_name") or f"run_{run_index}"
            pbar.set_postfix_str(run_name)
            collected_runs.append(
                metrics_source.collect_run(
                    run_index=run_index,
                    run_data=run_data,
                    sweep_data=sweep_data,
                )
            )
            pbar.update(1)

    return collected_runs


def _format_metric_value(value: Any) -> str:
    """Format a metric value for terminal output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def _get_ranking_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized ranking entries for one run."""
    ranking_entries = run.get("ranking_metrics")
    if isinstance(ranking_entries, list) and ranking_entries:
        return [entry for entry in ranking_entries if isinstance(entry, dict)]

    selection_metric = run.get("selection_metric")
    selection_mode = run.get("selection_mode")
    if (
        isinstance(selection_metric, str)
        and selection_metric
        and selection_mode in {"min", "max"}
    ):
        return [
            {
                "metric": selection_metric,
                "resolved_metric": selection_metric,
                "mode": selection_mode,
                "value": run.get("selection_value"),
                "best_epoch": run.get("best_epoch"),
                "final_value": None,
            }
        ]
    return []


def _get_ranking_specs(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Resolve the shared ranking metric specs from the collected runs."""
    for run in runs:
        ranking_entries = _get_ranking_entries(run)
        if ranking_entries:
            return [
                {"metric": str(entry["metric"]), "mode": str(entry["mode"])}
                for entry in ranking_entries
                if isinstance(entry.get("metric"), str)
                and entry.get("mode") in {"min", "max"}
            ]
    return []


def _get_rank_method(runs: list[dict[str, Any]]) -> str:
    """Resolve the configured ranking method for the current sweep."""
    for run in runs:
        rank_method = run.get("rank_method")
        if rank_method in {"lexicographic", "pareto"}:
            return str(rank_method)
    return "lexicographic"


def _get_ranking_value(run: dict[str, Any], metric_name: str) -> Any:
    """Return the ranking value for one requested metric on one run."""
    for entry in _get_ranking_entries(run):
        if entry.get("metric") == metric_name:
            return entry.get("value")
    return None


def _get_ranking_best_epoch(run: dict[str, Any], metric_name: str) -> Any:
    """Return the best epoch for one requested metric on one run."""
    for entry in _get_ranking_entries(run):
        if entry.get("metric") == metric_name:
            return entry.get("best_epoch")
    return None


def _infer_metric_mode(metric_name: str) -> str | None:
    """Infer whether one metric should be minimized or maximized."""
    metric_lower = metric_name.lower()
    minimize_keywords = [
        "eer",
        "error",
        "loss",
        "apcer",
        "bpcer",
        "far",
        "frr",
        "fnr",
        "fpr",
        "miss",
        "false",
        "distance",
        "cost",
    ]
    maximize_keywords = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "roc",
        "score",
        "iou",
        "dice",
        "jaccard",
        "map",
        "ndcg",
    ]

    for keyword in minimize_keywords:
        if keyword in metric_lower:
            return "min"
    for keyword in maximize_keywords:
        if keyword in metric_lower:
            return "max"
    return None


def _build_requested_ranking_specs(
    metric_names: list[str] | None,
    metric_modes: list[str] | None,
) -> list[dict[str, str]]:
    """Build requested ranking specs from CLI arguments."""
    resolved_metric_names = metric_names or ["test/accuracy"]
    resolved_metric_modes = metric_modes or []
    if resolved_metric_modes and len(resolved_metric_modes) != len(
        resolved_metric_names
    ):
        raise ValueError("Provide one --mode per --metric, or omit --mode entirely.")

    ranking_specs: list[dict[str, str]] = []
    for index, metric_name in enumerate(resolved_metric_names):
        metric_mode = (
            resolved_metric_modes[index]
            if index < len(resolved_metric_modes)
            else (_infer_metric_mode(metric_name) or "max")
        )
        ranking_specs.append({"metric": metric_name, "mode": metric_mode})
    return ranking_specs


def _resolve_primary_metric_details(
    runs: list[dict[str, Any]],
) -> tuple[str, str, str | None]:
    """Resolve the primary metric, effective mode, and any override note."""
    ranking_specs = _get_ranking_specs(runs)
    if ranking_specs:
        primary_spec = ranking_specs[0]
        return primary_spec["metric"], primary_spec["mode"], None

    for run in runs:
        metric = run.get("selection_metric")
        explicit_mode = run.get("selection_mode")
        if not isinstance(metric, str) or not metric:
            continue

        normalized_mode = (
            explicit_mode
            if isinstance(explicit_mode, str) and explicit_mode in {"min", "max"}
            else None
        )
        inferred_mode = _infer_metric_mode(metric)
        if inferred_mode is not None and normalized_mode not in {None, inferred_mode}:
            return (
                metric,
                inferred_mode,
                f"inferred `{inferred_mode}` from metric name; tracker stored `{normalized_mode}`",
            )
        return metric, normalized_mode or inferred_mode or "-", None

    return "-", "-", None


def _sort_runs_for_display(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort runs by selection value when available, otherwise by run index."""
    ranking_specs = _get_ranking_specs(runs)
    if not ranking_specs:
        runs_with_metric = [
            run for run in runs if isinstance(run.get("selection_value"), (int, float))
        ]
        if not runs_with_metric:
            return sorted(runs, key=lambda run: run["run_index"])

        _, effective_mode, _ = _resolve_primary_metric_details(runs_with_metric)
        reverse = effective_mode == "max"
        metric_sorted = sorted(
            runs_with_metric,
            key=lambda run: float(run["selection_value"]),
            reverse=reverse,
        )
        runs_without_metric = [
            run
            for run in runs
            if not isinstance(run.get("selection_value"), (int, float))
        ]
        runs_without_metric.sort(key=lambda run: run["run_index"])
        ordered_runs = metric_sorted + runs_without_metric
    elif _get_rank_method(runs) == "pareto":
        ordered_runs = _sort_runs_by_pareto_front(runs, ranking_specs)
    else:
        ordered_runs = _sort_runs_lexicographically(runs, ranking_specs)

    for rank, run in enumerate(ordered_runs, start=1):
        run["_rank"] = rank
    return ordered_runs


def _sort_runs_lexicographically(
    runs: list[dict[str, Any]],
    ranking_specs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Sort runs lexicographically using the configured ranking metrics."""
    def _build_key(run: dict[str, Any]) -> tuple[Any, ...]:
        key_parts: list[Any] = []
        for ranking_spec in ranking_specs:
            metric_value = _get_ranking_value(run, ranking_spec["metric"])
            if not isinstance(metric_value, (int, float)):
                key_parts.extend([1, float("inf")])
                continue
            sortable_value = (
                float(metric_value)
                if ranking_spec["mode"] == "min"
                else -float(metric_value)
            )
            key_parts.extend([0, sortable_value])
        key_parts.append(run["run_index"])
        return tuple(key_parts)

    return sorted(runs, key=_build_key)


def _dominates(
    lhs_run: dict[str, Any],
    rhs_run: dict[str, Any],
    ranking_specs: list[dict[str, str]],
) -> bool:
    """Return whether one run Pareto-dominates another."""
    lhs_better = False
    for ranking_spec in ranking_specs:
        lhs_value = _get_ranking_value(lhs_run, ranking_spec["metric"])
        rhs_value = _get_ranking_value(rhs_run, ranking_spec["metric"])
        if not isinstance(lhs_value, (int, float)) or not isinstance(
            rhs_value,
            (int, float),
        ):
            return False
        if ranking_spec["mode"] == "min":
            if lhs_value > rhs_value:
                return False
            if lhs_value < rhs_value:
                lhs_better = True
        else:
            if lhs_value < rhs_value:
                return False
            if lhs_value > rhs_value:
                lhs_better = True
    return lhs_better


def _sort_runs_by_pareto_front(
    runs: list[dict[str, Any]],
    ranking_specs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Sort runs by Pareto front and lexicographic tie-breaks."""
    complete_runs = [
        run
        for run in runs
        if all(
            isinstance(_get_ranking_value(run, spec["metric"]), (int, float))
            for spec in ranking_specs
        )
    ]
    incomplete_runs = [
        run
        for run in runs
        if run not in complete_runs
    ]
    remaining_runs = complete_runs[:]
    ordered_runs: list[dict[str, Any]] = []
    front_index = 1
    while remaining_runs:
        current_front = [
            run
            for run in remaining_runs
            if not any(
                _dominates(other_run, run, ranking_specs)
                for other_run in remaining_runs
                if other_run is not run
            )
        ]
        current_front = _sort_runs_lexicographically(current_front, ranking_specs)
        for run in current_front:
            run["_front"] = front_index
        ordered_runs.extend(current_front)
        remaining_runs = [
            run for run in remaining_runs if run not in current_front
        ]
        front_index += 1

    for run in incomplete_runs:
        run["_front"] = None
    ordered_runs.extend(_sort_runs_lexicographically(incomplete_runs, ranking_specs))
    return ordered_runs


def _summarize_status_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Count runs by analyzer status."""
    counts: dict[str, int] = {}
    for run in runs:
        status = str(run.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _resolve_primary_metric(runs: list[dict[str, Any]]) -> tuple[str, str]:
    """Resolve the main selection metric and mode for one sweep report."""
    metric, mode, _ = _resolve_primary_metric_details(runs)
    return metric, mode


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
) -> list[dict[str, str]]:
    """Collect basic statistics for the configured ranking metrics."""
    metric_values: dict[str, list[float]] = {}
    for ranking_spec in _get_ranking_specs(runs):
        metric_name = ranking_spec["metric"]
        metric_values[metric_name] = [
            float(_get_ranking_value(run, metric_name))
            for run in runs
            if isinstance(_get_ranking_value(run, metric_name), (int, float))
        ]

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
    primary_metric, primary_mode, primary_mode_note = _resolve_primary_metric_details(
        sorted_runs
    )
    rank_method = _get_rank_method(sorted_runs)
    ranking_specs = _get_ranking_specs(sorted_runs)
    status_counts = _summarize_status_counts(sorted_runs)
    top_runs = _get_top_runs(sorted_runs)
    lines = [
        f"Sweep: {sweep_path.name}",
        f"Tracked runs: {len(sorted_runs)}",
        f"Rank method: {rank_method}",
        f"Primary metric: {primary_metric}",
        f"Mode: {primary_mode}",
        (
            "Ranking metrics: "
            + ", ".join(
                f"{spec['metric']} ({spec['mode']})" for spec in ranking_specs
            )
            if ranking_specs
            else "Ranking metrics: -"
        ),
        (
            "Status summary: "
            + ", ".join(
                f"{status}={count}"
                for status, count in sorted(status_counts.items())
            )
        ),
        "",
    ]
    if primary_mode_note:
        lines.extend([f"Mode note: {primary_mode_note}", ""])

    ranking_headers = ["Rank"]
    if rank_method == "pareto":
        ranking_headers.append("Front")
    ranking_headers.extend(["Run", "Status"])
    ranking_headers.extend(
        [
            f"{spec['metric']} [{spec['mode']}]"
            for spec in ranking_specs
        ]
    )
    ranking_rows = [
        (
            [str(run.get("_rank", rank))]
            + (
                [str(run.get("_front", "-"))]
                if rank_method == "pareto"
                else []
            )
            + [
                run["run_name"],
                str(run.get("status", "-")),
                *[
                    _format_metric_value(_get_ranking_value(run, spec["metric"]))
                    for spec in ranking_specs
                ],
            ]
        )
        for rank, run in enumerate(sorted_runs, start=1)
    ]
    lines.extend(
        [
            "Ranking",
            *(
                _render_table(
                    ranking_headers,
                    ranking_rows,
                )
            ),
        ]
    )
    if top_runs:
        lines.extend(["", "Top Performers"])
        for rank, run in enumerate(top_runs, start=1):
            lines.append(
                f"{rank}. {run['run_name']} | "
                + " | ".join(
                    (
                        f"{spec['metric']}="
                        f"{_format_metric_value(_get_ranking_value(run, spec['metric']))}"
                        f"@{_get_ranking_best_epoch(run, spec['metric']) or '-'}"
                    )
                    for spec in ranking_specs
                )
            )

    metric_statistics = _collect_metric_statistics(sorted_runs)
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
    primary_metric, primary_mode, primary_mode_note = _resolve_primary_metric_details(
        sorted_runs
    )
    rank_method = _get_rank_method(sorted_runs)
    ranking_specs = _get_ranking_specs(sorted_runs)
    status_counts = _summarize_status_counts(sorted_runs)
    top_runs = _get_top_runs(sorted_runs)
    lines = [
        "# Sweep Analysis",
        "",
        f"- Sweep: `{sweep_path.name}`",
        f"- Tracked runs: {len(sorted_runs)}",
        f"- Rank method: `{rank_method}`",
        f"- Primary metric: `{primary_metric}`",
        f"- Mode: `{primary_mode}`",
        "- Ranking metrics: "
        + (
            ", ".join(
                f"`{spec['metric']}` ({spec['mode']})" for spec in ranking_specs
            )
            if ranking_specs
            else "-"
        ),
        f"- Status summary: "
        + ", ".join(
            f"`{status}={count}`" for status, count in sorted(status_counts.items())
        ),
        "",
    ]
    if primary_mode_note:
        lines.extend([f"- Mode note: {primary_mode_note}", ""])
    lines.extend(["## Ranking", ""])

    ranking_headers = ["Rank"]
    if rank_method == "pareto":
        ranking_headers.append("Front")
    ranking_headers.extend(["Run", "Status"])
    ranking_headers.extend(
        [f"{spec['metric']} [{spec['mode']}]" for spec in ranking_specs]
    )
    ranking_rows = [
        (
            [str(run.get("_rank", rank))]
            + (
                [str(run.get("_front", "-"))]
                if rank_method == "pareto"
                else []
            )
            + [
                run["run_name"],
                str(run.get("status", "-")),
                *[
                    _format_metric_value(_get_ranking_value(run, spec["metric"]))
                    for spec in ranking_specs
                ],
            ]
        )
        for rank, run in enumerate(sorted_runs, start=1)
    ]
    lines.extend(
        _render_markdown_table(
            ranking_headers,
            ranking_rows,
        )
    )
    lines.append("")
    if top_runs:
        lines.extend(["## Top Performers", ""])
        for rank, run in enumerate(top_runs, start=1):
            lines.extend(
                [
                    f"### {rank}. {run['run_name']}",
                    "",
                    *[
                        (
                            f"- `{spec['metric']}`: "
                            f"`{_format_metric_value(_get_ranking_value(run, spec['metric']))}` "
                            f"(best epoch `{_get_ranking_best_epoch(run, spec['metric']) or '-'}`)"
                        )
                        for spec in ranking_specs
                    ],
                ]
            )
            lines.append("")

    metric_statistics = _collect_metric_statistics(sorted_runs)
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
            "  dl-analyze --sweep experiments/lr_sweep.yaml --json\n"
            "  dl-analyze --sweep experiments/lr_sweep.yaml "
            "--metric test/eer --mode min\n"
            "  dl-analyze --sweep experiments/lr_sweep.yaml "
            "--metric test/eer --mode min "
            "--metric test/accuracy --mode max "
            "--rank-method lexicographic\n\n"
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
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help=(
            "Metric used for ranking. Repeat to add lexicographic tie-breakers. "
            "Defaults to test/accuracy."
        ),
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["min", "max"],
        dest="modes",
        help=(
            "Optimization mode for each --metric. Repeat in the same order as "
            "--metric. When omitted, modes are inferred or default to max."
        ),
    )
    parser.add_argument(
        "--rank-method",
        choices=["lexicographic", "pareto"],
        default="lexicographic",
        help="How to rank runs from the requested metrics.",
    )
    args = parser.parse_args(argv)

    sweep_path = Path(args.sweep).resolve()
    try:
        ranking_specs = _build_requested_ranking_specs(args.metrics, args.modes)
    except ValueError as exc:
        parser.error(str(exc))

    runs = collect_sweep_runs(
        sweep_path,
        ranking_metrics=ranking_specs,
        rank_method=args.rank_method,
    )
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
