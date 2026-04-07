"""Local metrics source for file-based sweep analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.core import BaseMetricsSource, register_metrics_source
from dl_core.utils.artifact_manager import resolve_existing_run_artifact_dir
from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)


def _normalize_metric_key(key: str) -> str:
    """Normalize metric keys so separator differences still match."""
    return "".join(char for char in key.casefold() if char.isalnum())


@register_metrics_source("local")
class LocalMetricsSource(BaseMetricsSource):
    """Read normalized run data from local artifact files."""

    def prepare_sweep(
        self,
        run_items: list[tuple[int, dict[str, Any]]],
        sweep_data: dict[str, Any],
        progress_callback: Any | None = None,
    ) -> None:
        """Prefetch requested local metric histories once per sweep."""
        ranking_specs = self._get_requested_ranking_specs(sweep_data)
        if not ranking_specs:
            for _ in run_items:
                if progress_callback is not None:
                    progress_callback()
            return

        history_cache = sweep_data.setdefault("_local_metric_history_cache", {})
        for run_index, run_data in run_items:
            config_path = self._resolve_config_path(run_index, run_data, sweep_data)
            artifact_dir = self._infer_artifact_dir(run_data, config_path)
            history_path = self._resolve_metrics_path(
                run_data,
                artifact_dir,
                filename="history.json",
                tracker_key="metrics_history_path",
            )

            run_cache: dict[str, list[dict[str, int | float]]] = {}
            if history_path is not None and history_path.exists():
                history_payload = self.load_json(history_path)
                for ranking_spec in ranking_specs:
                    metric_name = ranking_spec["metric"]
                    run_cache[metric_name] = self._extract_local_metric_history(
                        history_payload,
                        metric_name,
                    )

            history_cache[str(run_index)] = run_cache
            if progress_callback is not None:
                progress_callback()

    def collect_run(
        self,
        run_index: int,
        run_data: dict[str, Any],
        sweep_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect one normalized analyzer record from local artifact files.

        Args:
            run_index: Sweep run index
            run_data: Per-run tracker payload
            sweep_data: Full sweep tracker payload

        Returns:
            Normalized run record for the analyzer.
        """
        config_path = self._resolve_config_path(run_index, run_data, sweep_data)
        artifact_dir = self._infer_artifact_dir(run_data, config_path)
        summary_path = self._resolve_metrics_path(
            run_data,
            artifact_dir,
            filename="summary.json",
            tracker_key="metrics_summary_path",
        )
        history_path = self._resolve_metrics_path(
            run_data,
            artifact_dir,
            filename="history.json",
            tracker_key="metrics_history_path",
        )

        summary: dict[str, Any] = {}
        if summary_path is not None and summary_path.exists():
            summary = self.load_json(summary_path)

        run_name = (
            run_data.get("tracking_run_name")
            or summary.get("run_name")
            or Path(config_path or f"run_{run_index}.yaml").stem
        )

        ranking_entries = self._build_requested_ranking_entries(
            run_index=run_index,
            summary=summary,
            history_path=history_path,
            sweep_data=sweep_data,
        )
        if ranking_entries and not any(
            isinstance(entry.get("value"), (int, float)) for entry in ranking_entries
        ):
            ranking_entries = []
        selection_metric = summary.get("selection_metric")
        selection_mode = summary.get("selection_mode")
        selection_value = self._resolve_selection_value(summary)
        best_epoch = summary.get("best_epoch")
        if ranking_entries:
            first_entry = ranking_entries[0]
            selection_metric = first_entry["metric"]
            selection_mode = first_entry["mode"]
            selection_value = first_entry["value"]
            best_epoch = first_entry["best_epoch"]

        return {
            "run_index": run_index,
            "run_name": run_name,
            "status": run_data.get("status", "unknown"),
            "error_message": (
                run_data.get("error_message") or summary.get("error_message")
            ),
            "tracking_backend": run_data.get(
                "tracking_backend",
                sweep_data.get("tracking_backend"),
            ),
            "metrics_source_backend": run_data.get(
                "metrics_source_backend",
                sweep_data.get("metrics_source_backend"),
            ),
            "tracking_run_ref": run_data.get("tracking_run_ref"),
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
            "config_path": str(config_path) if config_path is not None else None,
            "metrics_summary_path": (
                str(summary_path) if summary_path is not None else None
            ),
            "metrics_history_path": (
                str(history_path) if history_path is not None else None
            ),
            "summary_available": bool(summary),
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selection_value": selection_value,
            "best_epoch": best_epoch,
            "final_epoch": summary.get("final_epoch"),
            "best_metrics": summary.get("best_metrics", {}),
            "final_metrics": summary.get("final_metrics", {}),
            "ranking_metrics": ranking_entries,
            "rank_method": self._resolve_rank_method(sweep_data),
        }

    def _resolve_config_path(
        self,
        run_index: int,
        run_data: dict[str, Any],
        sweep_data: dict[str, Any],
    ) -> Path | None:
        """
        Resolve the local generated config path for one tracked run.

        Args:
            run_index: Sweep run index
            run_data: Sweep tracker entry for one run
            sweep_data: Full sweep tracker payload

        Returns:
            Local config path when it can be resolved.
        """
        config_path_value = run_data.get("config_path")
        if isinstance(config_path_value, str) and config_path_value:
            return Path(config_path_value)

        tracking_dir_value = sweep_data.get("_tracking_dir")
        if not isinstance(tracking_dir_value, str) or not tracking_dir_value:
            return None

        tracking_dir = Path(tracking_dir_value)
        run_name = run_data.get("tracking_run_name")
        candidate_names = []
        if isinstance(run_name, str) and run_name:
            candidate_names.append(f"{run_name}.yaml")
        candidate_names.append(f"run_{run_index}.yaml")

        for candidate_name in candidate_names:
            candidate_path = tracking_dir / candidate_name
            if candidate_path.exists():
                return candidate_path

        return None

    def _infer_artifact_dir(
        self,
        run_data: dict[str, Any],
        config_path: Path | None,
    ) -> Path | None:
        """
        Resolve the local artifact directory for a tracked run.

        Args:
            run_data: Sweep tracker entry for one run
            config_path: Resolved local config path for the run

        Returns:
            Path to the artifact directory, if it can be resolved.
        """
        artifact_dir = run_data.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir:
            return Path(artifact_dir)

        if config_path is None:
            return None
        if not config_path.exists():
            return None

        config = self.load_yaml(config_path)
        runtime_config = config.get("runtime", {})
        run_name = resolve_config_run_name(config, config_path=config_path)
        output_dir = runtime_config.get("output_dir", "artifacts")
        experiment_name = resolve_config_experiment_name(
            config,
            config_path=config_path,
        )

        sweep_name = None
        sweep_file = config.get("sweep_file")
        if isinstance(sweep_file, str) and sweep_file:
            sweep_name = Path(sweep_file).stem

        return resolve_existing_run_artifact_dir(
            run_name=run_name,
            output_dir=output_dir,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
        )

    def _resolve_metrics_path(
        self,
        run_data: dict[str, Any],
        artifact_dir: Path | None,
        *,
        filename: str,
        tracker_key: str,
    ) -> Path | None:
        """
        Resolve a metrics artifact path from tracker data or artifact structure.

        Args:
            run_data: Sweep tracker entry for one run
            artifact_dir: Resolved artifact directory, if available
            filename: File expected under ``final/metrics/``
            tracker_key: Explicit tracker key for the artifact

        Returns:
            Resolved path, if available.
        """
        tracker_value = run_data.get(tracker_key)
        if isinstance(tracker_value, str) and tracker_value:
            return Path(tracker_value)

        if artifact_dir is None:
            return None

        return artifact_dir / "final" / "metrics" / filename

    def _resolve_selection_config(
        self,
        config_path: Path | None,
    ) -> tuple[str | None, str | None]:
        """
        Resolve the configured ranking metric and mode from a run config.

        Args:
            config_path: Local config path for the run

        Returns:
            Tuple of selection metric and selection mode.
        """
        if config_path is None or not config_path.exists():
            return None, None

        config = self.load_yaml(config_path)
        callbacks_config = config.get("callbacks", {})
        if not isinstance(callbacks_config, dict):
            return None, None

        checkpoint_config = callbacks_config.get("checkpoint")
        if not isinstance(checkpoint_config, dict):
            return None, None

        monitor = checkpoint_config.get("monitor")
        mode = checkpoint_config.get("mode", "min")
        if not isinstance(monitor, str) or not monitor:
            return None, None
        if mode not in {"min", "max"}:
            mode = "min"
        return monitor, mode

    def _resolve_selection_value(self, summary: dict[str, Any]) -> Any:
        """
        Resolve the value used for ranking a run in analyzer output.

        Args:
            summary: Loaded local metrics summary payload

        Returns:
            Numeric selection value when available, otherwise ``None``.
        """
        selection_value = summary.get("selection_value")
        if isinstance(selection_value, (int, float)):
            return selection_value

        selection_metric = summary.get("selection_metric")
        if not isinstance(selection_metric, str) or not selection_metric:
            return selection_value

        for section_name in ("best_metrics", "final_metrics", "best", "final"):
            section = summary.get(section_name)
            if not isinstance(section, dict):
                continue

            if selection_metric in section:
                return section[selection_metric]

            normalized_selection = _normalize_metric_key(selection_metric)
            for metric_name, metric_value in section.items():
                if _normalize_metric_key(metric_name) == normalized_selection:
                    return metric_value

        return selection_value

    def _get_requested_ranking_specs(
        self,
        sweep_data: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Return explicit analyzer ranking specs, if configured."""
        ranking_specs = sweep_data.get("_ranking_metrics")
        if not isinstance(ranking_specs, list):
            return []
        normalized_specs: list[dict[str, str]] = []
        for ranking_spec in ranking_specs:
            if not isinstance(ranking_spec, dict):
                continue
            metric_name = ranking_spec.get("metric")
            metric_mode = ranking_spec.get("mode")
            if not isinstance(metric_name, str) or not metric_name:
                continue
            if metric_mode not in {"min", "max"}:
                continue
            normalized_specs.append({"metric": metric_name, "mode": metric_mode})
        return normalized_specs

    def _resolve_rank_method(self, sweep_data: dict[str, Any]) -> str:
        """Return the configured analyzer rank method."""
        rank_method = sweep_data.get("_rank_method")
        if rank_method in {"lexicographic", "pareto", "rank-sum"}:
            return str(rank_method)
        return "lexicographic"

    def _extract_local_metric_history(
        self,
        history_payload: dict[str, Any],
        metric_name: str,
    ) -> list[dict[str, int | float]]:
        """Extract one normalized local metric history from ``history.json``."""
        split_name, separator, metric_key = metric_name.partition("/")
        if not separator:
            return []

        split_history = history_payload.get(split_name)
        if not isinstance(split_history, dict):
            return []

        normalized_metric_key = _normalize_metric_key(metric_key)
        metric_history: list[dict[str, int | float]] = []
        for epoch_key in sorted(split_history, key=lambda value: int(value)):
            epoch_metrics = split_history.get(epoch_key)
            if not isinstance(epoch_metrics, dict):
                continue

            metric_value = epoch_metrics.get(metric_key)
            if not isinstance(metric_value, (int, float)):
                for recorded_metric, recorded_value in epoch_metrics.items():
                    if _normalize_metric_key(recorded_metric) == normalized_metric_key:
                        metric_value = recorded_value
                        break

            if not isinstance(metric_value, (int, float)):
                continue

            metric_history.append(
                {"step": int(epoch_key), "value": float(metric_value)}
            )

        return metric_history

    def _resolve_best_epoch(
        self,
        history: list[dict[str, int | float]],
        mode: str,
    ) -> tuple[int | None, float | None]:
        """Resolve the best epoch and value for one metric history."""
        if not history:
            return None, None

        best_epoch: int | None = None
        best_value: float | None = None
        for point in history:
            epoch = int(point["step"])
            metric_value = float(point["value"])
            if best_value is None:
                best_epoch = epoch
                best_value = metric_value
                continue

            is_better = (
                metric_value < best_value if mode == "min" else metric_value > best_value
            )
            if is_better:
                best_epoch = epoch
                best_value = metric_value

        return best_epoch, best_value

    def _resolve_metric_from_mapping(
        self,
        metrics: dict[str, Any],
        metric_name: str,
    ) -> Any:
        """Resolve one metric value from a metric mapping using normalized keys."""
        if metric_name in metrics:
            return metrics[metric_name]

        normalized_metric_name = _normalize_metric_key(metric_name)
        for recorded_metric, recorded_value in metrics.items():
            if _normalize_metric_key(recorded_metric) == normalized_metric_name:
                return recorded_value
        return None

    def _build_requested_ranking_entries(
        self,
        *,
        run_index: int,
        summary: dict[str, Any],
        history_path: Path | None,
        sweep_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build normalized ranking entries from requested analyzer metrics."""
        ranking_specs = self._get_requested_ranking_specs(sweep_data)
        if not ranking_specs:
            return []

        history_cache = sweep_data.get("_local_metric_history_cache", {})
        run_history_cache = (
            history_cache.get(str(run_index), {})
            if isinstance(history_cache, dict)
            else {}
        )
        if not isinstance(run_history_cache, dict):
            run_history_cache = {}

        history_payload: dict[str, Any] | None = None
        ranking_entries: list[dict[str, Any]] = []
        final_metrics = summary.get("final_metrics", {})
        best_metrics = summary.get("best_metrics", {})
        for ranking_spec in ranking_specs:
            metric_name = ranking_spec["metric"]
            history = run_history_cache.get(metric_name)
            if history is None:
                if history_payload is None:
                    history_payload = (
                        self.load_json(history_path)
                        if history_path is not None and history_path.exists()
                        else {}
                    )
                history = self._extract_local_metric_history(history_payload, metric_name)

            best_epoch, best_value = self._resolve_best_epoch(
                history if isinstance(history, list) else [],
                ranking_spec["mode"],
            )
            final_value = (
                self._resolve_metric_from_mapping(final_metrics, metric_name)
                if isinstance(final_metrics, dict)
                else None
            )
            if final_value is None and isinstance(history, list) and history:
                final_value = history[-1]["value"]
            if best_value is None and isinstance(best_metrics, dict):
                best_value = self._resolve_metric_from_mapping(best_metrics, metric_name)

            ranking_entries.append(
                {
                    "metric": metric_name,
                    "mode": ranking_spec["mode"],
                    "value": best_value,
                    "best_epoch": best_epoch,
                    "final_value": final_value,
                }
            )

        return ranking_entries
