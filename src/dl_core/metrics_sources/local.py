"""Local metrics source for file-based sweep analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.core import BaseMetricsSource, register_metrics_source
from dl_core.utils.artifact_manager import get_run_artifact_dir
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
        artifact_dir = self._infer_artifact_dir(run_data)
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
            or Path(run_data.get("config_path", f"run_{run_index}.yaml")).stem
        )

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
            "selection_value": self._resolve_selection_value(summary),
            "best_epoch": summary.get("best_epoch"),
            "final_epoch": summary.get("final_epoch"),
            "best_metrics": summary.get("best_metrics", {}),
            "final_metrics": summary.get("final_metrics", {}),
        }

    def _infer_artifact_dir(self, run_data: dict[str, Any]) -> Path | None:
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

        return Path(
            get_run_artifact_dir(
                run_name=run_name,
                output_dir=output_dir,
                experiment_name=experiment_name,
                sweep_name=sweep_name,
            )
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
