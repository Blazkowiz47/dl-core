"""Tests for artifact sync CLI behavior."""

from __future__ import annotations

import json
from pathlib import Path

from dl_core.core import BaseMetricsSource, register_metrics_source
from dl_core.sync import main as sync_main


@register_metrics_source("sync_test")
class _SyncTestMetricsSource(BaseMetricsSource):
    """Small metrics source used to test artifact sync wiring."""

    def collect_run(
        self,
        run_index: int,
        run_data: dict[str, object],
        sweep_data: dict[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError

    def sync_run_artifacts(
        self,
        run_index: int,
        run_data: dict[str, object],
        sweep_data: dict[str, object],
        *,
        force: bool = False,
    ) -> dict[str, object]:
        tracking_dir = Path(str(sweep_data["_tracking_dir"]))
        artifact_dir = tracking_dir / "synced" / f"run_{run_index}"
        final_metrics_dir = artifact_dir / "final" / "metrics"
        final_metrics_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "config.yaml").write_text("{}", encoding="utf-8")
        (final_metrics_dir / "summary.json").write_text("{}", encoding="utf-8")
        (final_metrics_dir / "history.json").write_text("{}", encoding="utf-8")
        if force:
            (artifact_dir / "forced.txt").write_text("1", encoding="utf-8")
        return {
            "config_path": str(tracking_dir / f"run_{run_index}.yaml"),
            "artifact_dir": str(artifact_dir),
            "metrics_summary_path": str(final_metrics_dir / "summary.json"),
            "metrics_history_path": str(final_metrics_dir / "history.json"),
        }


def test_sync_cli_updates_sweep_tracker_paths(tmp_path: Path) -> None:
    """Artifact sync should patch tracker paths for completed runs."""
    sweep_path = tmp_path / "experiments" / "demo.yaml"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text("base_config: configs/base.yaml\n", encoding="utf-8")

    tracking_dir = sweep_path.parent / sweep_path.stem
    tracking_dir.mkdir(parents=True, exist_ok=True)
    (tracking_dir / "run_0.yaml").write_text("{}", encoding="utf-8")
    (tracking_dir / "run_1.yaml").write_text("{}", encoding="utf-8")
    tracking_payload = {
        "experiment_name": "demo",
        "sweep_id": "sync_demo",
        "tracking_backend": "sync_test",
        "metrics_source_backend": "sync_test",
        "runs": {
            "0": {"status": "completed", "tracking_backend": "sync_test"},
            "1": {"status": "pending", "tracking_backend": "sync_test"},
        },
    }
    (tracking_dir / "sweep_tracking.json").write_text(
        json.dumps(tracking_payload, indent=2),
        encoding="utf-8",
    )

    assert sync_main(["--sweep", str(sweep_path), "--artifacts"]) == 0

    updated = json.loads((tracking_dir / "sweep_tracking.json").read_text())
    run_zero = updated["runs"]["0"]
    run_one = updated["runs"]["1"]
    assert run_zero["artifact_dir"].endswith("synced/run_0")
    assert run_zero["metrics_summary_path"].endswith("final/metrics/summary.json")
    assert run_zero["metrics_history_path"].endswith("final/metrics/history.json")
    assert run_one.get("artifact_dir") is None
