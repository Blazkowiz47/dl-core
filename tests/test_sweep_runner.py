"""Tests for sweep runner helper behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.core import BaseExecutor
from dl_core.sweep.runner import _filter_prepared_configs
from dl_core.utils.sweep_tracker import SweepTracker


class ClaimingExecutor(BaseExecutor):
    """Minimal executor for testing sweep run claiming."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize test executor state."""
        super().__init__(*args, **kwargs)
        self.executed_runs: list[int] = []

    def setup(self, total_runs: int) -> None:
        """No-op setup for tests."""

    def execute_run(
        self,
        run_index: int,
        config_path: Path,
    ) -> dict[str, Any]:
        """Record that the run reached execution."""
        del config_path
        self.executed_runs.append(run_index)
        return {"success": True}

    def teardown(self) -> None:
        """No-op teardown for tests."""


def test_filter_prepared_configs_applies_only_and_skip_patterns() -> None:
    """Run-name filters should keep only the requested subset."""
    prepared_configs = [
        (0, {"seed": 2025}, "backbone_swin_s_seed_2025"),
        (1, {"seed": 2025}, "backbone_swin_b_seed_2025"),
        (2, {"seed": 2026}, "backbone_convnext_seed_2026"),
    ]

    filtered = _filter_prepared_configs(
        prepared_configs,
        ["backbone_swin_*"],
        ["*_b_*"],
    )

    assert filtered == [(0, {"seed": 2025}, "backbone_swin_s_seed_2025")]


def test_sweep_tracker_claims_only_pending_or_failed_runs(tmp_path: Path) -> None:
    """Run claiming should prevent duplicate execution of live runs."""
    sweep_path = tmp_path / "experiments" / "demo_sweep.yaml"
    sweep_path.parent.mkdir(parents=True)
    sweep_path.write_text("base_config: configs/base.yaml\n")

    tracker = SweepTracker(sweep_path, "demo", "sweep-001")
    tracker.initialize_sweep(total_runs=3, user="tester")
    tracker.update_run_status(1, "running")
    tracker.update_run_status(2, "completed")

    assert tracker.try_claim_run(0, config_path="run-0.yaml") is True
    assert tracker.try_claim_run(1, config_path="run-1.yaml") is False
    assert tracker.try_claim_run(2, config_path="run-2.yaml") is False

    sweep_data = tracker.get_sweep_data()
    assert sweep_data["runs"]["0"]["status"] == "running"
    assert sweep_data["runs"]["0"]["config_path"] == "run-0.yaml"
    assert sweep_data["runs"]["1"]["status"] == "running"
    assert sweep_data["runs"]["2"]["status"] == "completed"


def test_sequential_sweep_skips_runs_claimed_by_another_process(
    tmp_path: Path,
) -> None:
    """Queued descriptors should be re-checked before sequential execution."""
    sweep_path = tmp_path / "experiments" / "demo_sweep.yaml"
    sweep_path.parent.mkdir(parents=True)
    sweep_path.write_text("base_config: configs/base.yaml\n")
    config_0 = tmp_path / "run-0.yaml"
    config_1 = tmp_path / "run-1.yaml"
    config_0.write_text("runtime:\n  name: run-0\n")
    config_1.write_text("runtime:\n  name: run-1\n")

    tracker = SweepTracker(sweep_path, "demo", "sweep-001")
    tracker.initialize_sweep(total_runs=2, user="tester")
    tracker.update_run_status(1, "running")

    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-002",
        resume=True,
    )

    progress = executor.run_sweep([(0, config_0), (1, config_1)], max_workers=1)

    assert executor.executed_runs == [0]
    assert progress == {"completed": 1, "failed": 0, "skipped": 1, "total": 1}
    sweep_data = tracker.get_sweep_data()
    assert sweep_data["runs"]["0"]["status"] == "completed"
    assert sweep_data["runs"]["1"]["status"] == "running"


def test_parallel_wrapper_skips_runs_claimed_by_another_process(
    tmp_path: Path,
) -> None:
    """Parallel workers should re-check the tracker before execution."""
    sweep_path = tmp_path / "experiments" / "demo_sweep.yaml"
    sweep_path.parent.mkdir(parents=True)
    sweep_path.write_text("base_config: configs/base.yaml\n")
    config_path = tmp_path / "run-0.yaml"
    config_path.write_text("runtime:\n  name: run-0\n")

    tracker = SweepTracker(sweep_path, "demo", "sweep-001")
    tracker.initialize_sweep(total_runs=1, user="tester")
    tracker.update_run_status(0, "running")

    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-002",
        resume=True,
    )

    result = executor._execute_single_run_wrapper(0, config_path)

    assert result == {"success": True, "skipped": True}
    assert executor.executed_runs == []
