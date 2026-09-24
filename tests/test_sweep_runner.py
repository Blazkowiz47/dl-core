"""Tests for sweep runner helper behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dl_core.core import BaseExecutor
from dl_core.executors.local import LocalExecutor
from dl_core.sweep import runner
from dl_core.sweep.config.config_builder import ConfigBuilder
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


class _SlowExecutor(ClaimingExecutor):
    """Record real process-pool starts until a test releases the workers."""

    def __init__(self, *args: Any, started_dir: Path, release_file: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started_dir = started_dir
        self.release_file = release_file

    def execute_run(self, run_index: int, config_path: Path) -> dict[str, Any]:
        (self.started_dir / str(run_index)).touch()
        deadline = time.monotonic() + 5
        while not self.release_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return {"success": True}


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


@pytest.mark.parametrize(
    "filter_args",
    [
        ["--only", "run_002"],
        ["--skip", "run_000", "--skip", "run_001"],
    ],
)
def test_filtered_sweep_keeps_original_index_when_saving_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filter_args: list[str]
) -> None:
    """Filtering must not rename run 2 or record it as tracker run 0."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n", encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    base_path.write_text("runtime: {}\n", encoding="utf-8")
    sweep_config = {"base_config": str(base_path), "grid": {}, "tracking": {}}
    run_configs = [{"executor": {"name": "local"}, "runtime": {}} for _ in range(3)]
    submitted: list[tuple[int, Path]] = []

    def run_sweep(
        descriptors: list[tuple[int, Path]], max_workers: int
    ) -> dict[str, int]:
        submitted.extend(descriptors)
        return {"completed": 1, "failed": 0, "running": 0, "unknown": 0}

    monkeypatch.setattr("sys.argv", ["dl-sweep", str(sweep_path), *filter_args])
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(runner, "load_user_sweep", lambda path: sweep_config.copy())
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner,
        "generate_all_run_configs",
        lambda sweep, base: (ConfigBuilder(sweep), run_configs),
    )
    monkeypatch.setattr(
        runner.EXECUTOR_REGISTRY, "get",
        lambda *args, **kwargs: SimpleNamespace(run_sweep=run_sweep),
    )

    assert runner.main() == 0
    assert submitted == [(2, tmp_path / "sweep" / "run_002.yaml")]
    saved = yaml.safe_load(submitted[0][1].read_text(encoding="utf-8"))
    assert saved["runtime"]["name"] == "run_002"


def test_sweep_components_override_base_before_grid() -> None:
    """A sweep GPU choice must not be replaced by the base CPU default."""
    base = {
        "accelerator": {"type": "cpu"},
        "executor": {"name": "local"},
    }
    sweep = {
        "accelerator": {"type": "single_gpu"},
        "executor": {"name": "azure", "compute_target": "gpu-cluster"},
        "grid": {},
    }

    run = ConfigBuilder(sweep).generate_run_configs(base, seeds=[1])[0]
    assert run["accelerator"]["type"] == "single_gpu"
    assert run["executor"]["name"] == "azure"

    sweep["grid"] = {"accelerator.type": ["multi_gpu"]}
    overridden = ConfigBuilder(sweep).generate_run_configs(base, seeds=[1])[0]
    assert overridden["accelerator"]["type"] == "multi_gpu"


def test_base_executor_rejects_unexpected_legacy_option() -> None:
    """The compatibility path must not hide misspelled constructor options."""
    with pytest.raises(TypeError, match="Unexpected executor options: misspelled"):
        ClaimingExecutor({}, "demo", "sweep-1", misspelled=True)


def test_sweep_constructs_executor_with_base_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated executors without their own __init__ must work in dl-sweep."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n", encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    base_path.write_text("runtime: {}\n", encoding="utf-8")
    run_config = {"executor": {"name": "generated"}, "runtime": {}}

    class GeneratedExecutor(BaseExecutor):
        def setup(self, total_runs: int) -> None:
            pass

        def execute_run(self, run_index: int, config_path: Path) -> dict[str, Any]:
            return {"success": True}

        def teardown(self) -> None:
            pass

        def run_sweep(
            self, descriptors: list[tuple[int, Path]], max_workers: int = 1
        ) -> dict[str, int]:
            assert self.executor_config["compute_target"] == "cpu"
            assert self.executor_config["environment_name"] == "test-env"
            assert self.executor_config["max_workers"] == 2
            return {"completed": 1, "failed": 0, "running": 0, "unknown": 0}

    class Builder:
        def prepare_configs(self, configs: list[dict[str, Any]]) -> list[Any]:
            return [(0, configs[0], "run_000")]

        def save_configs(self, prepared: list[Any], output_dir: Path) -> list[Any]:
            return [(0, prepared[0][1], tmp_path / "run_000.yaml")]

    monkeypatch.setattr(
        "sys.argv",
        [
            "dl-sweep", str(sweep_path), "--compute", "cpu",
            "--environment", "test-env", "--max-workers", "2",
        ],
    )
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(
        runner, "load_user_sweep", lambda path: {"base_config": str(base_path)}
    )
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner,
        "generate_all_run_configs",
        lambda *args: (Builder(), [run_config]),
    )
    def get_executor(name: str, *args: Any, **kwargs: Any) -> GeneratedExecutor:
        assert name == "generated"
        assert kwargs["compute_target"] == "cpu"
        assert kwargs["environment_name"] == "test-env"
        assert kwargs["max_workers"] == 2
        return GeneratedExecutor(*args, **kwargs)

    monkeypatch.setattr(runner.EXECUTOR_REGISTRY, "get", get_executor)

    assert runner.main() == 0


def test_resume_rejects_mismatched_legacy_filtered_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Old filtered trackers cannot safely map row 0 to the new full grid."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n", encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    base_path.write_text("runtime: {}\n", encoding="utf-8")
    tracker = SweepTracker(sweep_path, "demo", "sweep-1")
    tracker.initialize_sweep(total_runs=1, user="tester")
    tracker.update_run_status(0, "completed", tracking_run_name="run_000")

    monkeypatch.setattr("sys.argv", ["dl-sweep", str(sweep_path), "--resume"])
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(
        runner, "load_user_sweep", lambda path: {"base_config": str(base_path)}
    )
    monkeypatch.setattr(
        runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo"
    )
    monkeypatch.setattr(
        runner,
        "generate_all_run_configs",
        lambda *args: (None, [{}, {}, {}]),
    )

    assert runner.main() == 1
    assert "Cannot resume" in capsys.readouterr().out
    assert tracker.get_sweep_data()["runs"]["0"]["status"] == "completed"


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


def test_tracker_write_failure_does_not_claim_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An atomic replacement failure must propagate and leave the run pending."""
    tracker = SweepTracker(tmp_path / "sweep.yaml", "demo", "sweep-1")
    tracker.initialize_sweep(total_runs=1, user="tester")

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            tracker.try_claim_run(0)

    assert tracker.get_sweep_data()["runs"]["0"]["status"] == "pending"
    assert not list(tracker.json_path.parent.glob("sweep_tracking_*.tmp"))


def test_tracker_read_and_missing_status_writes_fail_loudly(tmp_path: Path) -> None:
    """Corrupt or absent tracking data cannot be treated as a successful write."""
    tracker = SweepTracker(tmp_path / "sweep.yaml", "demo", "sweep-1")
    with pytest.raises(FileNotFoundError):
        tracker.update_run_status(0, "completed")
    with pytest.raises(FileNotFoundError):
        tracker.try_claim_run(0)

    tracker.initialize_sweep(total_runs=1, user="tester")
    tracker.json_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        tracker.get_sweep_data()


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
    assert progress == {
        "completed": 1, "failed": 0, "skipped": 1,
        "running": 0, "unknown": 0, "total": 1,
    }
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


def test_sequential_sweep_records_execution_error_and_continues(tmp_path: Path) -> None:
    """An exception must release the claimed run for a future retry."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, config_path in configs:
        config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-001",
    )

    def execute(index: int, config_path: Path) -> dict[str, Any]:
        if index == 0:
            raise RuntimeError("submission rejected")
        return ClaimingExecutor.execute_run(executor, index, config_path)

    executor.execute_run = execute
    progress = executor.run_sweep(configs, max_workers=1)

    assert executor.executed_runs == [1]
    assert progress == {
        "completed": 1, "failed": 1, "skipped": 0,
        "running": 0, "unknown": 0, "total": 2,
    }
    statuses = executor.tracker.get_sweep_data()["runs"]
    assert statuses["0"]["status"] == "failed"
    assert "submission rejected" in statuses["0"]["error_message"]
    assert statuses["1"]["status"] == "completed"
    assert executor.tracker.try_claim_run(0, config_path=str(configs[0][1]))


@pytest.mark.parametrize("max_workers", [1, 2])
def test_status_hook_is_used_at_any_worker_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_workers: int,
) -> None:
    """A custom executor should classify the same result in both paths."""
    monkeypatch.setattr(
        "dl_core.core.base_executor.ProcessPoolExecutor", ThreadPoolExecutor
    )

    class StatusExecutor(ClaimingExecutor):
        def execute_run(self, run_index: int, config_path: Path) -> dict[str, Any]:
            del config_path
            return {"state": "running" if run_index == 0 else "unknown"}

        def _classify_run_result(self, result: dict[str, Any]) -> str:
            return str(result["state"])

    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, config_path in configs:
        config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = StatusExecutor(
        {"tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-001",
    )

    progress = executor.run_sweep(configs, max_workers=max_workers)

    assert executor.submitted_runs == [0]
    assert executor.unknown_runs == [1]
    assert progress == {
        "completed": 0, "failed": 0, "skipped": 0,
        "running": 1, "unknown": 1, "total": 2,
    }


def test_interrupt_releases_sequential_claim(tmp_path: Path) -> None:
    """Ctrl+C leaves the interrupted run eligible for --resume."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-001",
    )
    executor.execute_run = lambda index, path: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )

    with pytest.raises(KeyboardInterrupt):
        executor.run_sweep([(0, config_path)], max_workers=1)

    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "failed"
    assert executor.tracker.try_claim_run(0, config_path=str(config_path))


def test_unknown_interrupt_policy_keeps_claim_non_retryable(tmp_path: Path) -> None:
    """An executor can keep ambiguous submissions out of the retry queue."""
    class UnknownExecutor(ClaimingExecutor):
        def _classify_run_result(self, result: dict[str, Any]) -> str:
            return "unknown" if result.get("unknown") else super()._classify_run_result(result)

    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = UnknownExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo", sweep_id="sweep-1",
    )
    executor.execute_run = lambda index, path: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )

    with pytest.raises(KeyboardInterrupt):
        executor.run_sweep([(0, config_path)], max_workers=1)

    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "unknown"
    assert not executor.tracker.try_claim_run(0, config_path=str(config_path))


def test_parallel_interrupt_cancels_queued_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only started local runs become retryable; unstarted jobs stay pending."""
    monkeypatch.setattr(
        "dl_core.core.base_executor.ProcessPoolExecutor", ThreadPoolExecutor
    )
    started = threading.Event()
    release = threading.Event()

    def interrupt(futures: Any) -> Any:
        assert started.wait(5)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "dl_core.core.base_executor.as_completed",
        interrupt,
    )
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(3)]
    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo", sweep_id="sweep-1",
    )

    def execute(index: int, path: Path) -> dict[str, Any]:
        started.set()
        release.wait(10)
        return {"success": True}

    executor.execute_run = execute
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.run_sweep(configs, max_workers=2)
        statuses = executor.tracker.get_sweep_data()["runs"]
        assert statuses["0"]["status"] == "failed"
        assert statuses["2"]["status"] == "pending"
    finally:
        release.set()


def test_real_process_pool_does_not_start_more_runs_after_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued process-pool work must not outlive the interrupted submission loop."""
    started_dir = tmp_path / "started"
    started_dir.mkdir()
    release_file = tmp_path / "release"

    def interrupt(futures: Any) -> Any:
        deadline = time.monotonic() + 5
        while len(list(started_dir.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(list(started_dir.iterdir())) == 2
        raise KeyboardInterrupt()

    monkeypatch.setattr("dl_core.core.base_executor.as_completed", interrupt)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(6)]
    executor = _SlowExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo", sweep_id="sweep-1",
        started_dir=started_dir, release_file=release_file,
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.run_sweep(configs, max_workers=2)
        statuses = executor.tracker.get_sweep_data()["runs"]
        assert statuses["0"]["status"] == "failed"
        assert statuses["1"]["status"] == "failed"
        assert all(statuses[str(index)]["status"] == "pending" for index in range(2, 6))
    finally:
        release_file.touch()
    time.sleep(0.3)
    assert {path.name for path in started_dir.iterdir()} == {"0", "1"}


def test_local_executor_tracks_legacy_resume_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sweep tracking must point to the directory where the trainer writes."""
    sweep_path = tmp_path / "demo_sweep.yaml"
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "artifacts"
    old_run = output_dir / "demo-exp" / "demo_sweep" / "demo-run"
    checkpoint_dir = old_run / "final" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pth").write_bytes(b"checkpoint")
    config_path.write_text(
        f"runtime:\n  name: demo-run\n  output_dir: {output_dir}\n"
        "tracking:\n  experiment_name: demo-exp\n"
        f"sweep_file: {sweep_path}\n"
        "trainer:\n  EpochTrainer:\n    continue_model: null\n",
        encoding="utf-8",
    )
    executor = LocalExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        "demo-exp", "sweep-1",
    )
    executor.build_command = lambda *args: ["unused"]
    monkeypatch.setattr(
        "dl_core.executors.local.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = executor.execute_run(0, config_path)

    assert result["artifact_dir"] == str(old_run.resolve())
    assert result["metrics_summary_path"] == str(
        old_run / "final" / "metrics" / "summary.json"
    )


def test_local_executor_tracks_old_yml_sweep_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed .yml sweep must keep its old directory and tracker path."""
    sweep_path = tmp_path / "demo_sweep.yml"
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "artifacts"
    old_run = output_dir / "sweeps" / "demo_sweep.yml" / "demo-run"
    checkpoint_dir = old_run / "final" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pth").write_bytes(b"checkpoint")
    config_path.write_text(
        f"runtime:\n  name: demo-run\n  output_dir: {output_dir}\n"
        f"sweep_file: {sweep_path}\n"
        "trainer:\n  standard:\n    continue_model: null\n",
        encoding="utf-8",
    )
    executor = LocalExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        "demo", "sweep-1",
    )
    executor.build_command = lambda *args: ["unused"]
    monkeypatch.setattr(
        "dl_core.executors.local.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = executor.execute_run(0, config_path)

    assert result["artifact_dir"] == str(old_run.resolve())


def test_tracker_error_after_accepted_run_aborts_without_releasing_claim(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An accepted job must not become claimable after a tracker write fails."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = ClaimingExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-001",
    )
    executor.execute_run = lambda index, path: {
        "success": True, "tracking_run_id": "accepted-job"
    }
    executor._update_tracker = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("tracker disk failure")
    )

    with pytest.raises(OSError, match="tracker disk failure"):
        executor.run_sweep([(0, config_path)], max_workers=1)

    assert "accepted-job" in caplog.text
    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "running"
    assert executor.failed_runs == []


def test_sweep_cli_exit_codes_reflect_failed_and_unknown_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Automation should distinguish failed, unknown, and running results."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    run_config = {"executor": {"name": "local"}, "runtime": {"name": "demo"}}

    class Builder:
        def prepare_configs(self, configs: list[dict[str, Any]]) -> list[Any]:
            return [(0, configs[0], "demo")]

        def save_configs(
            self, prepared: list[tuple[int, dict[str, Any], str]], output_dir: Path
        ) -> list[Any]:
            return [(prepared[0][0], prepared[0][1], config_path)]

    progress = {"completed": 0, "failed": 0, "running": 1, "unknown": 0, "total": 1}
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(sweep_path)])
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(runner, "load_user_sweep", lambda path: {"base_config": str(config_path)})
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner, "generate_all_run_configs", lambda *args: (Builder(), [run_config])
    )
    monkeypatch.setattr(
        runner.EXECUTOR_REGISTRY, "get",
        lambda *args, **kwargs: SimpleNamespace(run_sweep=lambda *a, **k: progress),
    )

    assert runner.main() == 0
    progress.update(running=0, unknown=1)
    assert runner.main() == 3
    progress.update(unknown=0, failed=1)
    assert runner.main() == 1
    assert "1 failed" in capsys.readouterr().out


def test_resume_does_not_call_unknown_runs_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown run needs reconciliation, not a success message."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    tracker = SweepTracker(sweep_path, "demo", "sweep-1")
    tracker.initialize_sweep(total_runs=1, user="tester")
    tracker.update_run_status(0, "unknown")
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(sweep_path), "--resume"])
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(runner, "load_user_sweep", lambda path: {"base_config": str(config_path)})
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner, "generate_all_run_configs", lambda *args: (None, [{}])
    )

    assert runner.main() == 3
    output = capsys.readouterr().out
    assert "1 unknown" in output
    assert "All runs are completed" not in output


def test_resume_exit_code_counts_other_unknown_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful retry cannot hide another run's unknown tracker status."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    tracker = SweepTracker(sweep_path, "demo", "sweep-1")
    tracker.initialize_sweep(total_runs=2, user="tester")
    tracker.update_run_status(0, "failed")
    tracker.update_run_status(1, "unknown")
    run_config = {"executor": {"name": "local"}, "runtime": {"name": "demo"}}

    class Builder:
        def prepare_configs(self, configs: list[dict[str, Any]]) -> list[Any]:
            return [(0, configs[0], "demo")]

        def save_configs(
            self, prepared: list[tuple[int, dict[str, Any], str]], output_dir: Path
        ) -> list[Any]:
            return [(prepared[0][0], prepared[0][1], config_path)]

    def run_sweep(*args: Any, **kwargs: Any) -> dict[str, int]:
        tracker.update_run_status(0, "completed")
        return {"completed": 1, "failed": 0, "running": 0, "unknown": 0, "total": 1}

    monkeypatch.setattr("sys.argv", ["dl-sweep", str(sweep_path), "--resume"])
    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(runner, "load_user_sweep", lambda path: {"base_config": str(config_path)})
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner, "generate_all_run_configs",
        lambda *args: (Builder(), [run_config.copy(), run_config.copy()]),
    )
    monkeypatch.setattr(
        runner.EXECUTOR_REGISTRY, "get",
        lambda *args, **kwargs: SimpleNamespace(run_sweep=run_sweep),
    )

    assert runner.main() == 3
