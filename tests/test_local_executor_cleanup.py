"""Local launch failures, stopped results, and unconditional lifecycle cleanup."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import pytest

from dl_core.core import BaseExecutor
from dl_core.executors.local import LocalExecutor
from dl_core.utils.sweep_tracker import SweepTracker


@pytest.mark.parametrize("max_workers", [1, 2])
def test_stopped_result_survives_shared_result_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_workers: int,
) -> None:
    """Both shared dispatch paths preserve stopped instead of coercing it to failed."""
    class StoppedExecutor(BaseExecutor):
        def setup(self, total_runs: int) -> None:
            pass

        def execute_run(self, index: int, config_path: Path) -> dict[str, Any]:
            return {"success": False, "stopped": True}

        def teardown(self) -> None:
            pass

    monkeypatch.setattr("dl_core.core.base_executor.ProcessPoolExecutor", ThreadPoolExecutor)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n")
    executor = StoppedExecutor({"sweep_file": str(sweep_path)}, "demo", "sweep-1")
    progress = executor.run_sweep([(index, tmp_path / f"run-{index}.yaml") for index in range(2)], max_workers)
    assert progress["completed"] == progress["failed"] == 0
    assert progress["stopped"] == progress["total"] == 2
    assert all(run["status"] == "stopped" for run in executor.tracker.get_sweep_data()["runs"].values())
    assert not executor.tracker.try_claim_run(0)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("setup failed")])
def test_local_setup_error_finalizes_tracker_and_executor_once(
    monkeypatch: pytest.MonkeyPatch, error: BaseException,
) -> None:
    """An interrupt before execution still closes the local tracker lifecycle."""
    executor = LocalExecutor({}, "demo", "sweep-1")
    calls: list[str] = []

    def setup(total_runs: int) -> None:
        raise error

    monkeypatch.setattr(executor, "setup", setup)
    monkeypatch.setattr(executor, "teardown", lambda: calls.append("executor"))
    monkeypatch.setattr(executor.run_tracker, "teardown_sweep", lambda: calls.append("tracker"))
    with pytest.raises(type(error)):
        executor.run_sweep([], max_workers=1)
    assert calls.count("tracker") == calls.count("executor") == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("launch failed")])
def test_single_local_run_finalizes_executor_after_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException,
) -> None:
    """The direct dl-run lifecycle also tears down after errors and interrupts."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: job\n")
    executor = LocalExecutor({}, "demo", "run-1")
    finalized: list[bool] = []

    def execute_run(index: int, path: Path) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(executor, "execute_run", execute_run)
    monkeypatch.setattr(executor, "teardown", lambda: finalized.append(True))
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            executor.run(str(config_path))
    else:
        assert not executor.run(str(config_path))
    assert finalized == [True]


def test_launch_error_preserves_other_results_and_closes_owned_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed spawn does not lose other jobs, descriptors, or signal handlers."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n")
    descriptors = []
    for index in range(3):
        path = tmp_path / f"run-{index}.yaml"
        path.write_text(f"runtime:\n  name: job-{index}\n  output_dir: {tmp_path / 'artifacts'}\n")
        descriptors.append((index, path))
    executor = LocalExecutor({"sweep_file": str(sweep_path)}, "demo", "sweep-1")
    monkeypatch.setattr(executor, "build_command", lambda path, config: (
        [str(tmp_path / "missing-command")] if Path(path).stem == "run-1"
        else [sys.executable, "-c", "print('dummy finished')"]
    ))
    processes: list[subprocess.Popen[bytes]] = []
    original_popen = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("dl_core.executors.local_supervisor.subprocess.Popen", popen)
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    closed: list[bool] = []
    monkeypatch.setattr(executor.run_tracker, "teardown_sweep", lambda: closed.append(True))
    progress = executor.run_sweep(descriptors, max_workers=2)
    assert progress["completed"] == 2 and progress["failed"] == 1
    assert [run["status"] for run in executor.tracker.get_sweep_data()["runs"].values()] == ["completed", "failed", "completed"]
    assert all(process.poll() == 0 and process.stdout.closed for process in processes)
    assert all(signal.getsignal(signum) == handler for signum, handler in handlers.items())
    assert closed == [True]


def test_signal_during_prepare_does_not_launch_the_claimed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop-all request before Popen restores the unstarted row to pending."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n")
    path = tmp_path / "run.yaml"
    path.write_text(f"runtime:\n  name: job\n  output_dir: {tmp_path / 'artifacts'}\n")
    executor = LocalExecutor({"sweep_file": str(sweep_path)}, "demo", "sweep-1")
    prepare = executor._prepare_run

    def prepare_and_signal(index: int, config_path: Path) -> tuple[list[str], dict[str, Any]]:
        result = prepare(index, config_path)
        signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(executor, "_prepare_run", prepare_and_signal)
    monkeypatch.setattr("dl_core.executors.local_supervisor.subprocess.Popen", lambda *args, **kwargs: pytest.fail("launched after stop-all"))
    with pytest.raises(KeyboardInterrupt):
        executor.run_sweep([(0, path)])
    assert executor.claimed_runs == []
    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "pending"


def test_unknown_shutdown_result_is_not_stopped_or_retryable(tmp_path: Path) -> None:
    """An unconfirmed termination remains unknown rather than becoming stopped."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n")
    tracker = SweepTracker(sweep_path, "demo", "sweep-1")
    tracker.initialize_sweep(total_runs=1, user="tester")
    assert tracker.try_claim_run(0)
    executor = LocalExecutor({"sweep_file": str(sweep_path)}, "demo", "sweep-1", resume=True)
    result = executor._finish_run(
        {"artifact_dir": str(tmp_path / "artifacts"), "tracking_run_name": "job"},
        None, stopped=True, unknown=True,
    )
    executor._record_run_result(0, tmp_path / "run.yaml", result)
    assert executor.get_progress()["stopped"] == 0
    assert executor.get_progress()["unknown"] == 1
    assert tracker.get_sweep_data()["runs"]["0"]["status"] == "unknown"
    assert not tracker.try_claim_run(0, claimable_statuses=("pending", "failed", "stopped"))
