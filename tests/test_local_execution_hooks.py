"""Local execute_run overrides retain the public sequential and parallel hooks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

import pytest

from dl_core.executors.local import LocalExecutor


class CustomLocalExecutor(LocalExecutor):
    """An existing custom implementation that never launches standard workers."""

    def execute_run(self, run_index: int, config_path: Path) -> dict[str, Any]:
        config_path.with_suffix(".hook").write_text(str(run_index))
        return {
            "success": run_index == 0, "stopped": run_index == 1,
            "tracking_run_name": f"custom-{run_index}",
        }

    def build_command(self, config_path: str, run_config: dict | None = None) -> list[str]:
        raise AssertionError("The custom execution hook was bypassed")


class WrappedLocalExecutor(LocalExecutor):
    """A wrapper that customizes the standard hook's returned metadata."""

    def execute_run(self, run_index: int, config_path: Path) -> dict[str, Any]:
        result = super().execute_run(run_index, config_path)
        config_path.with_suffix(".hook").write_text(str(run_index))
        return {**result, "tracking_run_name": f"wrapped-{run_index}"}

    def build_command(self, config_path: str, run_config: dict | None = None) -> list[str]:
        return [sys.executable, "-c", "print('dummy finished')"]


@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("executor_type", [CustomLocalExecutor, WrappedLocalExecutor])
def test_custom_local_execution_hook_is_called_for_every_claimed_run(
    tmp_path: Path, max_workers: int, executor_type: type[LocalExecutor],
) -> None:
    """Real process-pool and sequential dispatch preserve results from the hook."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n")
    descriptors = []
    for index in range(3):
        path = tmp_path / f"run-{index}.yaml"
        path.write_text(f"runtime:\n  name: job-{index}\n  output_dir: {tmp_path / 'artifacts'}\n")
        descriptors.append((index, path))
    executor = executor_type({"sweep_file": str(sweep_path)}, "demo", "sweep-1")
    progress = executor.run_sweep(descriptors, max_workers=max_workers)
    assert [path.with_suffix(".hook").read_text() for _, path in descriptors] == ["0", "1", "2"]
    runs = executor.tracker.get_sweep_data()["runs"]
    prefix = "custom" if executor_type is CustomLocalExecutor else "wrapped"
    assert [run["tracking_run_name"] for run in runs.values()] == [f"{prefix}-{index}" for index in range(3)]
    if executor_type is CustomLocalExecutor:
        assert progress["completed"] == progress["stopped"] == progress["failed"] == 1
        assert [run["status"] for run in runs.values()] == ["completed", "stopped", "failed"]
    else:
        assert progress["completed"] == 3 and progress["failed"] == 0


@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("override_scope", ["instance", "class"])
def test_runtime_execution_override_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_workers: int, override_scope: str,
) -> None:
    """Compatibility also covers runtime hook replacement on instances or classes."""
    monkeypatch.setattr("dl_core.core.base_executor.ProcessPoolExecutor", ThreadPoolExecutor)
    executor = LocalExecutor({}, "demo", "sweep-1")
    calls: list[int] = []

    def execute_run(index: int, config_path: Path) -> dict[str, Any]:
        calls.append(index)
        return {"success": True}

    if override_scope == "instance":
        monkeypatch.setattr(executor, "execute_run", execute_run)
    else:
        def class_execute_run(self: LocalExecutor, index: int, config_path: Path) -> dict[str, Any]:
            return execute_run(index, config_path)

        monkeypatch.setattr(LocalExecutor, "execute_run", class_execute_run)
    monkeypatch.setattr(executor, "_prepare_run", lambda *args: pytest.fail("hook was bypassed"))
    progress = executor.run_sweep([(index, tmp_path / f"run-{index}.yaml") for index in range(2)], max_workers)
    assert sorted(calls) == [0, 1]
    assert progress["completed"] == 2


def test_direct_parallel_api_honors_custom_execution_hook(tmp_path: Path) -> None:
    """The public parallel method also dispatches overrides without recursion."""
    executor = CustomLocalExecutor({}, "demo", "sweep-1")
    descriptors = [(index, tmp_path / f"run-{index}.yaml") for index in range(3)]
    executor.execute_runs_parallel(descriptors, max_workers=2)
    assert all(path.with_suffix(".hook").exists() for _, path in descriptors)
    progress = executor.get_progress()
    assert progress["completed"] == progress["stopped"] == progress["failed"] == 1
