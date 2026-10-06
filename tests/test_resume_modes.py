"""Local resume status selection, claiming, and nonlocal compatibility."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dl_core.core import BaseExecutor
from dl_core.executors.local import LocalExecutor
from dl_core.sweep import runner
from dl_core.sweep.config.config_builder import ConfigBuilder
from dl_core.utils.sweep_tracker import SweepTracker

FLAGS = ["--resume", "--resume-failed", "--resume-stopped", "--resume-all"]


@pytest.fixture
def resume_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide every status plus a pending grid row outside the stored selection."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: base.yaml\n", encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    base_path.write_text("runtime: {}\n", encoding="utf-8")
    run_configs = [
        {"executor": {"name": "local"}, "runtime": {}, "seed": 7, "fixture_index": index}
        for index in range(7)
    ]
    sweep = {"base_config": str(base_path), "grid": {"fixture_index": list(range(7))}}
    names = {
        index: name for index, _, name in ConfigBuilder(sweep).prepare_configs(run_configs)
        if index < 6
    }
    tracker = SweepTracker(sweep_path, "demo", "original-sweep")
    tracker.initialize_sweep(
        total_runs=7, user="tester", selected_run_indices=list(range(6)),
        selected_run_names=names, tracking_context="original-context",
    )
    for index, status in enumerate(["pending", "failed", "stopped", "completed", "running", "unknown"]):
        tracker.update_run_status(index, status)
    attempts: list[int] = []
    instances: list[LocalExecutor] = []
    outcome = {"success": True}

    class RecordingExecutor(LocalExecutor):
        def _execute_runs(self, descriptors: list[tuple[int, Path]], max_workers: int) -> None:
            BaseExecutor._execute_runs(self, descriptors, max_workers=1)

        def execute_run(self, index: int, config_path: Path) -> dict[str, Any]:
            attempts.append(index)
            return dict(outcome)

    def get_executor(name: str, *args: Any, **kwargs: Any) -> LocalExecutor:
        executor = RecordingExecutor(*args, **kwargs)
        instances.append(executor)
        return executor

    monkeypatch.setattr(runner, "setup_logging", lambda level: None)
    monkeypatch.setattr(runner, "load_builtin_components", lambda: None)
    monkeypatch.setattr(runner, "load_local_components", lambda path: None)
    monkeypatch.setattr(runner, "load_user_sweep", lambda path: deepcopy(sweep))
    monkeypatch.setattr(runner, "ensure_tracking_experiment_name", lambda *args, **kwargs: "demo")
    monkeypatch.setattr(
        runner, "generate_all_run_configs",
        lambda sweep, base: (ConfigBuilder(sweep), deepcopy(run_configs)),
    )
    monkeypatch.setattr(runner.EXECUTOR_REGISTRY, "get", get_executor)
    return SimpleNamespace(
        path=sweep_path, tracker=tracker, attempts=attempts, configs=run_configs,
        instances=instances, outcome=outcome,
    )


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--resume", [0]), ("--resume-failed", [1]), ("--resume-stopped", [2]), ("--resume-all", [0, 1, 2])],
)
def test_resume_flags_execute_and_claim_exact_statuses(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    flag: str, expected: list[int],
) -> None:
    """Filtering and atomic claiming agree, including stopped-only resumes."""
    case = resume_case
    before = case.tracker.get_sweep_data()
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), flag])
    assert runner.main() == 0
    assert case.attempts == expected
    executor = case.instances[0]
    assert executor.resume is True
    assert executor.claimed_runs == expected
    assert executor.tracking_context == "original-context"
    after = case.tracker.get_sweep_data()
    assert after["sweep_id"] == "original-sweep"
    assert after["selected_run_names"] == before["selected_run_names"]
    assert set(after["runs"]) == set(before["runs"])
    for index, run in before["runs"].items():
        if int(index) in expected:
            assert after["runs"][index]["status"] == "completed"
        else:
            assert after["runs"][index] == run


@pytest.mark.parametrize("flag", FLAGS)
def test_empty_resume_selection_succeeds_without_reinitializing(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], flag: str,
) -> None:
    """Excluded historical failures and unknown rows do not fail an empty command."""
    case = resume_case
    for index in (0, 1, 2):
        case.tracker.update_run_status(index, "completed")
    original = case.tracker.json_path.read_bytes()
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), flag])
    assert runner.main() == 0
    assert case.attempts == []
    assert case.instances == []
    assert case.tracker.json_path.read_bytes() == original
    output = capsys.readouterr().out
    assert "No " in output and "runs to resume" in output
    assert "1 running" in output and "1 unknown" in output
    assert "All runs are completed" not in output


def test_current_failure_still_fails_pending_only_resume(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the current command's claimed work determines local failure results."""
    case = resume_case
    case.outcome["success"] = False
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), "--resume"])
    assert runner.main() == 1
    assert case.attempts == [0]
    assert case.tracker.get_sweep_data()["runs"]["0"]["status"] == "failed"


def test_changed_status_is_rechecked_at_atomic_claim(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped row claimed elsewhere after filtering is skipped by this command."""
    case = resume_case
    original_claim = BaseExecutor._claim_run_for_execution

    def claim(executor: BaseExecutor, index: int, config_path: Path) -> bool:
        case.tracker.update_run_status(index, "running")
        return original_claim(executor, index, config_path)

    monkeypatch.setattr(BaseExecutor, "_claim_run_for_execution", claim)
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), "--resume-stopped"])
    assert runner.main() == 0
    assert case.attempts == []
    assert case.instances[0].claimed_runs == []
    assert case.instances[0].skipped_runs == [2]


def test_cli_local_override_selects_local_policy_before_filtering(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--executor local takes precedence over a remote executor in generated configs."""
    case = resume_case
    for config in case.configs:
        config["executor"]["name"] = "azure"
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), "--executor", "local", "--resume-stopped"])
    assert runner.main() == 0
    assert case.attempts == [2]


@pytest.mark.parametrize("first_executor", ["local", "azure"])
@pytest.mark.parametrize("flags", [("--resume",), ("--resume-stopped",), ("--resume-all",), ("--overwrite",)])
def test_mixed_executor_grid_is_rejected_before_tracking_or_dispatch(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], first_executor: str, flags: tuple[str, ...],
) -> None:
    """Filtering must never change which backend receives the resume policy."""
    case = resume_case
    for config in case.configs:
        config["executor"]["name"] = "azure" if first_executor == "local" else "local"
    case.configs[0]["executor"]["name"] = first_executor
    original = case.tracker.json_path.read_bytes()
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), *flags])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "mixed executor names" in capsys.readouterr().err
    assert case.instances == case.attempts == []
    assert case.tracker.json_path.read_bytes() == original


def test_explicit_local_override_unifies_a_mixed_grid(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit backend selection is reused even after status filtering."""
    case = resume_case
    case.configs[2]["executor"]["name"] = "azure"
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), "--executor", "local", "--resume-stopped"])
    assert runner.main() == 0
    assert case.attempts == [2]
    assert case.instances[0].executor_config["name"] == "local"
    assert case.instances[0].sweep_config["_resume_statuses"] == ("stopped",)


@pytest.mark.parametrize("flag", FLAGS[1:])
def test_nonlocal_executor_rejects_local_only_resume_flags(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, flag: str,
) -> None:
    """Remote sweeps do not acquire new resume modes accidentally."""
    case = resume_case
    for config in case.configs:
        config["executor"]["name"] = "azure"
    before = case.tracker.json_path.read_bytes()
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), flag])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert case.attempts == []
    assert case.tracker.json_path.read_bytes() == before


def test_azure_resume_keeps_failed_pending_and_historical_exit_code(
    resume_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared runner changes preserve existing Azure status selection and exit policy."""
    case = resume_case
    for config in case.configs:
        config["executor"]["name"] = "azure"
    monkeypatch.setattr("sys.argv", ["dl-sweep", str(case.path), "--resume"])
    assert runner.main() == 3
    assert case.attempts == [0, 1]
    assert case.tracker.get_sweep_data()["runs"]["2"]["status"] == "stopped"


@pytest.mark.parametrize("flags", [*combinations(FLAGS, 2), *((flag, "--overwrite") for flag in FLAGS)])
def test_resume_flags_are_exclusive_and_reject_overwrite(
    monkeypatch: pytest.MonkeyPatch, flags: tuple[str, str],
) -> None:
    """Invalid mode combinations fail before any files or executors are touched."""
    monkeypatch.setattr("sys.argv", ["dl-sweep", "missing.yaml", *flags])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
