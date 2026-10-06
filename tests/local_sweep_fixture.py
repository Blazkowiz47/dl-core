"""Executable dummy jobs and CLI harness for real terminal-signal tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def write_marker(path: Path, data: dict[str, Any]) -> None:
    """Atomically publish fixture state so observers never read partial JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def run_job(directory: Path, index: int, *, descendant: bool) -> int:
    """Keep a dummy process alive until signalled or explicitly completed."""
    received = 0

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal received
        received = signum

    ignore_signals = (directory / "ignore-signals").exists()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN if ignore_signals else on_signal)
    if descendant:
        write_marker(directory / f"descendant-{index}.json", {"pid": os.getpid()})
        while not received:
            time.sleep(0.02)
        return 0

    child = subprocess.Popen(
        [sys.executable, __file__, "descendant", str(directory), str(index)],
        start_new_session=(directory / "detached-descendants").exists(),
    )
    write_marker(directory / f"started-{index}.json", {"pid": os.getpid(), "child": child.pid})
    try:
        while not received and not (directory / f"finish-{index}").exists():
            print(f"dummy output {index}", flush=True)
            time.sleep(0.05)
        if received:
            (directory / f"signal-{index}").write_text(str(received))
            # Leave time to observe the still-running claim during shutdown.
            time.sleep(0.15)
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=2)
        return 130 if received else 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


def run_cli(directory: Path, mode: str, count: int, workers: int) -> int:
    """Use the real orchestration code with Python dummy-job commands."""
    import yaml

    from dl_core.executors.local import LocalExecutor
    from dl_core.executors import local_supervisor
    from dl_core.executors.local_supervisor import LocalSupervisor
    from dl_core.sweep import runner
    from dl_core import single_run

    class FixtureExecutor(LocalExecutor):
        def build_command(
            self, config_path: str, run_config: dict | None = None
        ) -> list[str]:
            config = run_config or yaml.safe_load(Path(config_path).read_text())
            return [sys.executable, __file__, "job", str(directory), str(config["fixture_index"])]

    LocalSupervisor.INTERRUPT_GRACE_SECONDS = 0.6
    LocalSupervisor.TERMINATE_GRACE_SECONDS = 0.3
    LocalSupervisor.KILL_GRACE_SECONDS = 0.5
    original_read = LocalSupervisor._read_output

    def read_output(supervisor: LocalSupervisor, run: object) -> bool:
        if (directory / "raise-output-error").exists():
            raise OSError("Dummy output reader failed")
        return original_read(supervisor, run)

    LocalSupervisor._read_output = read_output
    original_select = local_supervisor.select.select

    def select_menu(*args: Any) -> Any:
        ready = original_select(*args)
        if ready[0] and (directory / "pause-menu-read").exists():
            (directory / "menu-read-ready").touch()
            deadline = time.monotonic() + 5
            while not (directory / "continue-menu-read").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Menu read was not released")
                time.sleep(0.005)
            (directory / "pause-menu-read").unlink()
        return ready

    local_supervisor.select.select = select_menu
    original_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def make_executor(name: str, *args: object, **kwargs: object) -> FixtureExecutor:
        assert name == "local"
        executor = FixtureExecutor(*args, **kwargs)
        original_teardown = executor.run_tracker.teardown_sweep

        def teardown_tracker() -> None:
            with (directory / "tracker-teardown").open("a") as handle:
                handle.write("closed\n")
            original_teardown()

        executor.run_tracker.teardown_sweep = teardown_tracker
        return executor

    base_path = directory / "base.yaml"
    base = {
        "executor": {"name": "local"},
        "runtime": {"output_dir": str(directory / "artifacts")},
        "trainer": {"fixture": {}}, "fixture_index": 0,
    }
    base_path.write_text(yaml.safe_dump(base))
    try:
        if mode == "single":
            single_run.load_builtin_components = lambda: None
            single_run.load_local_components = lambda path: None
            single_run.validate_config = lambda *args, **kwargs: True
            single_run.EXECUTOR_REGISTRY.get = make_executor
            sys.argv = ["dl-run", "-c", str(base_path)]
            return single_run.main()
        sweep_path = directory / "sweep.yaml"
        sweep_path.write_text(yaml.safe_dump({
            "base_config": str(base_path), "grid": {"fixture_index": list(range(count))},
            "seeds": [7], "tracking": {"run_name_template": "job_{fixture_index}"},
        }))
        runner.load_builtin_components = lambda: None
        runner.load_local_components = lambda path: None
        runner.EXECUTOR_REGISTRY.get = make_executor
        sys.argv = ["dl-sweep", str(sweep_path), "--max-workers", str(workers)]
        return runner.main()
    finally:
        write_marker(directory / "handlers-restored.json", {
            str(signum): signal.getsignal(signum) == previous
            for signum, previous in original_handlers.items()
        })


if __name__ == "__main__":
    role, root = sys.argv[1:3]
    directory = Path(root)
    if role in {"job", "descendant"}:
        sys.exit(run_job(directory, int(sys.argv[3]), descendant=role == "descendant"))
    sys.exit(run_cli(directory, role, int(sys.argv[3]), int(sys.argv[4])))
