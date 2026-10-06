"""Prove local Ctrl-C behavior using terminal signals and owned dummy jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Callable, Iterator

import psutil
import pytest

FIXTURE_SCRIPT = Path(__file__).with_name("local_sweep_fixture.py")


def process_alive(pid: int) -> bool:
    """Treat zombies as terminated while their parent finishes reaping."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class TerminalSweep:
    """An owning foreground process in a real pseudo-terminal."""

    def __init__(self, directory: Path, *, mode: str = "sweep", count: int = 4, workers: int = 3) -> None:
        import pty

        self.directory = directory
        self.output = ""
        self.exit_code: int | None = None
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execv(sys.executable, [
                sys.executable, str(FIXTURE_SCRIPT), mode, str(directory), str(count), str(workers),
            ])

    def poll(self) -> None:
        """Drain terminal output and observe process completion without blocking."""
        if select.select([self.fd], [], [], 0.05)[0]:
            try:
                self.output += os.read(self.fd, 65536).decode(errors="replace")
            except OSError:
                pass
        if self.exit_code is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.exit_code = os.waitstatus_to_exitcode(status)

    def wait_for(self, condition: Callable[[], bool], *, timeout: float = 10) -> None:
        """Wait for an observable condition with bounded terminal draining."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if condition():
                return
            if self.exit_code is not None:
                break
        raise AssertionError(f"Condition was not met (exit={self.exit_code}).\n{self.output}")

    def send(self, value: str) -> None:
        """Send terminal input, including Ctrl-C through the terminal driver."""
        os.write(self.fd, value.encode())

    def jobs(self) -> list[dict[str, int]]:
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob("started-*.json"))]

    def wait_started(self, count: int) -> None:
        self.wait_for(lambda: len(self.jobs()) == count and len(list(self.directory.glob("descendant-*.json"))) == count)
        # Allow the supervisor to discover descendants in separate sessions.
        self.poll()

    def statuses(self) -> dict[str, str]:
        data = json.loads((self.directory / "sweep" / "sweep_tracking.json").read_text())
        return {index: run["status"] for index, run in data["runs"].items()}

    def close(self) -> None:
        """Always terminate owned fixture processes, including failed-test paths."""
        pids = {pid for job in self.jobs() for pid in job.values()}
        try:
            pids.update(process.pid for process in psutil.Process(self.pid).children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        for pid in [*pids, self.pid]:
            if process_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.exit_code is None:
            _, status = os.waitpid(self.pid, 0)
            self.exit_code = os.waitstatus_to_exitcode(status)
        os.close(self.fd)


@pytest.fixture
def terminal_sweep(tmp_path: Path) -> Iterator[TerminalSweep]:
    """Create a local sweep with detached descendants and unconditional cleanup."""
    if os.name != "posix":
        pytest.skip("Pseudo-terminal signal delivery requires POSIX")
    (tmp_path / "detached-descendants").touch()
    sweep = TerminalSweep(tmp_path)
    try:
        sweep.wait_started(3)
        yield sweep
    finally:
        sweep.close()


def test_terminal_menu_preserves_jobs_and_stops_selected_rows(terminal_sweep: TerminalSweep) -> None:
    """First Ctrl-C, invalid input, Enter, and selection affect only chosen jobs."""
    sweep = terminal_sweep
    jobs = sweep.jobs()
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    assert all(process_alive(pid) for job in jobs for pid in job.values())
    assert sweep.statuses() == {"0": "running", "1": "running", "2": "running", "3": "pending"}

    logs = list((sweep.directory / "artifacts").rglob("sweep.log"))
    sizes = sum(path.stat().st_size for path in logs)
    sweep.wait_for(lambda: sum(path.stat().st_size for path in logs) > sizes)
    sweep.send("2,99\n")
    sweep.wait_for(lambda: "Invalid selection" in sweep.output)
    assert all(process_alive(pid) for job in jobs for pid in job.values())
    sweep.send("\n")
    sweep.wait_for(lambda: "Continuing local sweep." in sweep.output)
    sweep.send("\x03")
    sweep.wait_for(lambda: sweep.output.count("Runs to stop") == 3)
    assert all(process_alive(pid) for job in jobs for pid in job.values())
    sweep.send("2,3\n")
    sweep.wait_for(lambda: sweep.statuses()["1"] == sweep.statuses()["2"] == "stopped")
    assert process_alive(jobs[0]["pid"])
    assert all(not process_alive(pid) for job in jobs[1:] for pid in job.values())
    sweep.wait_started(4)
    (sweep.directory / "finish-0").touch()
    (sweep.directory / "finish-3").touch()
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 0
    assert sweep.statuses() == {"0": "completed", "1": "stopped", "2": "stopped", "3": "completed"}
    assert all(json.loads((sweep.directory / "handlers-restored.json").read_text()).values())
    assert (sweep.directory / "tracker-teardown").read_text() == "closed\n"


@pytest.mark.parametrize("stop_input", ["\x03", "\x04", "all\n"])
def test_terminal_stop_all_cleans_descendants_and_leaves_queue_pending(
    terminal_sweep: TerminalSweep, stop_input: str,
) -> None:
    """Second Ctrl-C, EOF, and all stop the command with exit code 130."""
    sweep = terminal_sweep
    jobs = sweep.jobs()
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    sweep.send(stop_input)
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 130
    assert sweep.statuses() == {"0": "stopped", "1": "stopped", "2": "stopped", "3": "pending"}
    assert all(not process_alive(pid) for job in jobs for pid in job.values())
    assert all(json.loads((sweep.directory / "handlers-restored.json").read_text()).values())
    assert (sweep.directory / "tracker-teardown").read_text() == "closed\n"


def test_completed_menu_row_keeps_its_result_and_number(terminal_sweep: TerminalSweep) -> None:
    """A completion during the prompt neither renumbers rows nor becomes stopped."""
    sweep = terminal_sweep
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    (sweep.directory / "finish-0").touch()
    sweep.wait_for(lambda: sweep.statuses()["0"] == "completed")
    assert not (sweep.directory / "started-3.json").exists()
    sweep.send("1,3\n")
    sweep.wait_for(lambda: sweep.statuses()["2"] == "stopped")
    assert sweep.statuses()["0"] == "completed"
    assert sweep.statuses()["1"] == "running"
    sweep.wait_started(4)
    (sweep.directory / "finish-1").touch()
    (sweep.directory / "finish-3").touch()
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 0


def test_output_error_still_cleans_all_processes_and_tracking(terminal_sweep: TerminalSweep) -> None:
    """An I/O exception cannot bypass process or tracker cleanup."""
    sweep = terminal_sweep
    jobs = sweep.jobs()
    (sweep.directory / "raise-output-error").touch()
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 1
    assert all(not process_alive(pid) for job in jobs for pid in job.values())
    assert all(json.loads((sweep.directory / "handlers-restored.json").read_text()).values())
    assert (sweep.directory / "tracker-teardown").read_text() == "closed\n"
    assert sweep.statuses()["3"] == "pending"


def test_second_ctrl_c_during_selective_shutdown_stops_everything(terminal_sweep: TerminalSweep) -> None:
    """Selecting runs keeps the second-Ctrl-C stop-all action armed during cleanup."""
    sweep = terminal_sweep
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    sweep.send("2\n")
    sweep.wait_for(lambda: (sweep.directory / "signal-1").exists())
    assert sweep.statuses()["1"] == "running"
    sweep.send("\x03")
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 130
    assert sweep.statuses() == {"0": "stopped", "1": "stopped", "2": "stopped", "3": "pending"}
    assert all(not process_alive(pid) for job in sweep.jobs() for pid in job.values())


def test_ctrl_c_flush_between_select_and_read_does_not_block(terminal_sweep: TerminalSweep) -> None:
    """A real terminal interrupt may flush the line that made select readable."""
    sweep = terminal_sweep
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    (sweep.directory / "pause-menu-read").touch()
    sweep.send("\n")
    sweep.wait_for(lambda: (sweep.directory / "menu-read-ready").exists())
    # The terminal driver flushes that Enter before the supervisor calls read.
    sweep.send("\x03")
    (sweep.directory / "continue-menu-read").touch()
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 130
    assert "Continuing local sweep." not in sweep.output
    assert sweep.statuses() == {"0": "stopped", "1": "stopped", "2": "stopped", "3": "pending"}
    assert all(not process_alive(pid) for job in sweep.jobs() for pid in job.values())
    assert all(json.loads((sweep.directory / "handlers-restored.json").read_text()).values())


def test_sequence_resets_after_selected_jobs_finish(terminal_sweep: TerminalSweep) -> None:
    """After selective cleanup, the next Ctrl-C opens another menu."""
    sweep = terminal_sweep
    sweep.send("\x03")
    sweep.wait_for(lambda: "Runs to stop" in sweep.output)
    sweep.send("2\n")
    sweep.wait_for(lambda: sweep.statuses()["1"] == "stopped")
    sweep.wait_started(4)
    sweep.send("\x03")
    sweep.wait_for(lambda: sweep.output.count("Runs to stop") == 2)
    assert process_alive(sweep.jobs()[0]["pid"])
    assert process_alive(sweep.jobs()[2]["pid"])
    assert process_alive(sweep.jobs()[3]["pid"])
    sweep.send("\x03")
    sweep.wait_for(lambda: sweep.exit_code is not None)
    assert sweep.exit_code == 130
    assert set(sweep.statuses().values()) == {"stopped"}


def test_stubborn_processes_are_killed_after_grace_periods(tmp_path: Path) -> None:
    """Ignoring graceful signals cannot leave jobs or detached descendants alive."""
    if os.name != "posix":
        pytest.skip("Pseudo-terminal signal delivery requires POSIX")
    (tmp_path / "ignore-signals").touch()
    (tmp_path / "detached-descendants").touch()
    sweep = TerminalSweep(tmp_path, count=2, workers=2)
    try:
        sweep.wait_started(2)
        sweep.send("\x03")
        sweep.wait_for(lambda: "Runs to stop" in sweep.output)
        sweep.send("\x03")
        sweep.wait_for(lambda: sweep.exit_code is not None)
        assert sweep.exit_code == 130
        assert set(sweep.statuses().values()) == {"stopped"}
        assert all(not process_alive(pid) for job in sweep.jobs() for pid in job.values())
    finally:
        sweep.close()


def test_single_run_ctrl_c_has_direct_cleanup(tmp_path: Path) -> None:
    """dl-run terminates its isolated job on the first Ctrl-C without a menu."""
    if os.name != "posix":
        pytest.skip("Pseudo-terminal signal delivery requires POSIX")
    (tmp_path / "detached-descendants").touch()
    sweep = TerminalSweep(tmp_path, mode="single", count=1, workers=1)
    try:
        sweep.wait_started(1)
        jobs = sweep.jobs()
        sweep.send("\x03")
        sweep.wait_for(lambda: sweep.exit_code is not None)
        assert sweep.exit_code == 130
        assert "Runs to stop" not in sweep.output
        assert all(not process_alive(pid) for job in jobs for pid in job.values())
        assert all(json.loads((tmp_path / "handlers-restored.json").read_text()).values())
    finally:
        sweep.close()


def test_noninteractive_ctrl_c_stops_all_owned_jobs(tmp_path: Path) -> None:
    """Piped execution never waits for terminal input after SIGINT."""
    (tmp_path / "detached-descendants").touch()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE_SCRIPT), "sweep", str(tmp_path), "3", "2"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    jobs: list[dict[str, int]] = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            jobs = [json.loads(path.read_text()) for path in tmp_path.glob("started-*.json")]
            if len(jobs) == 2 and len(list(tmp_path.glob("descendant-*.json"))) == 2:
                break
            assert process.poll() is None
            time.sleep(0.05)
        assert len(jobs) == 2
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 130
        assert b"Runs to stop" not in output
        assert all(not process_alive(pid) for job in jobs for pid in job.values())
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        for job in jobs:
            for pid in job.values():
                if process_alive(pid):
                    os.kill(pid, signal.SIGKILL)
