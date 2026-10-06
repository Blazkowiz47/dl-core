"""Own local run processes, terminal input, and bounded shutdown."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
import os
from pathlib import Path
import select
import selectors
import signal
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from .local import LocalExecutor


@dataclass
class LocalRun:
    """A child process and the metadata needed to finalize its run."""

    index: int
    config_path: Path
    process: subprocess.Popen[bytes]
    metadata: dict[str, Any]
    output_log: BinaryIO
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")(errors="replace")
    )
    stopped: bool = False
    stop_stage: int = 0
    stop_deadline: float = 0.0
    descendants: dict[int, psutil.Process] = field(default_factory=dict)


class LocalSupervisor:
    """Supervise owned processes while keeping Ctrl-C and menu input responsive."""

    POLL_SECONDS = 0.05
    INTERRUPT_GRACE_SECONDS = 10.0
    TERMINATE_GRACE_SECONDS = 3.0
    KILL_GRACE_SECONDS = 2.0

    def __init__(
        self,
        executor: LocalExecutor,
        *,
        menu_enabled: bool,
        claim_runs: bool = True,
        record_results: bool = True,
    ) -> None:
        self.executor = executor
        self.menu_enabled = menu_enabled
        self.claim_runs = claim_runs
        self.record_results = record_results
        self.active: dict[int, LocalRun] = {}
        self.results: dict[int, dict[str, Any]] = {}
        self.selector = selectors.DefaultSelector()
        self.interrupt_count = 0
        self.terminate_requested = False
        self.stop_all = False
        self.menu_open = False
        self.selection_pending = False
        self.menu_rows: list[int] = []
        self.menu_input = b""
        self.stdin_fd: int | None = None
        self.previous_handlers: dict[int, Any] = {}

    def run(
        self, descriptors: list[tuple[int, Path]], max_workers: int
    ) -> dict[int, dict[str, Any]]:
        """Launch at most max_workers runs, then return their final results."""
        next_run = 0
        try:
            self._install_handlers()
            while next_run < len(descriptors) or self.active or self.menu_open:
                self._handle_requests()
                self._advance_shutdown()
                self._collect_finished()
                if self.menu_open:
                    self._read_menu()
                if self.stop_all and not self.active:
                    break
                while (
                    not self.stop_all and not self.menu_open
                    and not any(run.stopped for run in self.active.values())
                    and len(self.active) < max_workers and next_run < len(descriptors)
                ):
                    # Check between launches so a signal cannot queue more work.
                    self._handle_requests()
                    if self.stop_all or self.menu_open:
                        break
                    index, config_path = descriptors[next_run]
                    next_run += 1
                    if self.claim_runs and not self.executor._claim_run_for_execution(index, config_path):
                        self.executor.skipped_runs.append(index)
                        continue
                    self._start_run(index, config_path)
                self._poll_output(self.POLL_SECONDS)
                if self.selection_pending and not any(run.stopped for run in self.active.values()):
                    if not self.stop_all and self.interrupt_count < 2:
                        self.interrupt_count = 0
                        self.selection_pending = False
        except BaseException:
            self.stop_all = True
            raise
        finally:
            try:
                self.menu_open = False
                self._shutdown_remaining()
            finally:
                try:
                    for run in list(self.active.values()):
                        try:
                            self._close_run(run)
                        except Exception:
                            self.executor.logger.exception(f"Could not close run {run.index + 1}")
                    self.selector.close()
                finally:
                    for signum, handler in self.previous_handlers.items():
                        signal.signal(signum, handler)
        if self.stop_all:
            raise KeyboardInterrupt("Local runs stopped by user")
        return self.results

    def _install_handlers(self) -> None:
        """Only the owning main thread installs temporary signal handlers."""
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._on_signal)
        if self.menu_enabled and sys.stdin.isatty() and sys.stdout.isatty():
            try:
                self.stdin_fd = sys.stdin.fileno()
            except (OSError, ValueError):
                pass

    def _on_signal(self, signum: int, _frame: Any) -> None:
        if signum == signal.SIGINT:
            self.interrupt_count += 1
        else:
            self.terminate_requested = True

    def _handle_requests(self) -> None:
        if self.terminate_requested or self.interrupt_count >= 2 or (
            self.interrupt_count and self.stdin_fd is None
        ):
            self.stop_all = True
            if self.menu_open:
                print("\nStopping all local runs...", flush=True)
                self.menu_open = False
            for run in self.active.values():
                self._request_stop(run)
        elif self.interrupt_count and not self.menu_open and not self.selection_pending and not any(
            run.stopped for run in self.active.values()
        ):
            self.menu_rows = [
                index for index, run in self.active.items() if run.process.poll() is None
            ]
            self.menu_input = b""
            self.menu_open = True
            print("\nRunning local runs:", flush=True)
            for row, index in enumerate(self.menu_rows, 1):
                run = self.active[index]
                print(f"  {row}  {run.metadata['tracking_run_name']}", flush=True)
            if not self.menu_rows:
                print("  No active runs. New launches are paused.", flush=True)
            print("Press Ctrl-C again to stop the whole local sweep.", flush=True)
            self._print_prompt()

    def _print_prompt(self) -> None:
        print("Runs to stop [e.g. 2,3; all; Enter to continue]: ", end="", flush=True)

    def _read_menu(self) -> None:
        # A timeout-based loop handles flags-only signals even when reads retry.
        if self.stdin_fd is None or not select.select([self.stdin_fd], [], [], 0)[0]:
            return
        data = os.read(self.stdin_fd, 4096)
        if not data:
            self.interrupt_count = 2
            self._handle_requests()
            return
        self.menu_input += data
        if b"\n" not in self.menu_input:
            return
        line, self.menu_input = self.menu_input.split(b"\n", 1)
        selection = line.decode("utf-8", errors="replace").strip().lower()
        if not selection:
            self.menu_open = False
            self.interrupt_count = 0
            print("Continuing local sweep.", flush=True)
            return
        if selection == "all":
            self.interrupt_count = 2
            self._handle_requests()
            return
        parts = [part.strip() for part in selection.split(",")]
        if not all(part.isascii() and part.isdecimal() for part in parts):
            print("Invalid selection. Enter displayed numbers separated by commas.", flush=True)
            self._print_prompt()
            return
        try:
            rows = {int(part) for part in parts}
        except ValueError:
            print("Invalid selection. No runs were stopped.", flush=True)
            self._print_prompt()
            return
        if not rows or any(row < 1 or row > len(self.menu_rows) for row in rows):
            print("Invalid selection. No runs were stopped.", flush=True)
            self._print_prompt()
            return
        self.menu_open = False
        self.selection_pending = True
        for row in sorted(rows):
            index = self.menu_rows[row - 1]
            run = self.active.get(index)
            if run is None or run.process.poll() is not None:
                print(f"Run {row} already finished; preserving its result.", flush=True)
            else:
                self._request_stop(run)

    def _start_run(self, index: int, config_path: Path) -> None:
        output_log: BinaryIO | None = None
        metadata: dict[str, Any] = {}
        try:
            command, metadata = self.executor._prepare_run(index, config_path)
            if self.executor.dry_run:
                self.results[index] = {**metadata, "success": True}
                if self.record_results:
                    self.executor._record_run_result(index, config_path, self.results[index])
                return
            self._handle_requests()
            while self.menu_open and not self.stop_all:
                self._advance_shutdown()
                self._collect_finished()
                self._read_menu()
                self._poll_output(self.POLL_SECONDS)
                self._handle_requests()
            if self.stop_all:
                if self.claim_runs:
                    self.executor._update_tracker(index, "pending", config_path)
                    self.executor.claimed_runs.remove(index)
                return
            log_dir = Path(metadata["artifact_dir"]) / "final" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            output_log = (log_dir / "sweep.log").open("ab", buffering=0)
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            run = LocalRun(index, config_path, process, metadata, output_log)
            # Register ownership before any pipe operation that might fail.
            self.active[index] = run
            assert process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            self.selector.register(process.stdout, selectors.EVENT_READ, run)
        except Exception as error:
            if index in self.active:
                raise
            if output_log is not None:
                output_log.close()
            result = {**metadata, "success": False, "error_message": str(error)}
            self.results[index] = result
            if self.record_results:
                self.executor._record_run_result(index, config_path, result)
            self.executor.logger.error(f"Run {index + 1} could not start: {error}")

    def _poll_output(self, timeout: float) -> None:
        for key, _ in self.selector.select(timeout):
            self._read_output(key.data)

    def _read_output(self, run: LocalRun) -> bool:
        pipe = run.process.stdout
        if pipe is None or pipe.closed:
            return False
        try:
            data = os.read(pipe.fileno(), 65536)
        except BlockingIOError:
            return False
        if not data:
            try:
                self.selector.unregister(pipe)
            except KeyError:
                pass
            return False
        run.output_log.write(data)
        text = run.decoder.decode(data)
        if not self.menu_open:
            sys.stdout.write(text)
            sys.stdout.flush()
        return True

    def _group_alive(self, run: LocalRun) -> bool:
        self._remember_descendants(run)
        for process in run.descendants.values():
            try:
                if process.is_running() and process.status() not in {
                    psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD,
                }:
                    return True
            except psutil.NoSuchProcess:
                continue
        try:
            os.killpg(run.process.pid, 0)
        except ProcessLookupError:
            return False
        # Exited descendants can remain zombies until their parent reaps them.
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if os.getpgid(process.pid) == run.process.pid and process.info["status"] not in {
                    psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD,
                }:
                    return True
            except ProcessLookupError:
                continue
        return False

    def _remember_descendants(self, run: LocalRun) -> None:
        try:
            for process in psutil.Process(run.process.pid).children(recursive=True):
                run.descendants[process.pid] = process
        except psutil.NoSuchProcess:
            pass

    def _send_signal(self, run: LocalRun, signum: int) -> bool:
        """Signal the owned group and descendants that opened separate sessions."""
        self._remember_descendants(run)
        sent = False
        groups = {run.process.pid}
        for process in run.descendants.values():
            try:
                if process.is_running():
                    groups.add(os.getpgid(process.pid))
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        for group in groups:
            try:
                os.killpg(group, signum)
                sent = True
            except ProcessLookupError:
                pass
        return sent

    def _request_stop(self, run: LocalRun, *, user_stop: bool = True) -> None:
        if run.stop_stage:
            return
        still_running = run.process.poll() is None
        signum = signal.SIGINT if user_stop and still_running else signal.SIGTERM
        if not self._send_signal(run, signum):
            return
        run.stopped = user_stop and still_running
        run.stop_stage = 1 if signum == signal.SIGINT else 2
        grace = self.INTERRUPT_GRACE_SECONDS if run.stop_stage == 1 else self.TERMINATE_GRACE_SECONDS
        run.stop_deadline = time.monotonic() + grace

    def _advance_shutdown(self) -> None:
        now = time.monotonic()
        for run in list(self.active.values()):
            if not run.stop_stage or now < run.stop_deadline:
                continue
            if run.stop_stage >= 3:
                if self._group_alive(run):
                    self._finish_run(run, unknown=True)
                continue
            signum = signal.SIGTERM if run.stop_stage == 1 else signal.SIGKILL
            self._send_signal(run, signum)
            run.stop_stage += 1
            grace = self.TERMINATE_GRACE_SECONDS if run.stop_stage == 2 else self.KILL_GRACE_SECONDS
            run.stop_deadline = now + grace

    def _collect_finished(self) -> None:
        for run in list(self.active.values()):
            self._remember_descendants(run)
            if run.process.poll() is None:
                continue
            if self._group_alive(run):
                # A finished launcher can leave descendants behind; reap those too.
                self._request_stop(run, user_stop=False)
                continue
            self._finish_run(run)

    def _finish_run(
        self, run: LocalRun, *, unknown: bool = False, drain_output: bool = True
    ) -> None:
        if drain_output:
            if unknown:
                self._read_output(run)
            else:
                while self._read_output(run):
                    pass
        result = self.executor._finish_run(
            run.metadata, run.process.poll(), stopped=run.stopped, unknown=unknown,
        )
        self.results[run.index] = result
        self.active.pop(run.index)
        self._close_run(run)
        if self.record_results:
            self.executor._record_run_result(run.index, run.config_path, result)
        if not self.menu_open:
            self.executor.logger.info(
                f"Run {run.index + 1}: {self.executor._classify_run_result(result)}"
            )

    def _close_run(self, run: LocalRun) -> None:
        try:
            if run.process.stdout is not None:
                try:
                    self.selector.unregister(run.process.stdout)
                except (KeyError, ValueError):
                    pass
                run.process.stdout.close()
        finally:
            run.output_log.close()
            if run.process.poll() is not None:
                run.process.wait()

    def _shutdown_remaining(self) -> None:
        if not self.active:
            return
        try:
            for run in self.active.values():
                self._request_stop(run)
            while self.active:
                self._advance_shutdown()
                self._poll_output(self.POLL_SECONDS)
                self._collect_finished()
        finally:
            # Even an output/tracker error must not abandon other owned children.
            for run in list(self.active.values()):
                try:
                    self._send_signal(run, signal.SIGKILL)
                    try:
                        run.process.wait(timeout=self.KILL_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    deadline = time.monotonic() + self.KILL_GRACE_SECONDS
                    while self._group_alive(run) and time.monotonic() < deadline:
                        time.sleep(self.POLL_SECONDS)
                    self._finish_run(
                        run, unknown=self._group_alive(run), drain_output=False,
                    )
                except Exception:
                    self.executor.logger.exception(f"Could not finalize run {run.index + 1}")
