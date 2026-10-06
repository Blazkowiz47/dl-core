"""Controlled signal and terminal-read races without blocking the test runner."""

from __future__ import annotations

import os
import signal

import pytest

from dl_core.executors.local import LocalExecutor
from dl_core.executors.local_supervisor import LocalSupervisor


@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("read_error", [False, True])
def test_menu_read_is_nonblocking_and_pending_stop_wins_over_enter(
    monkeypatch: pytest.MonkeyPatch, blocking: bool, read_error: bool,
) -> None:
    """An interrupt during read must survive Enter, EAGAIN, and stdin restoration."""
    read_fd, write_fd = os.pipe()
    supervisor = LocalSupervisor(LocalExecutor({}, "demo", "sweep-1"), menu_enabled=True)
    supervisor.stdin_fd = read_fd
    supervisor.menu_open = True
    supervisor.interrupt_count = 1
    os.set_blocking(read_fd, blocking)
    os.write(write_fd, b"\n")

    def read(fd: int, size: int) -> bytes:
        assert fd == read_fd
        assert not os.get_blocking(fd), "read may block after terminal input is flushed"
        supervisor._on_signal(signal.SIGINT, None)
        if read_error:
            raise BlockingIOError("terminal input was flushed")
        return b"\n"

    monkeypatch.setattr("dl_core.executors.local_supervisor.os.read", read)
    try:
        supervisor._read_menu()
        assert supervisor.stop_all
        assert not supervisor.menu_open
        assert supervisor.interrupt_count == 2
        assert os.get_blocking(read_fd) is blocking
    finally:
        supervisor.selector.close()
        os.close(read_fd)
        os.close(write_fd)


def test_stop_all_request_survives_a_racing_counter_reset() -> None:
    """Once the second interrupt arrives, later counter resets cannot cancel it."""
    supervisor = LocalSupervisor(LocalExecutor({}, "demo", "sweep-1"), menu_enabled=True)
    supervisor.interrupt_count = 1
    supervisor.stdin_fd = 0
    try:
        supervisor._on_signal(signal.SIGINT, None)
        supervisor.interrupt_count = 0
        supervisor._handle_requests()
        assert supervisor.stop_all
    finally:
        supervisor.selector.close()
