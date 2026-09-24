"""Tests for direct training-worker configuration."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from dl_core import worker


def test_worker_uses_runtime_log_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct worker launches should honor the documented runtime key."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  log_level: WARNING\n", encoding="utf-8")
    levels: list[str] = []
    trainer = SimpleNamespace(
        artifact_manager=SimpleNamespace(get_logs_dir=lambda: tmp_path),
        run=lambda: None,
    )

    monkeypatch.setattr(sys, "argv", ["dl-train-worker", "-c", str(config_path)])
    monkeypatch.setattr(worker, "load_builtin_components", lambda: None)
    monkeypatch.setattr(worker, "load_local_components", lambda path: None)
    monkeypatch.setattr(worker, "setup_logging", lambda level, *args: levels.append(level))
    monkeypatch.setattr(worker, "_configure_torch_sharing_strategy", lambda logger: None)
    monkeypatch.setattr(worker, "_install_signal_handlers", lambda logger: None)
    monkeypatch.setattr(worker.TRAINER_REGISTRY, "get", lambda *args: trainer)

    assert worker.main() == 0
    assert levels == ["WARNING", "WARNING"]
