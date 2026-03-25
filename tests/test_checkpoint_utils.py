"""Tests for local checkpoint discovery."""

from __future__ import annotations

from pathlib import Path

from dl_core.utils.checkpoint_utils import find_latest_checkpoint_local


def test_find_latest_checkpoint_prefers_latest_pth(tmp_path: Path) -> None:
    """`latest.pth` should take precedence when present."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_3.pth").write_text("epoch3", encoding="utf-8")
    latest_path = checkpoint_dir / "latest.pth"
    latest_path.write_text("latest", encoding="utf-8")

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(latest_path)


def test_find_latest_checkpoint_supports_pth_epoch_files(tmp_path: Path) -> None:
    """Checkpoint discovery should find the newest `.pth` epoch file."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_2.pth").write_text("epoch2", encoding="utf-8")
    latest_epoch = checkpoint_dir / "epoch_4.pth"
    latest_epoch.write_text("epoch4", encoding="utf-8")

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(latest_epoch)
