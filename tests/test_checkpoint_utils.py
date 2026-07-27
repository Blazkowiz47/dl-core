"""Tests for local checkpoint discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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


def test_find_latest_checkpoint_supports_rl_step_and_episode_files(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    old_step = checkpoint_dir / "step_000000000200.pth"
    latest_step = checkpoint_dir / "step_000000000400.pth"
    latest_episode = checkpoint_dir / "episode_00000020.pth"
    old_step.write_text("step200", encoding="utf-8")
    latest_step.write_text("step400", encoding="utf-8")
    latest_episode.write_text("episode20", encoding="utf-8")
    os.utime(old_step, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest_episode, ns=(2_000_000_000, 2_000_000_000))
    os.utime(latest_step, ns=(3_000_000_000, 3_000_000_000))

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(
        latest_step
    )


def test_find_latest_checkpoint_selects_highest_number_per_kind_first(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    old_step = checkpoint_dir / "step_000000000200.pth"
    highest_step = checkpoint_dir / "step_000000000400.pth"
    latest_episode = checkpoint_dir / "episode_00000020.pt"
    invalid_checkpoint = checkpoint_dir / "step_999.pth.tmp"
    for checkpoint in (
        old_step,
        highest_step,
        latest_episode,
        invalid_checkpoint,
    ):
        checkpoint.write_text("checkpoint", encoding="utf-8")
    os.utime(highest_step, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest_episode, ns=(2_000_000_000, 2_000_000_000))
    os.utime(old_step, ns=(3_000_000_000, 3_000_000_000))

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(
        latest_episode
    )


def test_find_latest_checkpoint_ignores_a_disappearing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    stable_checkpoint = checkpoint_dir / "epoch_2.pth"
    disappearing_checkpoint = checkpoint_dir / "step_4.pth"
    stable_checkpoint.write_text("stable", encoding="utf-8")
    disappearing_checkpoint.write_text("disappearing", encoding="utf-8")
    original_stat = Path.stat

    def stat_with_disappearing_checkpoint(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        if path == disappearing_checkpoint:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_disappearing_checkpoint)

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(
        stable_checkpoint
    )
