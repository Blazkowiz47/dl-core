"""Tests for local checkpoint discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from dl_core.utils.checkpoint_utils import (
    find_checkpoint_candidates_local,
    find_latest_checkpoint_local,
)


def _save_checkpoint(path: Path, value: int) -> None:
    """Write a minimal loadable checkpoint."""

    torch.save({"value": value}, path)


def test_find_latest_checkpoint_prefers_latest_pth(tmp_path: Path) -> None:
    """`latest.pth` should take precedence when present."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _save_checkpoint(checkpoint_dir / "epoch_3.pth", 3)
    latest_path = checkpoint_dir / "latest.pth"
    _save_checkpoint(latest_path, 4)

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(latest_path)


def test_find_latest_checkpoint_supports_pth_epoch_files(tmp_path: Path) -> None:
    """Checkpoint discovery should find the newest `.pth` epoch file."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _save_checkpoint(checkpoint_dir / "epoch_2.pth", 2)
    latest_epoch = checkpoint_dir / "epoch_4.pth"
    _save_checkpoint(latest_epoch, 4)

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(latest_epoch)


def test_find_latest_checkpoint_supports_rl_step_and_episode_files(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    old_step = checkpoint_dir / "step_000000000200.pth"
    latest_step = checkpoint_dir / "step_000000000400.pth"
    latest_episode = checkpoint_dir / "episode_00000020.pth"
    _save_checkpoint(old_step, 200)
    _save_checkpoint(latest_step, 400)
    _save_checkpoint(latest_episode, 20)
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
        if checkpoint == invalid_checkpoint:
            checkpoint.write_text("checkpoint", encoding="utf-8")
        else:
            _save_checkpoint(checkpoint, 1)
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
    _save_checkpoint(stable_checkpoint, 2)
    _save_checkpoint(disappearing_checkpoint, 4)
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


def test_find_latest_checkpoint_falls_back_from_corrupt_latest(
    tmp_path: Path,
) -> None:
    """Auto-resume should use a numbered checkpoint when latest is truncated."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pth").write_bytes(b"truncated")
    numbered = checkpoint_dir / "epoch_7.pth"
    _save_checkpoint(numbered, 7)

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(numbered)


def test_find_latest_checkpoint_supports_epoch_and_iteration_directories(
    tmp_path: Path,
) -> None:
    """Default trainer checkpoint directories must participate in fallback."""

    run_dir = tmp_path / "runs" / "demo"
    checkpoint_dir = run_dir / "final" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.pth").write_bytes(b"truncated")
    epoch_checkpoint = run_dir / "epoch_4" / "checkpoint.pth"
    epoch_checkpoint.parent.mkdir()
    _save_checkpoint(epoch_checkpoint, 4)
    iteration_checkpoint = run_dir / "iteration_8" / "checkpoint.pth"
    iteration_checkpoint.parent.mkdir()
    _save_checkpoint(iteration_checkpoint, 8)
    os.utime(epoch_checkpoint, ns=(1_000_000_000, 1_000_000_000))
    os.utime(iteration_checkpoint, ns=(2_000_000_000, 2_000_000_000))

    assert find_checkpoint_candidates_local(str(checkpoint_dir)) == [
        str(checkpoint_dir / "latest.pth"),
        str(iteration_checkpoint),
        str(epoch_checkpoint),
    ]
    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(
        iteration_checkpoint
    )


def test_find_latest_checkpoint_uses_best_as_last_fallback(tmp_path: Path) -> None:
    """A readable best alias should recover a run with a corrupt latest alias."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pth").write_bytes(b"truncated")
    best_checkpoint = checkpoint_dir / "best.pth"
    _save_checkpoint(best_checkpoint, 3)

    assert find_latest_checkpoint_local(str(checkpoint_dir)) == str(best_checkpoint)


def test_find_latest_checkpoint_fails_when_all_files_are_corrupt(
    tmp_path: Path,
) -> None:
    """Auto-resume must not silently restart when checkpoint artifacts exist."""

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pth").write_bytes(b"truncated")
    (checkpoint_dir / "epoch_7.pth").write_bytes(b"also-truncated")

    with pytest.raises(RuntimeError, match="none can be loaded"):
        find_latest_checkpoint_local(str(checkpoint_dir))
