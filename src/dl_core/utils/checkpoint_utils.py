"""Local checkpoint utilities for resuming training."""

import os
import re
import stat
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

import torch

logger = getLogger(__name__)


def atomic_torch_save(payload: Any, checkpoint_path: str | Path) -> Path:
    """Atomically replace one torch checkpoint and remove stale temp files."""

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in destination.parent.glob(f".{destination.name}.*.tmp"):
        try:
            stale_path.unlink()
        except OSError:
            logger.warning(f"Could not remove stale checkpoint temp file {stale_path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def _is_loadable_checkpoint(checkpoint_path: Path) -> bool:
    """Return whether a local checkpoint can be deserialized."""

    try:
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as mmap_error:
        try:
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as error:
            logger.warning(
                f"Ignoring unreadable checkpoint {checkpoint_path}: {error} "
                f"(mmap probe: {mmap_error})"
            )
            return False
    except Exception as error:
        logger.warning(f"Ignoring unreadable checkpoint {checkpoint_path}: {error}")
        return False
    return True


def find_checkpoint_candidates_local(checkpoint_dir: str) -> list[str]:
    """Return local resume candidates in deterministic preference order."""

    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        logger.warning(f"Checkpoint path is not a directory: {checkpoint_dir}")
        return []

    ordered_candidates: list[Path] = []
    latest_checkpoint = checkpoint_path / "latest.pth"
    if latest_checkpoint.is_file():
        ordered_candidates.append(latest_checkpoint)

    checkpoint_pattern = re.compile(r"(epoch|iteration|episode|step)_(\d+)\.pth?")
    directory_pattern = re.compile(r"(epoch|iteration)_(\d+)")
    checkpoint_candidates: dict[str, list[tuple[int, int, Path]]] = {}

    try:
        checkpoint_files = list(checkpoint_path.iterdir())
    except OSError as error:
        logger.warning(
            f"Failed to inspect checkpoint directory {checkpoint_dir}: {error}"
        )
        return []

    for file_path in checkpoint_files:
        match = checkpoint_pattern.fullmatch(file_path.name)
        if match is None:
            continue
        try:
            file_status = file_path.stat()
        except OSError:
            continue
        if not stat.S_ISREG(file_status.st_mode):
            continue

        checkpoint_type = match.group(1)
        checkpoint_number = int(match.group(2))
        checkpoint_candidates.setdefault(checkpoint_type, []).append(
            (checkpoint_number, file_status.st_mtime_ns, file_path)
        )

    final_dir = checkpoint_path.parent
    run_dir = final_dir.parent if final_dir.name == "final" else None
    if run_dir is not None and run_dir.is_dir():
        try:
            progress_directories = list(run_dir.iterdir())
        except OSError as error:
            logger.warning(f"Failed to inspect run directory {run_dir}: {error}")
            progress_directories = []
        for progress_dir in progress_directories:
            match = directory_pattern.fullmatch(progress_dir.name)
            if match is None or not progress_dir.is_dir():
                continue
            progress_checkpoint = progress_dir / "checkpoint.pth"
            try:
                file_status = progress_checkpoint.stat()
            except OSError:
                continue
            if not stat.S_ISREG(file_status.st_mode):
                continue
            checkpoint_type = match.group(1)
            checkpoint_number = int(match.group(2))
            checkpoint_candidates.setdefault(checkpoint_type, []).append(
                (checkpoint_number, file_status.st_mtime_ns, progress_checkpoint)
            )

    for candidates in checkpoint_candidates.values():
        candidates.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2].name,
            ),
            reverse=True,
        )

    while checkpoint_candidates:
        checkpoint_type = max(
            checkpoint_candidates,
            key=lambda kind: (
                checkpoint_candidates[kind][0][1],
                checkpoint_candidates[kind][0][0],
                kind,
                checkpoint_candidates[kind][0][2].name,
            ),
        )
        _checkpoint_number, _modified_time, latest_path = checkpoint_candidates[
            checkpoint_type
        ].pop(0)
        if not checkpoint_candidates[checkpoint_type]:
            del checkpoint_candidates[checkpoint_type]
        ordered_candidates.append(latest_path)

    best_checkpoint = checkpoint_path / "best.pth"
    if best_checkpoint.is_file():
        ordered_candidates.append(best_checkpoint)

    return [str(candidate) for candidate in ordered_candidates]


def find_latest_checkpoint_local(checkpoint_dir: str) -> str | None:
    """Return the first loadable local checkpoint in resume order."""

    checkpoint_candidates = find_checkpoint_candidates_local(checkpoint_dir)
    for checkpoint_path in checkpoint_candidates:
        candidate = Path(checkpoint_path)
        if _is_loadable_checkpoint(candidate):
            logger.info(f"Found latest checkpoint: {candidate}")
            return checkpoint_path

    if checkpoint_candidates:
        raise RuntimeError(
            f"Checkpoint artifacts exist in {checkpoint_dir}, but none can be loaded"
        )
    logger.info(f"No checkpoints found in {checkpoint_dir}")
    return None
