"""Local checkpoint utilities for resuming training."""

import os
import re
import stat
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)
from dl_core.utils.artifact_manager import resolve_existing_run_artifact_dir

logger = getLogger(__name__)


def _is_loadable_checkpoint(checkpoint_path: Path) -> bool:
    """Return whether a local checkpoint can be deserialized."""

    try:
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        logger.warning(f"Ignoring unreadable checkpoint {checkpoint_path}: {error}")
        return False
    return True


def find_latest_checkpoint_local(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in a local directory.

    Args:
        checkpoint_dir: Path to local checkpoint directory

    Returns:
        Path to latest checkpoint file or None if no checkpoints found
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        logger.warning(f"Checkpoint path is not a directory: {checkpoint_dir}")
        return None

    latest_checkpoint = checkpoint_path / "latest.pth"
    if latest_checkpoint.is_file():
        if _is_loadable_checkpoint(latest_checkpoint):
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)

    checkpoint_pattern = re.compile(
        r"(epoch|episode|step)_(\d+)\.pth?"
    )
    checkpoint_candidates: dict[str, list[tuple[int, int, Path]]] = {}

    try:
        checkpoint_files = list(checkpoint_path.iterdir())
    except OSError as error:
        logger.warning(
            f"Failed to inspect checkpoint directory {checkpoint_dir}: {error}"
        )
        return None

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

    if not checkpoint_candidates:
        logger.info(f"No checkpoints found in {checkpoint_dir}")
        return None

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
        checkpoint_number, _modified_time, latest_path = checkpoint_candidates[
            checkpoint_type
        ].pop(0)
        if not checkpoint_candidates[checkpoint_type]:
            del checkpoint_candidates[checkpoint_type]
        if not _is_loadable_checkpoint(latest_path):
            continue
        logger.info(
            f"Found latest checkpoint: {checkpoint_type} "
            f"{checkpoint_number} at {latest_path}"
        )
        return str(latest_path)

    logger.info(f"No loadable checkpoints found in {checkpoint_dir}")
    return None


def get_checkpoint_dir_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Get checkpoint directory path from config.

    This follows the same pattern as the dataset-driven trainers, which use
    ArtifactManager to determine the checkpoint directory.

    Args:
        config: Configuration dictionary

    Returns:
        Checkpoint directory path or None
    """
    try:
        # Try to construct checkpoint dir path from config
        # This mimics what ArtifactManager and the dataset-driven trainers do

        # Get runtime configuration used by the trainer artifact manager
        runtime_config = config.get("runtime", {})
        output_dir = runtime_config.get("output_dir", "artifacts")

        config_path = config.get("_config_path")
        experiment_name = resolve_config_experiment_name(
            config,
            config_path=config_path,
        )
        sweep_file = config.get("sweep_file")
        if sweep_file:
            sweep_file = Path(sweep_file).name.replace(".yaml", "")

        run_name = resolve_config_run_name(config, config_path=config_path)

        checkpoint_dir = (
            resolve_existing_run_artifact_dir(
                run_name=run_name,
                output_dir=output_dir,
                experiment_name=experiment_name,
                sweep_name=sweep_file,
            )
            / "final"
            / "checkpoints"
        )

        if checkpoint_dir.exists():
            return str(checkpoint_dir)

        logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    except Exception as e:
        logger.warning(f"Failed to determine checkpoint directory from config: {e}")
        return None
