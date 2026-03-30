"""Local checkpoint utilities for resuming training."""

import os
import re
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)

logger = getLogger(__name__)


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
    if latest_checkpoint.exists():
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    # Find all epoch checkpoint files
    checkpoint_pattern = re.compile(r"epoch_(\d+)\.(?:pt|pth)")
    checkpoint_epochs = []

    for file_path in checkpoint_path.iterdir():
        if not file_path.is_file():
            continue
        match = checkpoint_pattern.match(file_path.name)
        if match:
            epoch = int(match.group(1))
            checkpoint_epochs.append((epoch, str(file_path)))

    if not checkpoint_epochs:
        logger.info(f"No checkpoints found in {checkpoint_dir}")
        return None

    # Return path of checkpoint with highest epoch
    latest_epoch, latest_path = max(checkpoint_epochs, key=lambda x: x[0])
    logger.info(f"Found latest checkpoint: epoch {latest_epoch} at {latest_path}")
    return latest_path


def get_checkpoint_dir_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Get checkpoint directory path from config.

    This follows the same pattern as BaseTrainer which uses ArtifactManager
    to determine the checkpoint directory.

    Args:
        config: Configuration dictionary

    Returns:
        Checkpoint directory path or None
    """
    try:
        # Try to construct checkpoint dir path from config
        # This mimics what ArtifactManager and BaseTrainer do

        # Get runtime configuration (matches BaseTrainer lines 106-112)
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

        # Construct checkpoint dir path (matches ArtifactManager structure)
        if sweep_file:
            checkpoint_dir = (
                f"{output_dir}/{experiment_name}/{sweep_file}/{run_name}/"
                "final/checkpoints"
            )
        else:
            checkpoint_dir = (
                f"{output_dir}/{experiment_name}/{run_name}/final/checkpoints"
            )

        if os.path.exists(checkpoint_dir):
            return checkpoint_dir
        else:
            logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return None

    except Exception as e:
        logger.warning(f"Failed to determine checkpoint directory from config: {e}")
        return None
