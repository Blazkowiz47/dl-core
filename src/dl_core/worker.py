#!/usr/bin/env python3
"""
Entry point for training workers.

This file is called by torchrun (for multi-GPU) or directly by executors.
It directly instantiates and runs the trainer without any orchestration logic.

For orchestration, use `dl-run` for single runs or `dl-sweep` for sweeps.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from dl_core import load_builtin_components, load_local_components
from dl_core.core import TRAINER_REGISTRY
from dl_core.core.base_trainer import BaseTrainer
from dl_core.utils.checkpoint_utils import (
    find_latest_checkpoint_local,
    get_checkpoint_dir_from_config,
)
from dl_core.utils.logging import setup_logging


def _configure_torch_sharing_strategy(logger: logging.Logger) -> None:
    """Prefer file-system sharing to reduce dataloader file descriptor pressure."""
    try:
        if "file_system" not in mp.get_all_sharing_strategies():
            return
        if mp.get_sharing_strategy() == "file_system":
            return
        mp.set_sharing_strategy("file_system")
        logger.info(
            "Using torch multiprocessing sharing strategy: file_system"
        )
    except Exception as exc:
        logger.warning(
            f"Failed to configure torch multiprocessing sharing strategy: {exc}"
        )


def main():
    """Main training function (worker mode)."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Lab - Training Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    load_builtin_components()
    load_local_components(args.config)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    logger_level = config.get("runtime", {}).get("logger_level") or args.log_level
    setup_logging(logger_level)
    logger = logging.getLogger(__name__)
    _configure_torch_sharing_strategy(logger)

    # Get trainer name from config
    # Config structure: trainer: { <name>: {...} }
    trainer_dict = config.get("trainer", {})
    if isinstance(trainer_dict, dict) and trainer_dict:
        trainer_name = list(trainer_dict.keys())[0]  # Get first trainer name
    else:
        trainer_name = "standard"  # Fallback default

    if config.get("auto_resume_local", False):
        # Local executors: use local checkpoint
        # Only rank 0 searches to avoid concurrent filesystem access
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        ckpt_path = None
        if is_main:
            logger.info("Auto-resume enabled, checking for local checkpoint...")
            checkpoint_dir = get_checkpoint_dir_from_config(config)
            if checkpoint_dir:
                ckpt_path = find_latest_checkpoint_local(checkpoint_dir)
                if ckpt_path:
                    logger.info(f"Found local checkpoint: {ckpt_path}")

        # Broadcast checkpoint path from rank 0 to all other ranks
        if dist.is_initialized():
            # Broadcast using object list
            ckpt_path_list = [ckpt_path] if is_main else [None]
            dist.broadcast_object_list(ckpt_path_list, src=0)
            ckpt_path = ckpt_path_list[0]

            # Barrier to ensure all ranks have the path
            dist.barrier()

        # All ranks set the checkpoint path in their config
        if ckpt_path:
            config["trainer"][trainer_name]["continue_model"] = ckpt_path
            logger.info(f"Auto-resuming from local checkpoint: {ckpt_path}")

    # Create and run trainer (run() calls setup() then train())
    trainer: BaseTrainer = TRAINER_REGISTRY.get(trainer_name, config)
    setup_logging(logger_level, trainer.artifact_manager.get_logs_dir() / "train.log")
    trainer.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
