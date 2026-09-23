#!/usr/bin/env python3
"""
Entry point for training workers.

This file is called by torchrun (for multi-GPU) or directly by executors.
It directly instantiates and runs the trainer without any orchestration logic.

For orchestration, use `dl-run` for single runs or `dl-sweep` for sweeps.
"""

import argparse
import logging
import os
import signal
import sys
import yaml
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from dl_core import load_builtin_components, load_local_components
from dl_core.core import EpochTrainer, IterationTrainer, RLTrainer, TRAINER_REGISTRY
from dl_core.utils.logging import setup_logging


_INTERRUPT_ENV_KEY = "DL_CORE_INTERRUPT_REASON"


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


def _install_signal_handlers(logger: logging.Logger) -> None:
    """Convert termination signals into graceful trainer interrupts."""

    def _handle_signal(signum: int, _frame: object | None) -> None:
        signal_name = signal.Signals(signum).name
        os.environ[_INTERRUPT_ENV_KEY] = signal_name
        logger.warning(
            "Received %s, requesting graceful trainer shutdown",
            signal_name,
        )
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            logger.warning(
                "Failed to destroy process group during interrupt: %s",
                exc,
            )
        raise KeyboardInterrupt(f"Received {signal_name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_signal)


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
    _install_signal_handlers(logger)

    # Get trainer name from config
    # Config structure: trainer: { <name>: {...} }
    trainer_dict = config.get("trainer", {})
    if isinstance(trainer_dict, dict) and trainer_dict:
        trainer_name = list(trainer_dict.keys())[0]  # Get first trainer name
    else:
        trainer_name = "standard"  # Fallback default

    # Create and run trainer (run() calls setup() then train())
    trainer: EpochTrainer | IterationTrainer | RLTrainer = TRAINER_REGISTRY.get(
        trainer_name,
        config,
    )
    setup_logging(logger_level, trainer.artifact_manager.get_logs_dir() / "train.log")
    try:
        trainer.run()
    except KeyboardInterrupt as exc:
        logger.warning(str(exc) or "Training interrupted")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
