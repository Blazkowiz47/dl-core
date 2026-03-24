"""
Unified artifact management system.

This module provides standardized artifact layout and management for training runs,
ensuring consistent output structure across all experiments.
"""

import json
import shutil
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import yaml


class ArtifactManager:
    """
    Manages artifact creation and organization for training runs.

    Creates standardized directory structure and provides utilities for
    saving metrics, plots, checkpoints, and other artifacts.

    Directory Structure:
        output_dir/experiment_name/sweep_file/run_name/
        ├── config.yaml                # Saved run config
        ├── logs/                      # Training logs
        ├── checkpoints/               # Model weights
        ├── metrics/                   # Metrics JSON files
        ├── plots/                     # All visualizations
        │   ├── training/              # Training plots
        │   ├── evaluation/            # Evaluation plots
        │   └── misc/                  # Other plots
        ├── raw/                       # Raw predictions
        └── eval/                      # Evaluation reports
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str = "artifacts",
        experiment_name: str | None = None,
        sweep_name: str | None = None,
    ):
        """
        Initialize artifact manager.

        Args:
            run_name: Name for this training run (from config runtime.name)
            output_dir: Base output directory (default: "artifacts")
            experiment_name: Experiment name for directory structure
            sweep_name: Sweep name for directory structure
        """
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.sweep_name = sweep_name

        # Build directory structure: output_dir/experiment_name/run_name/
        if experiment_name and sweep_name:
            self.run_dir = self.output_dir / experiment_name / sweep_name / run_name
        elif experiment_name:
            self.run_dir = self.output_dir / experiment_name / run_name
        else:
            self.run_dir = self.output_dir / run_name

        self.logger = getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create artifact directory structure
        self.create_artifact_tree()

    def get_run_artifact_dir(self) -> str:
        """Get the artifact directory for this run."""
        return str(self.run_dir)

    def create_artifact_tree(self) -> None:
        """
        Create standardized artifact directory structure.

        Creates the following structure:
        output_dir/experiment_name/run_name/
        ├── config.yaml                # saved configuration
        ├── logs/                      # training and evaluation logs
        ├── checkpoints/               # model checkpoints
        ├── metrics/                   # evaluation metrics and scores
        ├── plots/                     # all visualizations
        │   ├── training/              # training-related plots
        │   ├── evaluation/            # evaluation-related plots
        │   └── misc/                  # other visualizations
        ├── raw/                       # raw prediction scores
        └── eval/                      # evaluation artifacts
        """
        # Create main directories
        directories = [
            self.run_dir / "logs",
            self.run_dir / "checkpoints",
            self.run_dir / "metrics",
            self.run_dir / "raw",
            self.run_dir / "eval",
        ]

        # Create plots subdirectories
        plots_dir = self.run_dir / "plots"
        plot_subdirs = [
            plots_dir / "training",
            plots_dir / "evaluation",
            plots_dir / "misc",
        ]

        # Create all directories
        for directory in directories + plot_subdirs:
            directory.mkdir(parents=True, exist_ok=True)

        # Create latest symlink at experiment level (with race condition handling for multi-GPU)
        if self.experiment_name and self.sweep_name:
            latest_link = (
                self.output_dir / self.experiment_name / self.sweep_name / "latest"
            )
        elif self.experiment_name:
            latest_link = self.output_dir / self.experiment_name / "latest"
        else:
            latest_link = self.output_dir / "latest"

        try:
            # Try to remove existing symlink/file if it exists
            try:
                latest_link.unlink()
            except FileNotFoundError:
                # Symlink doesn't exist, this is fine
                pass

            # Create new symlink pointing to run_name
            latest_link.symlink_to(self.run_name)
        except FileExistsError:
            # Another process created the symlink first (multi-GPU race condition)
            # This is fine, just log and continue
            self.logger.debug(
                "Symlink already created by another process (multi-GPU training)"
            )

        self.logger.debug(f"Created artifact tree at {self.run_dir}")

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save run configuration.

        Args:
            config: Configuration dictionary to save
        """
        config_path = self.run_dir / "config.yaml"

        # Add metadata
        config_with_metadata = {
            "run_metadata": {
                "run_name": self.run_name,
                "experiment_name": self.experiment_name,
                "sweep_name": self.sweep_name,
                "created_at": datetime.now().isoformat(),
                "artifact_dir": str(self.run_dir),
            },
            **config,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False, indent=2)

        self.logger.info(f"Saved config to {config_path}")

    def save_metrics(
        self, metrics: Dict[str, Any], filename: str = "metrics.json"
    ) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        metrics_path = self.run_dir / "metrics" / filename

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.debug(f"Saved metrics to {metrics_path}")

    def save_raw_scores(
        self, scores: Dict[str, Any], filename: str = "raw_scores.json"
    ) -> None:
        """
        Save raw prediction scores.

        Args:
            scores: Raw scores dictionary
            filename: Output filename
        """
        raw_path = self.run_dir / "raw" / filename

        with open(raw_path, "w") as f:
            json.dump(scores, f, indent=2)

        self.logger.debug(f"Saved raw scores to {raw_path}")

    def save_plot(self, plot_path: str, category: str = "") -> str:
        """
        Save plot to appropriate directory.

        Args:
            plot_path: Source path of the plot
            category: Optional category subdirectory ('training', 'evaluation', 'misc')

        Returns:
            Final path where plot was saved
        """
        plot_filename = Path(plot_path).name

        if category:
            dest_dir = self.run_dir / "plots" / category
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / plot_filename
        else:
            dest_path = self.run_dir / "plots" / plot_filename

        shutil.copy2(plot_path, dest_path)

        self.logger.debug(f"Saved plot to {dest_path}")

        return str(dest_path)

    def get_eval_dir(self) -> Path:
        """Get the evaluation artifacts directory."""
        return self.run_dir / "eval"

    def get_metrics_dir(self) -> Path:
        """Get the metrics directory."""
        return self.run_dir / "metrics"

    def get_plots_dir(self) -> Path:
        """Get the plots directory."""
        return self.run_dir / "plots"

    def get_training_plots_dir(self) -> Path:
        """Get the training plots directory."""
        return self.run_dir / "plots" / "training"

    def get_evaluation_plots_dir(self) -> Path:
        """Get the evaluation plots directory."""
        return self.run_dir / "plots" / "evaluation"

    def get_misc_plots_dir(self) -> Path:
        """Get the miscellaneous plots directory."""
        return self.run_dir / "plots" / "misc"

    def get_raw_dir(self) -> Path:
        """Get the raw data directory."""
        return self.run_dir / "raw"

    def get_checkpoints_dir(self) -> Path:
        """Get the checkpoints directory."""
        return self.run_dir / "checkpoints"

    def get_logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.run_dir / "logs"

    def write_eval_summary(self, summary: Dict[str, Any]) -> None:
        """
        Write evaluation summary to eval directory.

        Args:
            summary: Evaluation summary data
        """
        eval_dir = self.get_eval_dir()
        summary_path = eval_dir / "summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Saved evaluation summary to {summary_path}")

    def list_artifacts(self) -> Dict[str, list]:
        """
        List all artifacts in this run.

        Returns:
            Dictionary mapping artifact type to list of files
        """
        artifacts = {}

        for subdir in ["logs", "checkpoints", "metrics", "plots", "raw", "eval"]:
            subdir_path = self.run_dir / subdir
            if subdir_path.exists():
                artifacts[subdir] = [
                    f.name for f in subdir_path.iterdir() if f.is_file()
                ]
            else:
                artifacts[subdir] = []

        return artifacts

    def cleanup_old_artifacts(self, keep_latest: int = 10) -> None:
        """
        Clean up old artifact directories, keeping only the latest N runs.

        Args:
            keep_latest: Number of latest runs to keep
        """
        # Determine the directory to clean (experiment dir or output dir)
        if self.experiment_name and self.sweep_name:
            cleanup_dir = self.output_dir / self.experiment_name / self.sweep_name
        elif self.experiment_name:
            cleanup_dir = self.output_dir / self.experiment_name
        else:
            cleanup_dir = self.output_dir

        if not cleanup_dir.exists():
            return

        # Get all run directories sorted by modification time
        run_dirs = [
            d for d in cleanup_dir.iterdir() if d.is_dir() and d.name != "latest"
        ]
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old directories
        for old_dir in run_dirs[keep_latest:]:
            self.logger.info(f"Cleaning up old artifacts: {old_dir}")
            shutil.rmtree(old_dir)


def get_run_artifact_dir(
    run_name: str,
    output_dir: str = "artifacts",
    experiment_name: str | None = None,
    sweep_name: str | None = None,
) -> str:
    """
    Get artifact directory path for a run.

    Args:
        run_name: Run name
        output_dir: Base output directory
        experiment_name: Optional experiment name
        sweep_name: Optional sweep name

    Returns:
        Full path to run artifact directory
    """
    if experiment_name and sweep_name:
        return str(Path(output_dir) / experiment_name / sweep_name / run_name)
    elif experiment_name:
        return str(Path(output_dir) / experiment_name / run_name)
    return str(Path(output_dir) / run_name)


def create_artifact_tree(
    run_name: str,
    output_dir: str = "artifacts",
    experiment_name: str | None = None,
    sweep_name: str | None = None,
) -> ArtifactManager:
    """
    Create artifact directory tree for a run.

    Args:
        run_name: Run name
        output_dir: Base output directory
        experiment_name: Optional experiment name
        sweep_name: Optional sweep name

    Returns:
        ArtifactManager instance for the run
    """
    return ArtifactManager(run_name, output_dir, experiment_name, sweep_name)
