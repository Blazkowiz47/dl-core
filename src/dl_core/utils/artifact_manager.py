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
from typing import Any

import yaml


class ArtifactManager:
    """
    Manages artifact creation and organization for training runs.

    Creates standardized directory structure and provides utilities for
    saving metrics, plots, checkpoints, and other artifacts. Run directories are
    addressed by their concrete names; the manager does not create `latest`
    symlinks.

    Directory Structure:
        output_dir/runs/run_name/
        ├── config.yaml                # Saved run config
        ├── epoch_1/                   # Epoch-scoped artifacts
        │   ├── checkpoint.pth
        │   ├── metrics/
        │   ├── plots/
        │   ├── raw/
        │   └── eval/
        ├── epoch_2/
        │   └── ...
        └── final/                     # Final aliases and run summaries
            ├── checkpoints/
            ├── logs/
            ├── metrics/
            ├── plots/
            ├── raw/
            ├── eval/
            └── tracking/

        Sweep runs are grouped under:
        output_dir/sweeps/sweep_name/run_name/
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
            run_name: Name for this training run
            output_dir: Base output directory (default: "artifacts")
            experiment_name: Experiment name to record in metadata
            sweep_name: Sweep name used for grouped run directories
        """
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.sweep_name = sweep_name

        self.run_dir = Path(
            get_run_artifact_dir(
                run_name=run_name,
                output_dir=str(self.output_dir),
                experiment_name=experiment_name,
                sweep_name=sweep_name,
            )
        )

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
        output_dir/runs/run_name/
        ├── config.yaml                # saved configuration
        ├── epoch_<n>/                 # epoch-specific artifacts
        └── final/                     # final artifacts and aliases
            ├── checkpoints/           # saved checkpoints and final aliases
            ├── logs/                  # training and evaluation logs
            ├── metrics/               # evaluation metrics and scores
            ├── plots/                 # final visualizations or reports
            │   ├── training/
            │   ├── evaluation/
            │   └── misc/
            ├── raw/                   # final raw prediction exports
            ├── eval/                  # evaluation artifacts
            └── tracking/              # tracker session metadata
        """
        final_dir = self.get_final_dir()
        directories = [
            final_dir / "logs",
            final_dir / "checkpoints",
            final_dir / "metrics",
            final_dir / "metrics" / "series",
            final_dir / "raw",
            final_dir / "eval",
            final_dir / "tracking",
        ]

        # Create final plots subdirectories
        plots_dir = final_dir / "plots"
        plot_subdirs = [
            plots_dir / "training",
            plots_dir / "evaluation",
            plots_dir / "misc",
        ]

        # Create all directories
        for directory in directories + plot_subdirs:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Created artifact tree at {self.run_dir}")

    def get_epoch_dir(self, epoch: int) -> Path:
        """Get the root directory for one epoch's artifacts."""
        return self.run_dir / f"epoch_{epoch}"

    def get_epoch_metrics_dir(self, epoch: int) -> Path:
        """Get the metrics directory for one epoch's artifacts."""
        return self.get_epoch_dir(epoch) / "metrics"

    def get_epoch_plots_dir(self, epoch: int) -> Path:
        """Get the plots directory for one epoch's artifacts."""
        return self.get_epoch_dir(epoch) / "plots"

    def get_epoch_training_plots_dir(self, epoch: int) -> Path:
        """Get the training plots directory for one epoch's artifacts."""
        return self.get_epoch_plots_dir(epoch) / "training"

    def get_epoch_evaluation_plots_dir(self, epoch: int) -> Path:
        """Get the evaluation plots directory for one epoch's artifacts."""
        return self.get_epoch_plots_dir(epoch) / "evaluation"

    def get_epoch_misc_plots_dir(self, epoch: int) -> Path:
        """Get the miscellaneous plots directory for one epoch's artifacts."""
        return self.get_epoch_plots_dir(epoch) / "misc"

    def get_epoch_raw_dir(self, epoch: int) -> Path:
        """Get the raw-data directory for one epoch's artifacts."""
        return self.get_epoch_dir(epoch) / "raw"

    def get_epoch_eval_dir(self, epoch: int) -> Path:
        """Get the evaluation directory for one epoch's artifacts."""
        return self.get_epoch_dir(epoch) / "eval"

    def get_final_dir(self) -> Path:
        """Get the root directory for final run artifacts."""
        return self.run_dir / "final"

    def save_config(self, config: dict[str, Any]) -> None:
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

    def _write_json(self, path: Path, data: dict[str, Any]) -> Path:
        """Write JSON data to one resolved artifact path."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.debug(f"Saved JSON artifact to {path}")
        return path

    def save_json(self, relative_path: str | Path, data: dict[str, Any]) -> Path:
        """
        Save JSON data relative to the run directory.

        Args:
            relative_path: Relative path inside the run directory
            data: JSON-serializable dictionary

        Returns:
            Path to the written JSON file
        """
        return self._write_json(self.run_dir / relative_path, data)

    def save_final_json(self, relative_path: str | Path, data: dict[str, Any]) -> Path:
        """
        Save JSON data relative to the final artifact directory.

        Args:
            relative_path: Relative path inside ``final/``
            data: JSON-serializable dictionary

        Returns:
            Path to the written JSON file
        """
        return self._write_json(self.get_final_dir() / relative_path, data)

    def save_epoch_json(
        self,
        epoch: int,
        relative_path: str | Path,
        data: dict[str, Any],
    ) -> Path:
        """
        Save JSON data relative to one epoch artifact directory.

        Args:
            epoch: Epoch number
            relative_path: Relative path inside ``epoch_<n>/``
            data: JSON-serializable dictionary

        Returns:
            Path to the written JSON file
        """
        return self._write_json(self.get_epoch_dir(epoch) / relative_path, data)

    def append_jsonl(self, relative_path: str | Path, data: dict[str, Any]) -> Path:
        """
        Append one JSON record to a JSONL artifact.

        Args:
            relative_path: Relative path inside the run directory
            data: JSON-serializable dictionary

        Returns:
            Path to the JSONL file
        """
        file_path = self.run_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")

        self.logger.debug(f"Appended JSONL artifact to {file_path}")
        return file_path

    def append_final_jsonl(
        self,
        relative_path: str | Path,
        data: dict[str, Any],
    ) -> Path:
        """
        Append one JSON record to a JSONL artifact under ``final/``.

        Args:
            relative_path: Relative path inside ``final/``
            data: JSON-serializable dictionary

        Returns:
            Path to the JSONL file
        """
        return self.append_jsonl(Path("final") / relative_path, data)

    def save_metrics(
        self, metrics: dict[str, Any], filename: str = "metrics.json"
    ) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        self.save_final_json(Path("metrics") / filename, metrics)

    def save_epoch_metrics(
        self,
        epoch: int,
        metrics: dict[str, Any],
        filename: str = "metrics.json",
    ) -> Path:
        """
        Save metrics for one epoch under its artifact directory.

        Args:
            epoch: Epoch number
            metrics: Metrics dictionary
            filename: Output filename

        Returns:
            Path to the written metrics file
        """
        return self.save_epoch_json(epoch, Path("metrics") / filename, metrics)

    def save_raw_scores(
        self, scores: dict[str, Any], filename: str = "raw_scores.json"
    ) -> None:
        """
        Save raw prediction scores.

        Args:
            scores: Raw scores dictionary
            filename: Output filename
        """
        self.save_final_json(Path("raw") / filename, scores)

    def save_plot(
        self,
        plot_path: str,
        category: str = "",
        *,
        epoch: int | None = None,
    ) -> str:
        """
        Save plot to appropriate directory.

        Args:
            plot_path: Source path of the plot
            category: Optional category subdirectory ('training', 'evaluation', 'misc')
            epoch: Optional epoch number. When provided, the plot is copied
                under ``epoch_<n>/plots/...``. Otherwise it is copied under
                ``final/plots/...``.

        Returns:
            Final path where plot was saved
        """
        plot_filename = Path(plot_path).name

        if epoch is not None:
            plots_root = self.get_epoch_plots_dir(epoch)
        else:
            plots_root = self.get_plots_dir()

        if category:
            dest_dir = plots_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / plot_filename
        else:
            plots_root.mkdir(parents=True, exist_ok=True)
            dest_path = plots_root / plot_filename

        shutil.copy2(plot_path, dest_path)

        self.logger.debug(f"Saved plot to {dest_path}")

        return str(dest_path)

    def get_eval_dir(self) -> Path:
        """Get the evaluation artifacts directory."""
        return self.get_final_dir() / "eval"

    def get_metrics_dir(self) -> Path:
        """Get the metrics directory."""
        return self.get_final_dir() / "metrics"

    def get_plots_dir(self) -> Path:
        """Get the plots directory."""
        return self.get_final_dir() / "plots"

    def get_training_plots_dir(self) -> Path:
        """Get the training plots directory."""
        return self.get_plots_dir() / "training"

    def get_evaluation_plots_dir(self) -> Path:
        """Get the evaluation plots directory."""
        return self.get_plots_dir() / "evaluation"

    def get_misc_plots_dir(self) -> Path:
        """Get the miscellaneous plots directory."""
        return self.get_plots_dir() / "misc"

    def get_raw_dir(self) -> Path:
        """Get the raw data directory."""
        return self.get_final_dir() / "raw"

    def get_checkpoints_dir(self) -> Path:
        """Get the final checkpoint directory."""
        return self.get_final_dir() / "checkpoints"

    def get_logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.get_final_dir() / "logs"

    def get_run_info_path(self) -> Path:
        """Get the run metadata JSON path."""
        return self.get_final_dir() / "run_info.json"

    def get_metrics_summary_path(self) -> Path:
        """Get the metrics summary JSON path."""
        return self.get_metrics_dir() / "summary.json"

    def get_metrics_history_path(self) -> Path:
        """Get the metrics history JSON path."""
        return self.get_metrics_dir() / "history.json"

    def get_metric_streams_dir(self) -> Path:
        """Get the directory for per-metric JSONL streams."""
        return self.get_metrics_dir() / "series"

    def get_tracking_dir(self) -> Path:
        """Get the tracking metadata directory."""
        return self.get_final_dir() / "tracking"

    def get_tracking_session_path(self) -> Path:
        """Get the tracker session metadata JSON path."""
        return self.get_tracking_dir() / "session.json"

    def get_final_checkpoint_path(self, filename: str) -> Path:
        """Get one final checkpoint alias path under ``final/checkpoints``."""
        return self.get_checkpoints_dir() / filename

    def get_epoch_checkpoint_path(
        self,
        epoch: int,
        filename: str | None = None,
    ) -> Path:
        """Get one epoch checkpoint path under ``epoch_<n>/``."""
        checkpoint_name = filename or "checkpoint.pth"
        return self.get_epoch_dir(epoch) / checkpoint_name

    def save_run_info(self, run_info: dict[str, Any]) -> None:
        """
        Save run metadata to the artifact directory.

        Args:
            run_info: Run metadata dictionary
        """
        self.save_final_json("run_info.json", run_info)

    def save_tracking_session(self, session_data: dict[str, Any]) -> None:
        """
        Save tracker session metadata to the artifact directory.

        Args:
            session_data: Tracker-owned session metadata
        """
        self._write_json(self.get_tracking_session_path(), session_data)

    def write_eval_summary(self, summary: dict[str, Any]) -> None:
        """
        Write evaluation summary to eval directory.

        Args:
            summary: Evaluation summary data
        """
        summary_path = self.save_final_json(Path("eval") / "summary.json", summary)
        self.logger.info(f"Saved evaluation summary to {summary_path}")

    def list_artifacts(self) -> dict[str, list[str]]:
        """
        List all artifacts in this run.

        Returns:
            Dictionary mapping artifact type to list of files
        """
        artifacts: dict[str, list[str]] = {
            "epochs": sorted(
                path.name
                for path in self.run_dir.iterdir()
                if path.is_dir() and path.name.startswith("epoch_")
            ),
            "final": [],
        }

        final_dir = self.get_final_dir()
        if final_dir.exists():
            artifacts["final"] = sorted(path.name for path in final_dir.iterdir())

        return artifacts

    def cleanup_old_artifacts(self, keep_latest: int = 10) -> None:
        """
        Clean up old artifact directories, keeping only the latest N runs.

        Args:
            keep_latest: Number of latest runs to keep
        """
        if self.sweep_name:
            cleanup_dir = self.output_dir / "sweeps" / self.sweep_name
        else:
            cleanup_dir = self.output_dir / "runs"

        if not cleanup_dir.exists():
            return

        # Get all run directories sorted by modification time
        run_dirs = [d for d in cleanup_dir.iterdir() if d.is_dir()]
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
        experiment_name: Optional experiment name recorded as metadata only
        sweep_name: Optional sweep name used for grouped run directories

    Returns:
        Full path to run artifact directory
    """
    output_root = Path(output_dir)
    if sweep_name:
        return str(output_root / "sweeps" / sweep_name / run_name)
    return str(output_root / "runs" / run_name)


def get_legacy_run_artifact_dir(
    run_name: str,
    output_dir: str = "artifacts",
    experiment_name: str | None = None,
    sweep_name: str | None = None,
) -> str:
    """
    Get the legacy artifact directory path for a run.

    This preserves compatibility with runs created before the flattened
    ``runs/`` and ``sweeps/`` layout was introduced.
    """
    output_root = Path(output_dir)
    if experiment_name and sweep_name:
        return str(output_root / experiment_name / sweep_name / run_name)
    if experiment_name:
        return str(output_root / experiment_name / run_name)
    return str(output_root / run_name)


def resolve_existing_run_artifact_dir(
    run_name: str,
    output_dir: str = "artifacts",
    experiment_name: str | None = None,
    sweep_name: str | None = None,
) -> Path:
    """
    Resolve the run artifact directory, preferring the new layout with fallback.

    Returns the new flattened path when it exists, otherwise falls back to the
    legacy experiment-grouped path when present. If neither exists, the new
    path is returned.
    """
    new_path = Path(
        get_run_artifact_dir(
            run_name=run_name,
            output_dir=output_dir,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
        )
    )
    if new_path.exists():
        return new_path

    legacy_path = Path(
        get_legacy_run_artifact_dir(
            run_name=run_name,
            output_dir=output_dir,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
        )
    )
    if legacy_path.exists():
        return legacy_path

    return new_path


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
