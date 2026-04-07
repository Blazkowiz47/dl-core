"""Sweep progress tracker with JSON-based state management."""

from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class SweepTracker:
    """Manages sweep state tracking with JSON files."""

    def __init__(self, sweep_path: Path, experiment_name: str, sweep_id: str):
        """
        Initialize tracker for a sweep.

        Args:
            sweep_path: Path to sweep YAML file
            experiment_name: Sweep-level experiment or sweep name
            sweep_id: Unique sweep identifier
        """
        self.sweep_path = Path(sweep_path)
        # Store JSON inside the generated sweep directory next to run configs.
        sweep_dir = self.sweep_path.parent / self.sweep_path.stem
        self.json_path = sweep_dir / "sweep_tracking.json"
        self.lock_path = sweep_dir / "sweep_tracking.lock"
        self.experiment_name = experiment_name
        self.sweep_id = sweep_id
        self._lock = threading.Lock()

        logger.info(f"Initialized sweep tracker: {self.json_path}")

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare SweepTracker for pickling (used by ProcessPoolExecutor).

        Removes the unpicklable threading.Lock before serialization.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore SweepTracker after unpickling (used by ProcessPoolExecutor).

        Recreates the threading.Lock in the new process.
        """
        self.__dict__.update(state)
        # Recreate the lock in the new process
        self._lock = threading.Lock()

    def initialize_sweep(
        self,
        total_runs: int,
        user: str,
        tracking_context: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tracking_backend: Optional[str] = None,
        metrics_source_backend: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create initial JSON file with all runs as pending.

        Args:
            total_runs: Total number of runs in sweep
            user: Username running the sweep
            tracking_context: Optional tracker-specific parent or sweep context
            tracking_uri: Optional tracker endpoint or workspace URI
            tracking_backend: Tracker backend name for this sweep
            metrics_source_backend: Metrics source backend name for this sweep
            metadata: Additional metadata to store (optional)
        """
        with self._locked_access():
            # Initialize sweep data structure
            sweep_data = {
                "experiment_name": self.experiment_name,
                "sweep_config": self.sweep_path.name,
                "sweep_id": self.sweep_id,
                "user": user,
                "total_runs": total_runs,
                "tracking_context": tracking_context,
                "tracking_uri": tracking_uri,
                "tracking_backend": tracking_backend or "local",
                "metrics_source_backend": metrics_source_backend or "local",
                "runs": {},
                "last_update": datetime.now().isoformat(),
            }

            # Add optional metadata
            if metadata:
                sweep_data["metadata"] = metadata

            # Initialize all runs as pending
            for i in range(total_runs):
                sweep_data["runs"][str(i)] = {
                    "tracking_run_id": None,
                    "tracking_run_name": None,
                    "tracking_run_ref": None,
                    "tracking_backend": tracking_backend or "local",
                    "metrics_source_backend": metrics_source_backend or "local",
                    "config_path": None,
                    "artifact_dir": None,
                    "metrics_summary_path": None,
                    "metrics_history_path": None,
                    "status": "pending",
                    "updated_at": datetime.now().isoformat(),
                }

            # Write JSON file
            self._write_json(sweep_data)

            logger.info(f"Initialized sweep with {total_runs} runs")

    def update_run_status(
        self,
        run_index: int,
        status: str,
        tracking_run_id: Optional[str] = None,
        tracking_run_name: Optional[str] = None,
        tracking_run_ref: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        config_path: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        metrics_summary_path: Optional[str] = None,
        metrics_history_path: Optional[str] = None,
    ) -> None:
        """
        Update status of a specific run.

        Args:
            run_index: Index of run in sweep
            status: One of: pending, running, completed, failed, unknown
            tracking_run_id: External tracker run ID, if available
            tracking_run_name: External tracker run name, if available
            tracking_run_ref: Backend-specific tracker reference
            error_message: Error message (for failed runs)
            config_path: Path to the concrete run config file
            artifact_dir: Path to the run artifact directory
            metrics_summary_path: Path to the local metrics summary file
            metrics_history_path: Path to the local metrics history file
        """
        with self._locked_access():
            # Load current data
            sweep_data = self._read_json()

            if not sweep_data:
                logger.warning(f"Sweep JSON not found, cannot update run {run_index}")
                return

            # Update run data
            run_key = str(run_index)

            if run_key not in sweep_data["runs"]:
                # Initialize run if not exists
                sweep_data["runs"][run_key] = {}

            run_data = sweep_data["runs"][run_key]
            run_data["status"] = status
            run_data["updated_at"] = datetime.now().isoformat()

            if tracking_run_id:
                run_data["tracking_run_id"] = tracking_run_id

            if tracking_run_name:
                run_data["tracking_run_name"] = tracking_run_name

            if tracking_run_ref:
                run_data["tracking_run_ref"] = tracking_run_ref

            if error_message:
                run_data["error_message"] = error_message

            if config_path:
                run_data["config_path"] = config_path

            if artifact_dir:
                run_data["artifact_dir"] = artifact_dir

            if metrics_summary_path:
                run_data["metrics_summary_path"] = metrics_summary_path

            if metrics_history_path:
                run_data["metrics_history_path"] = metrics_history_path

            # Update last_update timestamp
            sweep_data["last_update"] = datetime.now().isoformat()

            # Write back to file
            self._write_json(sweep_data)

            logger.debug(
                f"Updated run {run_index}: status={status}, "
                f"tracking_run_id={tracking_run_id}"
            )

    def update_tracking_context(
        self,
        tracking_context: str,
        tracking_uri: Optional[str] = None,
    ) -> None:
        """
        Update external tracking context in sweep JSON.

        Args:
            tracking_context: Tracker-specific parent or sweep context
            tracking_uri: Optional tracker endpoint or workspace URI
        """
        with self._locked_access():
            sweep_data = self._read_json()

            if not sweep_data:
                logger.warning("Sweep JSON not found, cannot update tracking_context")
                return

            sweep_data["tracking_context"] = tracking_context
            if tracking_uri:
                sweep_data["tracking_uri"] = tracking_uri
            sweep_data["last_update"] = datetime.now().isoformat()

            self._write_json(sweep_data)

            logger.info("Updated sweep tracking context")

    def get_sweep_data(self) -> Dict[str, Any]:
        """
        Get current sweep data from JSON.

        Returns:
            Sweep data dictionary, or empty dict if file doesn't exist
        """
        with self._locked_access():
            return self._read_json()

    @contextmanager
    def _locked_access(self) -> Iterator[None]:
        """Acquire thread and process locks for sweep tracking file access."""
        with self._lock:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def cleanup_lock_file(self) -> None:
        """Remove the sweep lock file after sweep execution completes."""
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove sweep lock file: %s", self.lock_path)

    def get_completed_runs(self) -> List[int]:
        """
        Get indices of completed runs.

        Returns:
            List of run indices with status='completed'
        """
        sweep_data = self.get_sweep_data()

        completed = []
        for run_index_str, run_data in sweep_data.get("runs", {}).items():
            if run_data.get("status") == "completed":
                completed.append(int(run_index_str))

        return sorted(completed)

    def get_failed_runs(self) -> List[int]:
        """
        Get indices of failed runs.

        Returns:
            List of run indices with status='failed'
        """
        sweep_data = self.get_sweep_data()

        failed = []
        for run_index_str, run_data in sweep_data.get("runs", {}).items():
            if run_data.get("status") == "failed":
                failed.append(int(run_index_str))

        return sorted(failed)

    def get_unknown_runs(self) -> List[int]:
        """
        Get indices of runs with unknown status.

        Returns:
            List of run indices with status='unknown'
        """
        sweep_data = self.get_sweep_data()

        unknown = []
        for run_index_str, run_data in sweep_data.get("runs", {}).items():
            if run_data.get("status") == "unknown":
                unknown.append(int(run_index_str))

        return sorted(unknown)

    def get_pending_runs(self, expected_total_runs: Optional[int] = None) -> List[int]:
        """
        Get indices of pending runs.

        Args:
            expected_total_runs: Total runs expected in sweep (used to detect missing runs)

        Returns:
            List of run indices with status='pending' or missing from tracking file
        """
        sweep_data = self.get_sweep_data()

        pending = []
        tracked_runs = set()

        # First, collect runs that are explicitly marked as 'pending'
        for run_index_str, run_data in sweep_data.get("runs", {}).items():
            run_index = int(run_index_str)
            tracked_runs.add(run_index)
            if run_data.get("status") == "pending":
                pending.append(run_index)

        # Then, identify runs that are completely missing from tracking file
        # These should be treated as pending
        if expected_total_runs is not None:
            for i in range(expected_total_runs):
                if i not in tracked_runs:
                    pending.append(i)

        return sorted(pending)

    def _read_json(self) -> Dict[str, Any]:
        """
        Read JSON file (internal use, assumes lock is held).

        Returns:
            Sweep data dictionary, or empty dict if file doesn't exist
        """
        if not self.json_path.exists():
            return {}

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read sweep JSON: {e}")
            return {}

    def _write_json(self, data: Dict[str, Any]) -> None:
        """
        Write JSON file (internal use, assumes lock is held).

        Args:
            data: Sweep data to write
        """
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.json_path.parent,
                prefix=f"{self.json_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(data, handle, indent=2, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())

            temp_path.replace(self.json_path)
        except Exception as e:
            logger.error(f"Failed to write sweep JSON: {e}")
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
