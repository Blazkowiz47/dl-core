"""Simple local executor."""

import json
import subprocess
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from dl_core.core import BaseExecutor, register_executor
from dl_core.utils.artifact_manager import get_run_artifact_dir
from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)


@register_executor("local")
class LocalExecutor(BaseExecutor):
    """
    Simple local executor.

    - No external tracking backend integration
    - Supports parallel execution with ProcessPoolExecutor (via base class)
    - Logs to stdout/files only
    """

    def __init__(
        self,
        sweep_config: Dict[str, Any],
        experiment_name: str,
        sweep_id: str,
        dry_run: bool = False,
        tracking_context: Optional[str] = None,
        resume: bool = False,
        **kwargs,
    ):
        """Initialize local executor.

        Args:
            sweep_config: Sweep configuration
            experiment_name: Name of experiment
            sweep_id: Unique sweep identifier
            dry_run: If True, print commands without executing
            tracking_context: Not used by LocalExecutor
            resume: Not used by LocalExecutor (for compatibility)
            max_workers: Maximum number of parallel workers (default: 1, sequential)
        """
        super().__init__(
            sweep_config,
            experiment_name,
            sweep_id,
            dry_run=dry_run,
            tracking_context=tracking_context,
            resume=resume,
        )
        self.max_workers = kwargs.get("max_workers", 1)

    def setup(self, total_runs: int) -> None:
        """Setup local executor."""
        self.logger.info(f"LocalExecutor: Starting sweep with {total_runs} runs")
        mode = "parallel" if self.max_workers > 1 else "sequential"
        self.logger.info(
            f"Mode: Simple local execution ({mode}, max_workers={self.max_workers})"
        )

    def execute_run(
        self,
        run_index: int,
        config_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute run as subprocess.

        Args:
            run_index: Run index
            config_path: Path to the saved config file

        Returns:
            Dictionary with execution results:
            - "success" (bool): True if run succeeded, False otherwise
        """
        # Read config from path at the start as suggested
        with open(config_path, "r") as f:
            run_config = yaml.safe_load(f)

        # Inject sweep metadata
        self._inject_sweep_metadata(run_config)

        # Save the modified config back to the same file
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, sort_keys=False)

        # Get launch command based on accelerator config
        runtime_config = run_config.get("runtime", {})
        run_name = resolve_config_run_name(run_config, config_path=config_path)
        output_dir = runtime_config.get("output_dir", "artifacts")
        experiment_name = resolve_config_experiment_name(
            run_config,
            config_path=config_path,
        )
        sweep_name = None
        sweep_file = run_config.get("sweep_file")
        if sweep_file:
            sweep_name = Path(sweep_file).stem

        artifact_dir = Path(
            get_run_artifact_dir(
                run_name=run_name,
                output_dir=output_dir,
                experiment_name=experiment_name,
                sweep_name=sweep_name,
            )
        ).resolve()
        cmd = self.build_command(str(config_path), run_config)
        cmd_str = " ".join(cmd)

        # Run training
        self.logger.info(f"[{run_index + 1}] Command: {cmd_str}")

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute run {run_index + 1}")
            return {
                "success": True,
                "tracking_run_name": run_name,
                "artifact_dir": str(artifact_dir),
                "metrics_summary_path": str(
                    artifact_dir / "final" / "metrics" / "summary.json"
                ),
                "metrics_history_path": str(
                    artifact_dir / "final" / "metrics" / "history.json"
                ),
            }

        result = subprocess.run(cmd, check=False)

        success = result.returncode == 0
        if success:
            self.logger.info(f"Run {run_index + 1} completed successfully")
        else:
            self.logger.error(
                f"Run {run_index + 1} failed with code {result.returncode}"
            )

        tracking_session = self._load_tracking_session(artifact_dir)

        return {
            "success": success,
            "tracking_run_id": (
                tracking_session.get("run_id")
                if isinstance(tracking_session, dict)
                else None
            ),
            "tracking_run_name": (
                tracking_session.get("run_name")
                if isinstance(tracking_session, dict)
                else run_name
            ),
            "tracking_run_ref": tracking_session,
            "artifact_dir": str(artifact_dir),
            "metrics_summary_path": str(
                artifact_dir / "final" / "metrics" / "summary.json"
            ),
            "metrics_history_path": str(
                artifact_dir / "final" / "metrics" / "history.json"
            ),
        }

    def _load_tracking_session(self, artifact_dir: Path) -> Dict[str, Any] | None:
        """
        Load tracker session metadata written by a callback.

        Args:
            artifact_dir: Run artifact directory

        Returns:
            Parsed tracking session metadata when available.
        """
        session_path = artifact_dir / "final" / "tracking" / "session.json"
        if not session_path.exists():
            return None

        try:
            with open(session_path, "r", encoding="utf-8") as handle:
                session = json.load(handle)
        except Exception as exc:
            self.logger.warning(
                f"Failed to load tracking session from {session_path}: {exc}"
            )
            return None

        if not isinstance(session, dict):
            return None
        return session

    def _inject_sweep_metadata(self, config: Dict[str, Any]) -> None:
        """Inject sweep metadata into config for artifact directory structure."""
        runtime_config = config.get("runtime", {})
        run_name = runtime_config.get("name") if isinstance(runtime_config, dict) else None

        self.inject_tracking_params(
            config,
            tracking_context=self.tracking_context,
            tracking_uri=self.tracking_uri,
            run_name=run_name if isinstance(run_name, str) else None,
        )

        # Inject sweep_file for artifact directory structure
        if "sweep_file" in self.sweep_config:
            config["sweep_file"] = self.sweep_config["sweep_file"]

        # Enable auto-resume for local executor
        config["auto_resume_local"] = True

    def teardown(self) -> None:
        """Print final stats."""
        total = len(self.completed_runs) + len(self.failed_runs)
        self.logger.info(
            f"Sweep complete: {len(self.completed_runs)}/{total} succeeded"
        )
