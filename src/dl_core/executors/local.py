"""Simple local executor."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dl_core.core import BaseExecutor, config_field, register_executor
from dl_core.utils.artifact_manager import (
    get_run_artifact_dir,
    select_auto_resume_run_dir,
)
from dl_core.utils.config_names import (
    resolve_config_experiment_name,
    resolve_config_run_name,
)
from .local_supervisor import LocalSupervisor


@register_executor("local")
class LocalExecutor(BaseExecutor):
    """
    Simple local executor.

    - No external tracking backend integration
    - Owns sequential and parallel subprocesses in the sweep parent
    - Logs to stdout/files only
    """

    _finalize_on_exit = True

    CONFIG_FIELDS = [
        config_field(
            "max_workers",
            "int",
            "Maximum number of local subprocesses to launch in parallel.",
            default=1,
        )
    ]

    def __init__(
        self,
        sweep_config: Dict[str, Any],
        experiment_name: str,
        sweep_id: str,
        dry_run: bool = False,
        tracking_context: Optional[str] = None,
        resume: bool = False,
        **kwargs: Any,
    ):
        """Initialize local executor.

        Args:
            sweep_config: Sweep configuration
            experiment_name: Name of experiment
            sweep_id: Unique sweep identifier
            dry_run: If True, print commands without executing
            tracking_context: Existing tracker context when resuming
            resume: Reuse the existing sweep tracker instead of initializing it
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
        self.max_workers = self.executor_config.get(
            "max_workers", kwargs.get("max_workers", 1)
        )

    def setup(self, total_runs: int) -> None:
        """Setup local executor."""
        self.logger.info(f"LocalExecutor: Starting sweep with {total_runs} runs")
        mode = "parallel" if self.max_workers > 1 else "sequential"
        self.logger.info(
            f"Mode: Simple local execution ({mode}, max_workers={self.max_workers})"
        )

    def _prepare_run(
        self, run_index: int, config_path: Path
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare a worker command and its local artifact metadata."""
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

        trainer_config = run_config.get("trainer", {})
        selected_trainer = next(iter(trainer_config.values()), {})
        continue_model = selected_trainer.get("continue_model")
        if run_config.get("auto_resume_local") and not continue_model:
            artifact_dir = select_auto_resume_run_dir(
                run_name,
                output_dir,
                experiment_name,
                sweep_name,
                str(config_path),
                preserve_yml_name=True,
                sweep_file=sweep_file,
            )
        else:
            artifact_dir = Path(
                get_run_artifact_dir(
                    run_name, output_dir, experiment_name, sweep_name
                )
            )
        artifact_dir = artifact_dir.resolve()
        cmd = self.build_command(str(config_path), run_config)
        cmd_str = " ".join(cmd)

        # Run training
        self.logger.info(f"[{run_index + 1}] Command: {cmd_str}")

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute run {run_index + 1}")
        return cmd, {
            "tracking_run_name": run_name,
            "artifact_dir": str(artifact_dir),
            "metrics_summary_path": str(
                artifact_dir / "final" / "metrics" / "summary.json"
            ),
            "metrics_history_path": str(
                artifact_dir / "final" / "metrics" / "history.json"
            ),
        }

    def _finish_run(
        self, metadata: Dict[str, Any], returncode: Optional[int], *,
        stopped: bool = False, unknown: bool = False,
    ) -> Dict[str, Any]:
        """Attach tracking references after a supervised run exits."""
        tracking_session = self._load_tracking_session(Path(metadata["artifact_dir"]))
        return {
            **metadata,
            "success": returncode == 0 and not stopped and not unknown,
            "stopped": stopped and not unknown,
            "unknown": unknown,
            "tracking_run_id": (
                tracking_session.get("run_id")
                if isinstance(tracking_session, dict)
                else None
            ),
            "tracking_run_name": (
                tracking_session.get("run_name")
                if isinstance(tracking_session, dict)
                else metadata["tracking_run_name"]
            ),
            "tracking_run_ref": tracking_session,
        }

    def execute_run(self, run_index: int, config_path: Path) -> Dict[str, Any]:
        """Execute one owned subprocess with direct Ctrl-C cleanup."""
        supervisor = LocalSupervisor(
            self, menu_enabled=False, claim_runs=False, record_results=False,
        )
        return supervisor.run([(run_index, config_path)], max_workers=1)[run_index]

    def _execute_runs(
        self, run_descriptors: List[Tuple[int, Path]], max_workers: int
    ) -> None:
        """Supervise local sweeps at every worker count in the owning parent."""
        LocalSupervisor(self, menu_enabled=True).run(run_descriptors, max_workers)

    def execute_runs_parallel(
        self, run_descriptors: List[Tuple[int, Path]], max_workers: int
    ) -> None:
        """Execute parallel local runs with the same parent supervisor."""
        self._execute_runs(run_descriptors, max_workers)

    def _classify_run_result(self, result: Dict[str, Any]) -> str:
        """Keep unconfirmed local shutdowns out of automatic retries."""
        if result.get("unknown"):
            return "unknown"
        return super()._classify_run_result(result)

    def _record_run_result(
        self, run_index: int, config_path: Path, result: Dict[str, Any]
    ) -> None:
        """Persist one claimed run's result and count it for this invocation."""
        status = self._classify_run_result(result)
        counters = {
            "completed": self.completed_runs, "failed": self.failed_runs,
            "stopped": self.stopped_runs, "unknown": self.unknown_runs,
        }
        counters[status].append(run_index)
        self._update_tracker(
            run_index, status, config_path, result=result,
            error_message=result.get("error_message"),
        )

    def get_progress(self) -> Dict[str, int]:
        """Return current-command counts, including intentionally stopped runs."""
        return {**super().get_progress(), "stopped": len(self.stopped_runs)}

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

        # Enable auto-resume for local executor
        config["auto_resume_local"] = True

    def teardown(self) -> None:
        """Print final stats."""
        total = self.get_progress()["total"]
        self.logger.info(
            f"Local execution finished: {len(self.completed_runs)}/{total} succeeded, "
            f"{len(self.stopped_runs)} stopped"
        )
