"""Base class for sweep execution backends."""

import logging
import shutil
import sys
import traceback
import yaml
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from dl_core.utils.sweep_tracker import SweepTracker

if TYPE_CHECKING:
    pass


class BaseExecutor(ABC):
    """
    Abstract base class for sweep executors.

    Executors handle the actual execution of sweep runs,
    while sharing config generation logic.
    """

    def __init__(
        self,
        sweep_config: Dict[str, Any],
        experiment_name: str,
        sweep_id: str,
        dry_run: bool = False,
        tracking_context: Optional[str] = None,
        resume: bool = False,
    ):
        """
        Initialize executor.

        Args:
            sweep_config: Full sweep configuration
            experiment_name: Experiment name for this sweep
            sweep_id: Unique identifier for this sweep
            dry_run: If True, print what would be done without executing
            tracking_context: Existing tracker-specific context when resuming
            resume: True if this is resuming an existing sweep
        """
        self.sweep_config = sweep_config
        self.experiment_name = experiment_name
        self.sweep_id = sweep_id
        self.dry_run = dry_run
        self.tracking_context = tracking_context
        self.resume = resume
        self.completed_runs = []
        self.failed_runs = []
        self.unknown_runs = []  # Jobs with indeterminate status due to connection issues
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract common configs from sweep config (available to all executors)
        self.executor_config = sweep_config.get("executor", {})

        # Initialize sweep tracker if sweep_file provided
        self.tracker = None
        if "sweep_file" in sweep_config:
            sweep_path = Path(sweep_config["sweep_file"])
            self.tracker = SweepTracker(
                sweep_path=sweep_path,
                experiment_name=experiment_name,
                sweep_id=sweep_id,
            )

        if self.dry_run:
            self.logger.info("[DRY RUN] Executor initialized in dry-run mode")

    @abstractmethod
    def setup(self, total_runs: int) -> None:
        """
        Setup before running sweep.

        Args:
            total_runs: Total number of runs in sweep
        """
        pass

    @abstractmethod
    def execute_run(
        self,
        run_index: int,
        config_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute a single run.

        Args:
            run_index: Index of this run in sweep
            config_path: Path to the saved config file

        Returns:
            Dictionary with execution results:
            - "success" (bool): True if run succeeded, False otherwise
            - "tracking_run_id" (Optional[str]): External tracker run ID
            - "tracking_run_name" (Optional[str]): External tracker run name
        """
        pass

    def execute_runs_parallel(
        self,
        run_descriptors: List[Tuple[int, Path]],
        max_workers: int,
    ) -> None:
        """
        Execute multiple runs in parallel using ProcessPoolExecutor.

        Standard pattern for all executors:
        1. Each worker reads config from the provided path
        2. Use ProcessPoolExecutor to execute runs in parallel

        Args:
            run_descriptors: List of tuples (run_index, config_path)
            max_workers: Maximum number of parallel workers (default: 4)
        """
        total_runs = len(run_descriptors)

        self.logger.info(
            f"Running {total_runs} runs in parallel (max_workers={max_workers})"
        )

        # Execute runs in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all runs
            future_to_index = {
                executor.submit(
                    self._execute_single_run_wrapper, run_index, config_path
                ): run_index
                for run_index, config_path in run_descriptors
            }

            # Process completed runs
            for future in as_completed(future_to_index):
                run_index = future_to_index[future]
                try:
                    result = future.result()
                    success = result.get("success", False)
                    tracking_run_id = result.get("tracking_run_id")
                    tracking_run_name = result.get("tracking_run_name")

                    if success:
                        self.completed_runs.append(run_index)
                        # Update tracker: completed
                        if self.tracker and not self.dry_run:
                            self.tracker.update_run_status(
                                run_index,
                                "completed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=tracking_run_name,
                            )
                        self.logger.info(
                            f"Run {run_index + 1}/{total_runs} completed successfully"
                        )
                    else:
                        self.failed_runs.append(run_index)
                        # Update tracker: failed
                        if self.tracker and not self.dry_run:
                            self.tracker.update_run_status(
                                run_index,
                                "failed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=tracking_run_name,
                            )
                        self.logger.error(f"Run {run_index + 1}/{total_runs} failed")
                except Exception as e:
                    self.failed_runs.append(run_index)
                    # Update tracker: failed with error
                    if self.tracker and not self.dry_run:
                        self.tracker.update_run_status(
                            run_index, "failed", error_message=str(e)
                        )
                    self.logger.error(
                        f"Run {run_index + 1}/{total_runs} failed with exception: {e}"
                    )

    def _execute_single_run_wrapper(
        self, run_index: int, config_path: Path
    ) -> Dict[str, Any]:
        """
        Wrapper for executing a single run in parallel mode.

        Subclasses can override execute_run directly or this wrapper for custom behavior.

        Args:
            run_index: Run index
            config_path: Path to the saved config file

        Returns:
            Dictionary with execution results (see execute_run for details)
        """
        # Update tracker: running (thread-safe)
        if self.tracker and not self.dry_run:
            self.tracker.update_run_status(run_index, "running")

        # Call execute_run with the config path
        return self.execute_run(run_index, config_path)

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup after sweep completes."""
        pass

    def get_progress(self) -> Dict[str, int]:
        """Get sweep progress."""
        return {
            "completed": len(self.completed_runs),
            "failed": len(self.failed_runs),
            "total": len(self.completed_runs) + len(self.failed_runs),
        }

    # High-level execution methods

    def run_sweep(
        self, run_descriptors: List[Tuple[int, Path]], max_workers: int = 1
    ) -> Dict[str, int]:
        """
        Execute a complete sweep with multiple run configurations.

        This is the main entry point for sweep execution. It handles:
        - Setup
        - Parallel or sequential execution based on max_workers
        - Teardown
        - Progress tracking

        Args:
            run_descriptors: List of tuples (run_index, config_path)
            max_workers: Maximum number of parallel workers (default: 1, sequential)

        Returns:
            Dictionary with sweep progress (completed, failed, total)

        Example:
            executor = LocalExecutor(sweep_config, "experiment", "sweep_123", dry_run=False)
            config_paths = [(0, Path("config_0.yaml")), (1, Path("config_1.yaml"))]
            result = executor.run_sweep(config_paths, max_workers=4)
            print(f"Completed: {result['completed']}/{result['total']}")
        """
        total_runs = len(run_descriptors)

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute {total_runs} runs")
            self.logger.info(
                f"[DRY RUN] Execution mode: {'parallel' if max_workers > 1 else 'sequential'}"
            )
            if max_workers > 1:
                self.logger.info(f"[DRY RUN] Max workers: {max_workers}")

        try:
            # Initialize sweep tracker (only if not resuming - avoid overwriting existing data)
            if self.tracker and not self.dry_run and not self.resume:
                self.tracker.initialize_sweep(
                    total_runs=total_runs,
                    user=self.sweep_config.get("user", "unknown"),
                    tracking_context=self.tracking_context,
                    metadata={
                        "base_config": self.sweep_config.get("base_config"),
                        "executor": self.__class__.__name__,
                    },
                )

            # Setup executor
            self.setup(total_runs)

            # Execute runs (parallel or sequential)
            if max_workers > 1:
                self.execute_runs_parallel(run_descriptors, max_workers)
            else:
                # Sequential execution
                for run_index, config_path in run_descriptors:
                    if self.tracker and not self.dry_run:
                        self.tracker.update_run_status(run_index, "running")

                    result = self.execute_run(run_index, config_path)
                    success = result.get("success", False)
                    tracking_run_id = result.get("tracking_run_id")
                    tracking_run_name = result.get("tracking_run_name")

                    if success:
                        self.completed_runs.append(run_index)
                        # Update tracker: completed
                        if self.tracker and not self.dry_run:
                            self.tracker.update_run_status(
                                run_index,
                                "completed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=tracking_run_name,
                            )
                    else:
                        self.failed_runs.append(run_index)
                        # Update tracker: failed
                        if self.tracker and not self.dry_run:
                            self.tracker.update_run_status(
                                run_index,
                                "failed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=tracking_run_name,
                            )

            # Teardown
            self.teardown()

            # Return progress
            return self.get_progress()

        except Exception as e:
            self.logger.error(f"Sweep execution failed: {e}")
            traceback.print_exc()
            raise

    def run(self, config_path: str, run_name: Optional[str] = None) -> bool:
        """
        Execute a single run configuration (non-sweep mode).

        This is useful for:
        - Testing a single configuration
        - Debugging
        - Running a specific experiment without sweep setup

        Args:
            config_path: Path to the configuration YAML file
            run_name: Optional name for the run (default: config filename without extension)

        Returns:
            True if run succeeded, False otherwise

        Example:
            executor = LocalExecutor({}, "experiment", "run_001", dry_run=False)
            success = executor.run("path/to/config.yaml", run_name="test_run")
        """
        try:
            # Load config from file
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.error(f"Config file not found: {config_path}")
                return False

            # Use config filename as default run name
            run_name = run_name or config_file.stem

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would execute single run: {run_name}")
                self.logger.info(f"[DRY RUN] Config file: {config_path}")

            # Setup for single run
            self.setup(total_runs=1)

            # Execute single run
            run_index = 0

            if not self.dry_run:
                self.logger.info(f"Executing single run: {run_name}")
                self.logger.info(f"Config file: {config_path}")

            # Execute (pass the config path)
            result = self.execute_run(run_index, config_file)
            success = result.get("success", False)

            # Teardown
            self.teardown()

            if success:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would complete run '{run_name}'")
                else:
                    self.logger.info(f"✓ Run '{run_name}' completed successfully")
            else:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Simulated run '{run_name}'")
                else:
                    self.logger.error(f"✗ Run '{run_name}' failed")

            return success

        except Exception as e:
            self.logger.error(f"Single run execution failed: {e}")
            traceback.print_exc()
            return False

    # Common helper methods for config management

    def create_sweep_config_directory(self) -> Path:
        """
        Create sweep-specific directory for storing run configs.

        Returns:
            Path to the config directory

        Example:
            If sweep_file = "experiments/lr_sweep.yaml"
            Returns: "experiments/lr_sweep/"
        """
        sweep_file = self.sweep_config.get("sweep_file", "")
        if sweep_file:
            sweep_path = Path(sweep_file)
            # Create directory with sweep name (without .yaml extension)
            config_dir = sweep_path.parent / sweep_path.stem
        else:
            # Fallback to generic sweep_configs directory
            config_dir = Path("sweep_configs") / self.sweep_id

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create directory: {config_dir}")
        else:
            config_dir.mkdir(parents=True, exist_ok=True)

        return config_dir

    def save_run_config(
        self,
        config: Dict[str, Any],
        run_index: int,
        config_dir: Optional[Path] = None,
        run_name: Optional[str] = None,
    ) -> Path:
        """
        Save run configuration to a file.

        Note: For parallel execution, use grid_utils.generate_and_save_run_configs().
        This method is mainly used for sequential execution and custom executors.

        Args:
            config: Run configuration to save
            run_index: Index of the run
            config_dir: Directory to save config (created if None)
            run_name: Optional run name to use for filename (defaults to run_{index})

        Returns:
            Path to the saved config file
        """
        if config_dir is None:
            config_dir = self.create_sweep_config_directory()

        # Use run_name if provided, otherwise use run_index
        if run_name:
            config_file = f"{run_name}.yaml"
        else:
            config_file = f"run_{run_index:03d}.yaml"

        config_path = config_dir / config_file

        if self.dry_run:
            self.logger.debug(f"[DRY RUN] Would save config to: {config_path}")
        else:
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)

        return config_path

    def cleanup_config_directory(self, config_dir: Path) -> None:
        """
        Clean up generated config directory.

        Args:
            config_dir: Directory to remove
        """
        if config_dir.exists():
            try:
                shutil.rmtree(config_dir)
                self.logger.info(f"Cleaned up {config_dir}/")
            except Exception as e:
                self.logger.warning(f"Failed to clean up config files: {e}")

    def generate_run_name(self, run_config: Dict[str, Any], run_index: int) -> str:
        """
        Generate a descriptive run name from the grid parameters.

        Note: For parallel execution, use grid_utils.generate_and_save_run_configs().
        This method is mainly used for sequential execution and custom executors.

        Args:
            run_config: Complete run configuration
            run_index: Run index

        Returns:
            Descriptive run name showing what varies in this run
        """
        tracking_config = self.sweep_config.get("tracking", {})
        run_name_template = tracking_config.get("run_name_template")

        if run_name_template:
            # Use ConfigBuilder to generate run name from template
            # Import here to avoid circular dependency at module load time
            from dl_core.sweep.config import ConfigBuilder

            builder: ConfigBuilder = ConfigBuilder(self.sweep_config)
            run_name = builder._generate_run_name_from_template(
                run_config, run_name_template
            )
            return run_name

        # Default behavior: generate from grid parameters
        grid = self.sweep_config.get("grid", {})
        if not grid:
            return f"run_{run_index:03d}"

        # Extract values for grid parameters from config
        parts = []
        for param_path in grid.keys():
            # Navigate config to get the value
            keys = param_path.split(".")
            value: Any = run_config
            try:
                for key in keys:
                    if isinstance(value, list):
                        # Handle list indexing (e.g., "model.0.name")
                        value = value[int(key)]  # type: ignore
                    else:
                        value = value[key]  # type: ignore

                # Format the parameter name and value
                param_name = keys[-1]  # Last part of the path
                if isinstance(value, float):
                    parts.append(f"{param_name}_{value:.1e}")
                else:
                    parts.append(f"{param_name}_{value}")
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        # Add seed if present
        seed = run_config.get("seed")
        if seed is not None:
            parts.append(f"seed_{seed}")

        return "_".join(parts) if parts else f"run_{run_index:03d}"

    def build_command(
        self, config_path: str, run_config: Optional[Dict[str, Any]] = None
    ) -> list[str]:
        """
        Build command to execute a run based on accelerator configuration.

        If run_config is not provided, reads it from config_path to determine accelerator settings.
        The sweep runner copies sweep-level accelerator into each run config.

        Args:
            config_path: Path to the saved config file
            run_config: Optional run configuration (if not provided, will be loaded from config_path)

        Returns:
            Command as list of strings (e.g., ["python", "-m", "dl_core.worker", ...])

        Note:
            For multi-GPU training, uses torchrun with --nproc_per_node based on devices.
            Subclasses can override this method for custom command building (e.g., Azure returns string).
        """
        # Load config if not provided
        if run_config is None:
            with open(config_path, "r") as f:
                run_config = yaml.safe_load(f)

        # Read accelerator from top-level (accelerator is a standalone component now)
        accelerator_config = run_config.get("accelerator", {}) if run_config else {}
        accelerator_type = accelerator_config.get("type", "cpu")

        # Get log level from runtime config
        log_level = (
            run_config.get("runtime", {}).get("log_level", "INFO")
            if run_config
            else "INFO"
        )

        if accelerator_type == "multi_gpu":
            # Get number of GPUs from devices list
            devices = accelerator_config.get("devices", [0])
            if isinstance(devices, list):
                nproc = len(devices)
            else:
                nproc = 1

            # Use torchrun for multi-GPU training
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "--standalone",
                "-m",
                "dl_core.worker",
                "-c",
                config_path,
                "--log-level",
                log_level,
            ]
        else:
            # Single GPU or CPU - use regular python
            cmd = [
                sys.executable,
                "-m",
                "dl_core.worker",
                "-c",
                config_path,
                "--log-level",
                log_level,
            ]

        return cmd

    def inject_tracking_params(
        self,
        config: Dict[str, Any],
        tracking_context: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Inject generic tracking metadata into config.

        Args:
            config: Configuration to modify (modified in-place)
            tracking_context: Tracker-specific group or parent identifier
            tracking_uri: Tracker endpoint or workspace URI
            run_name: Name for the run
        """
        tracking = config.setdefault("tracking", {})
        if tracking_context:
            tracking["context"] = tracking_context
        if tracking_uri:
            tracking["uri"] = tracking_uri
        if run_name:
            tracking["run_name"] = run_name

        # Inject sweep_file for artifact directory structure
        if "sweep_file" in self.sweep_config:
            config["sweep_file"] = self.sweep_config["sweep_file"]
