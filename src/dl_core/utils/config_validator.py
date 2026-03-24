"""
Configuration validator for v2.0 configs.

Validates that config files meet all requirements before training starts.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any


class ConfigValidator:
    """
    Validate configuration files.

    Checks:
    - Required sections present
    - Valid accelerator types
    - Dataset paths exist (warning)
    - Config structure and consistency
    """

    REQUIRED_SECTIONS = ["models", "dataset", "optimizers", "accelerator"]
    OPTIONAL_SECTIONS = [
        "runtime",
        "trainer",
        "schedulers",
        "criterions",
        "callbacks",
        "metric_managers",
        "experiment",
        "executor",
        "ema",
    ]
    VALID_ACCELERATORS = ["cpu", "single_gpu", "multi_gpu"]

    def __init__(self, config_path: str):
        """
        Initialize validator.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_config(self) -> bool:
        """
        Load config file.

        Returns:
            True if loaded successfully
        """
        if not self.config_path.exists():
            self.errors.append(f"Config file not found: {self.config_path}")
            return False

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False

    def validate(self) -> bool:
        """
        Run all validations.

        Returns:
            True if config is valid (no errors)
        """
        self.errors = []
        self.warnings = []

        # Load config
        if not self.load_config():
            return False

        # Run validation checks
        self._check_required_sections()
        self._check_dataset_config()
        self._check_model_config()
        self._check_optimizer_config()
        self._check_scheduler_config()
        self._check_accelerator_config()
        self._check_experiment_config()

        return len(self.errors) == 0

    def _check_required_sections(self) -> None:
        """Check all required top-level sections present."""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                self.errors.append(f"Missing required section: '{section}'")

    def _check_dataset_config(self) -> None:
        """Validate dataset configuration."""
        if "dataset" not in self.config:
            return

        dataset = self.config["dataset"]

        # Check classes (REQUIRED)
        classes = dataset.get("classes")
        if classes is None:
            # Backward compatibility: allow class_order but warn
            legacy_classes = dataset.get("class_order")
            if legacy_classes is None:
                self.errors.append(
                    "Missing 'dataset.classes' - required to define label mapping"
                )
            else:
                self.warnings.append(
                    "'dataset.class_order' is deprecated; rename to 'classes'"
                )
                classes = legacy_classes

        # Check num_classes
        if "num_classes" in dataset and classes is not None:
            num_classes = dataset["num_classes"]
            expected = len(classes)
            if num_classes != expected:
                self.errors.append(
                    f"Mismatch: num_classes={num_classes} but "
                    f"classes has {expected} entries"
                )

        # Check dataset paths (warning only)
        for key in ["train_root", "val_root", "test_root"]:
            if key in dataset:
                path = Path(dataset[key])
                if not path.exists():
                    self.warnings.append(f"Dataset path does not exist: {key}={path}")

    def _check_model_config(self) -> None:
        """Validate model configuration."""
        if "models" not in self.config:
            if "model" in self.config:
                self.errors.append(
                    "Use 'models' (plural) instead of deprecated 'model'"
                )
            return

        models = self.config["models"]

        if isinstance(models, dict):
            # v2.0 dict format: {model_name: {params}}
            if len(models) == 0:
                self.errors.append("'models' dict is empty")
                return

            # Check each model has a 'name' parameter
            for model_key, model_config in models.items():
                if not isinstance(model_config, dict):
                    self.errors.append(f"Model '{model_key}' config must be a dict")
                elif "name" not in model_config:
                    self.warnings.append(
                        f"Model '{model_key}' missing 'name' parameter (will use key name)"
                    )

        elif isinstance(models, list):
            self.errors.append(
                "'models' must be a dict (legacy list format is not supported)"
            )
        else:
            self.errors.append("'models' must be a dict")

    def _check_optimizer_config(self) -> None:
        """Validate optimizer configuration."""
        if "optimizers" not in self.config:
            if "optimizer" in self.config:
                self.errors.append(
                    "Use 'optimizers' (plural) instead of deprecated 'optimizer'"
                )
            return

        optimizers = self.config["optimizers"]

        if isinstance(optimizers, dict):
            if len(optimizers) == 0:
                self.errors.append("'optimizers' dict is empty")
                return

            if "name" in optimizers:
                return

            # Standard trainer uses the flat dict as the default path.
            # Nested mappings remain valid for advanced trainers.
            for opt_key, opt_config in optimizers.items():
                if not isinstance(opt_config, dict):
                    self.errors.append(f"Optimizer '{opt_key}' config must be a dict")
                elif "name" not in opt_config:
                    self.warnings.append(
                        f"Optimizer '{opt_key}' missing 'name' parameter (will use key name)"
                    )

        elif isinstance(optimizers, list):
            self.errors.append(
                "'optimizers' must be a dict (legacy list format is not supported)"
            )
        else:
            self.errors.append("'optimizers' must be a dict")

    def _check_scheduler_config(self) -> None:
        """Validate optional scheduler configuration."""
        if "schedulers" not in self.config:
            return

        schedulers = self.config["schedulers"]
        if not isinstance(schedulers, dict):
            self.errors.append("'schedulers' must be a dict")
            return

        if len(schedulers) == 0:
            self.warnings.append("'schedulers' is present but empty")
            return

        if "name" in schedulers:
            return

        for scheduler_key, scheduler_config in schedulers.items():
            if not isinstance(scheduler_config, dict):
                self.errors.append(
                    f"Scheduler '{scheduler_key}' config must be a dict"
                )
            elif "name" not in scheduler_config:
                self.warnings.append(
                    f"Scheduler '{scheduler_key}' missing 'name' parameter "
                    f"(will use key name)"
                )

    def _check_accelerator_config(self) -> None:
        """Validate top-level accelerator configuration."""
        accelerator_config = self.config.get("accelerator")

        if accelerator_config is None:
            self.errors.append(
                "Missing required top-level 'accelerator' configuration"
            )
            if "runtime" in self.config and "accelerator" in self.config["runtime"]:
                self.errors.append(
                    "Found deprecated 'runtime.accelerator'; move it to top-level "
                    "'accelerator'"
                )
            if "training" in self.config and "accelerator" in self.config["training"]:
                self.errors.append(
                    "Found deprecated 'training.accelerator'; move it to top-level "
                    "'accelerator'"
                )
            return

        # Accelerator can be string or dict
        if isinstance(accelerator_config, str):
            accel_type = accelerator_config
        elif isinstance(accelerator_config, dict):
            accel_type = accelerator_config.get("type")
        else:
            self.errors.append("Accelerator config must be string or dict")
            return

        if accel_type not in self.VALID_ACCELERATORS:
            self.errors.append(
                f"Invalid accelerator type: '{accel_type}'. "
                f"Must be one of {self.VALID_ACCELERATORS}"
            )

    def _check_experiment_config(self) -> None:
        """Validate experiment configuration."""
        if "experiment" not in self.config:
            self.warnings.append("Missing 'experiment' section - using defaults")
            return

        experiment = self.config["experiment"]

        # Experiment name recommended
        if "name" not in experiment:
            self.warnings.append(
                "Missing 'experiment.name' - recommended for organizing runs"
            )

    def print_report(self) -> bool:
        """
        Print validation report.

        Returns:
            True if valid (no errors)
        """
        print(f"\n{'=' * 70}")
        print(f"📋 Config Validation: {self.config_path.name}")
        print(f"{'=' * 70}")

        if self.errors:
            print("\n❌ ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ Config is valid!")
        elif not self.errors:
            print(f"\n✅ Config is valid (with {len(self.warnings)} warnings)")
        else:
            print(f"\n❌ Config has {len(self.errors)} error(s)")

        print("=" * 70 + "\n")

        return len(self.errors) == 0


def validate_config(config_path: str, verbose: bool = True) -> bool:
    """
    Validate a config file.

    Args:
        config_path: Path to YAML config
        verbose: Print validation report

    Returns:
        True if valid
    """
    validator = ConfigValidator(config_path)
    is_valid = validator.validate()

    if verbose:
        validator.print_report()

    return is_valid


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m dl_core.utils.config_validator <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    is_valid = validate_config(config_path)

    sys.exit(0 if is_valid else 1)
