"""
Configuration builder class.

Main class for building run configurations from sweep specifications.
Handles grid expansion, preset resolution, constraint validation, and config generation.
"""

from __future__ import annotations

import copy
import itertools
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_utils import (
    convert_numeric_strings,
    deep_get,
    deep_set,
    deep_update,
    load_global_presets,
    load_user_presets,
    load_experiment_presets,
)


class ConfigBuilder:
    """
    Builds run configurations from sweep specifications.

    This class handles:
    - Parameter grid expansion
    - Preset resolution
    - Constraint validation
    - Run name generation
    - Config saving

    Example:
        >>> builder = ConfigBuilder(sweep_config)
        >>> run_configs = builder.generate_run_configs(base_config)
        >>> config_paths = builder.save_configs(run_configs, output_dir)
    """

    def __init__(
        self,
        sweep_config: Dict[str, Any],
        template_presets: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ConfigBuilder with three-tier preset system.

        Preset hierarchy (later tiers override earlier):
        1. Bundled global presets
        2. Repository-level presets - optional
        3. Experiment-level presets - optional
        4. Template presets (from sweep template file) - optional

        Args:
            sweep_config: Full sweep configuration
            template_presets: Optional template-specific presets
        """
        self.sweep_config = sweep_config
        self.template_presets = template_presets or {}

        # Load three-tier presets
        sweep_path = sweep_config.get("sweep_file")
        global_presets = load_global_presets()
        user_presets = load_user_presets(sweep_path)
        experiment_presets = load_experiment_presets(sweep_path)

        # Merge presets: global → user → experiment → template
        # Later tiers override earlier tiers
        self.full_presets = deep_update(
            global_presets,
            deep_update(
                user_presets, deep_update(experiment_presets, self.template_presets)
            ),
        )

        # Extract sweep components
        self.grid = sweep_config.get("grid", {})
        self.fixed_params = sweep_config.get("fixed", {})
        self.constraints = sweep_config.get("parameter_constraints", [])
        self.tracking_config = sweep_config.get("tracking", {})

    def generate_run_configs(
        self, base_config: Dict[str, Any], seeds: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all run configurations from sweep spec.

        Args:
            base_config: Base training configuration
            seeds: Optional list of seeds (uses sweep config if not provided)

        Returns:
            List of complete run configurations
        """
        # Get seeds
        if seeds is None:
            seeds = self.sweep_config.get("seeds", [base_config.get("seed", 42)])

        # Resolve preset references in grid
        resolved_grid = self._resolve_preset_references(self.grid)

        # Expand parameter combinations
        if self.constraints:
            param_combinations = self._expand_with_constraints(
                resolved_grid, self.constraints
            )
        else:
            param_combinations = self._expand_grid(resolved_grid)

        # Generate run configs for each combination × seed
        all_configs = []
        for param_combo in param_combinations:
            for seed in seeds:
                # Merge: base + fixed + params + seed
                run_config = self._apply_parameters(
                    base_config.copy(), self.fixed_params
                )
                run_config = self._apply_parameters(run_config, param_combo)
                run_config["seed"] = seed

                # Merge sweep-level accelerator if present (grid values take precedence)
                sweep_accelerator = self.sweep_config.get("accelerator")
                if sweep_accelerator:
                    # Start with sweep accelerator as defaults, then apply grid overrides
                    merged_accelerator = copy.deepcopy(sweep_accelerator)
                    if "accelerator" in run_config:
                        merged_accelerator = deep_update(
                            merged_accelerator, run_config["accelerator"]
                        )
                    run_config["accelerator"] = merged_accelerator

                sweep_executor = self.sweep_config.get("executor")
                if sweep_executor:
                    merged_executor = copy.deepcopy(sweep_executor)
                    if "executor" in run_config:
                        merged_executor = deep_update(
                            merged_executor, run_config["executor"]
                        )
                    run_config["executor"] = merged_executor

                all_configs.append(run_config)

        return all_configs

    def save_configs(
        self,
        run_configs: List[Dict[str, Any]],
        output_dir: Path,
    ) -> List[Tuple[int, Dict[str, Any], Path]]:
        """
        Generate run names and save all configs to disk.

        Args:
            run_configs: List of complete run configurations
            output_dir: Directory to save config files
            dry_run: If True, don't actually save files

        Returns:
            List of tuples (run_index, run_config, config_path)
        """
        # Get run name template if specified
        run_name_template = self.tracking_config.get("run_name_template")

        output_dir.mkdir(parents=True, exist_ok=True)

        config_paths = []

        for idx, run_config in enumerate(run_configs):
            run_config = copy.deepcopy(run_config)
            # Get original run index if resuming, otherwise use enumeration index
            run_index = run_config.get("_sweep_run_index", idx)

            # Generate run name
            if run_name_template:
                run_name = self._generate_run_name_from_template(
                    run_config, run_name_template
                )
            else:
                run_name = self._generate_run_name_from_grid(run_config, run_index)

            runtime_config = run_config.setdefault("runtime", {})
            runtime_config["name"] = run_name

            tracking_config = run_config.get("tracking")
            if isinstance(tracking_config, dict):
                tracking_config["run_name"] = run_name

            # Save config to file
            config_file = f"{run_name}.yaml"
            config_path = output_dir / config_file

            with open(config_path, "w") as f:
                yaml.dump(run_config, f, sort_keys=False)

            config_paths.append((run_index, run_config, config_path))

        return config_paths

    def _resolve_preset_references(self, grid: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve preset references in grid."""
        resolved_grid = {}

        for key, value in grid.items():
            resolved_value = self._resolve_preset_value(value)

            # If resolved value is dict with dotted keys, merge into grid
            if isinstance(resolved_value, dict) and any(
                "." in k for k in resolved_value.keys()
            ):
                for param_key, param_value in resolved_value.items():
                    resolved_grid[param_key] = param_value
            else:
                resolved_grid[key] = resolved_value

        return resolved_grid

    def _resolve_preset_value(self, value: Any) -> Any:
        """Recursively resolve preset references."""
        if isinstance(value, str) and value.startswith("preset:"):
            preset_path = value.replace("preset:", "")
            path_parts = preset_path.split(".")

            current = self.full_presets
            try:
                for part in path_parts:
                    current = current[part]
                return current
            except KeyError:
                raise ValueError(f"Preset '{preset_path}' not found in presets.")

        elif isinstance(value, list):
            return [self._resolve_preset_value(item) for item in value]

        elif isinstance(value, dict):
            return {k: self._resolve_preset_value(v) for k, v in value.items()}

        else:
            return value

    def _expand_grid(self, grid: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand grid into all parameter combinations."""
        if not grid:
            return [{}]

        # Separate single values from lists
        param_lists = {}
        for key, value in grid.items():
            if isinstance(value, list):
                param_lists[key] = value
            else:
                param_lists[key] = [value]

        # Generate all combinations
        keys = list(param_lists.keys())
        value_combinations = itertools.product(*[param_lists[k] for k in keys])

        combinations = []
        for value_combo in value_combinations:
            combo_dict = dict(zip(keys, value_combo))
            combinations.append(combo_dict)

        return combinations

    def _expand_with_constraints(
        self, grid: Dict[str, Any], constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Expand grid with constraint validation and value application."""
        # Get all combinations first
        all_combinations = self._expand_grid(grid)

        # Filter and apply constraint values
        valid_combinations = []
        for combo in all_combinations:
            is_valid, matched_then_values = self._validate_constraints(
                combo, constraints
            )
            if is_valid:
                # Apply "then" values from matched if-then constraints
                if matched_then_values:
                    combo = self._apply_constraint_then_values(
                        combo, matched_then_values
                    )
                valid_combinations.append(combo)

        return valid_combinations

    def _validate_constraints(
        self, combo: Dict[str, Any], constraints: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a parameter combination against constraints.

        Supports dotted keys in constraints (e.g., "models.lora_dino.load_checkpoint.run_id")
        to match against nested dict values in combos.

        Returns:
            Tuple of (is_valid, matched_then_values)
            - is_valid: Whether combo passes all constraints
            - matched_then_values: List of "then" dicts from matched if-then constraints
        """
        matched_then_values = []

        for constraint in constraints:
            # Handle if-then constraints
            if "if" in constraint and "then" in constraint:
                if_conditions = constraint["if"]
                then_conditions = constraint["then"]

                # Check if ALL "if" conditions match
                if_matches = all(
                    self._check_constraint_condition(combo, k, v)
                    for k, v in if_conditions.items()
                )

                if if_matches:
                    # Store "then" values to be applied later
                    matched_then_values.append(then_conditions)

            # Handle allow_only constraints
            elif "allow_only" in constraint:
                allow_patterns = constraint["allow_only"]
                matches_any = any(
                    all(
                        self._check_constraint_condition(combo, k, v)
                        for k, v in pattern.items()
                    )
                    for pattern in allow_patterns
                )
                if not matches_any:
                    return False, []

            # Handle skip_if constraints
            elif "skip_if" in constraint:
                skip_pattern = constraint["skip_if"]
                matches_skip = all(
                    self._check_constraint_condition(combo, k, v)
                    for k, v in skip_pattern.items()
                )
                if matches_skip:
                    return False, []

        return True, matched_then_values

    def _check_constraint_condition(
        self, combo: Dict[str, Any], key: str, expected_value: Any
    ) -> bool:
        """
        Check if a constraint condition matches a combo value.

        Handles both direct keys and dotted paths into nested dicts.
        For example, if combo has {"models.lora_dino.load_checkpoint": {"run_id": "abc"}},
        then key="models.lora_dino.load_checkpoint.run_id" will match "abc".

        Args:
            combo: Parameter combination
            key: Constraint key (may be dotted path)
            expected_value: Value to match

        Returns:
            True if condition matches, False otherwise
        """
        # First try direct key lookup
        if key in combo:
            return combo[key] == expected_value

        # Try dotted path lookup - check if key is a sub-path of any combo key
        for combo_key, combo_value in combo.items():
            if key.startswith(combo_key + ".") and isinstance(combo_value, dict):
                # Extract the remaining path after combo_key
                remaining_path = key[len(combo_key) + 1 :]
                try:
                    actual_value = deep_get(combo_value, remaining_path)
                    return actual_value == expected_value
                except (KeyError, TypeError):
                    continue

        return False

    def _apply_constraint_then_values(
        self, combo: Dict[str, Any], matched_then_values: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply "then" values from matched if-then constraints to combo.

        Resolves preset references and expands dotted keys.
        If multiple constraints match, later values override earlier ones.

        Args:
            combo: Parameter combination to modify
            matched_then_values: List of "then" dicts from matched constraints

        Returns:
            Updated combo with "then" values applied
        """
        combo = copy.deepcopy(combo)

        for then_dict in matched_then_values:
            for key, value in then_dict.items():
                # Resolve preset references
                resolved_value = self._resolve_preset_value(value)

                # If resolved value is a dict with dotted keys, expand them
                if isinstance(resolved_value, dict):
                    has_dotted_keys = any("." in k for k in resolved_value.keys())
                    if has_dotted_keys:
                        # Merge all dotted keys into combo (don't keep the parent key)
                        for dotted_key, dotted_value in resolved_value.items():
                            combo[dotted_key] = dotted_value
                    else:
                        # Regular dict - set it directly
                        combo[key] = resolved_value
                else:
                    # Simple value - set it directly
                    combo[key] = resolved_value

        return combo

    def _apply_parameters(self, base_config: dict, parameters: Dict[str, Any]) -> dict:
        """Apply parameter combination to config using dotted notation."""
        config = copy.deepcopy(base_config)

        for key, value in parameters.items():
            if isinstance(value, dict):
                # Check if dict contains dotted keys (preset expansion)
                has_dotted_keys = any("." in k for k in value.keys())

                if has_dotted_keys:
                    # Recursively apply dotted parameters
                    config = self._apply_parameters(config, value)
                else:
                    # Regular nested dict - merge or set
                    try:
                        current_value = deep_get(config, key)
                        if isinstance(current_value, dict):
                            merged_value = deep_update(current_value, value)
                            deep_set(config, key, merged_value)
                        else:
                            deep_set(config, key, value)
                    except KeyError:
                        deep_set(config, key, value)
            else:
                deep_set(config, key, value)

        # Convert numeric strings
        config = convert_numeric_strings(config)

        return config

    def _generate_run_name_from_template(
        self, config: Dict[str, Any], template: str
    ) -> str:
        """Generate run name from template with variable substitution."""
        pattern = r"\{([^}]+)\}"

        def replacer(match):
            var_path = match.group(1)
            try:
                value = deep_get(config, var_path)
                return str(value)
            except (KeyError, TypeError, ValueError):
                return match.group(0)

        run_name = re.sub(pattern, replacer, template)

        # Add seed if not in template
        if "{seed}" not in template:
            seed = config.get("seed")
            if seed is not None:
                run_name = f"{run_name}_seed_{seed}"

        return run_name

    def _generate_run_name_from_grid(
        self, config: Dict[str, Any], run_index: int
    ) -> str:
        """Generate run name from grid parameters (fallback)."""
        if not self.grid:
            return f"run_{run_index:03d}"

        parts = []
        for param_path in self.grid.keys():
            try:
                value = deep_get(config, param_path)
                param_name = param_path.split(".")[-1]
                if isinstance(value, float):
                    parts.append(f"{param_name}_{value:.1e}")
                else:
                    parts.append(f"{param_name}_{value}")
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        # Add seed
        seed = config.get("seed")
        if seed is not None:
            parts.append(f"seed_{seed}")

        return "_".join(parts) if parts else f"run_{run_index:03d}"
