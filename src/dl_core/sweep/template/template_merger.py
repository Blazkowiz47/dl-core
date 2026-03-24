"""
Template merging and sweep processing.

This module handles:
- Merging user sweeps with templates
- Resolving preset references
- Loading user sweep files
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Union

import yaml

from .template_loader import SweepTemplate, TemplateError, load_template
from .template_validator import validate_sweep_config
from ..config import load_global_presets, load_user_presets, load_experiment_presets


def resolve_preset_references(params: dict, preset_configs: dict) -> dict:
    """
    Resolve preset references in parameters.

    Args:
        params: Parameters potentially containing preset references
        preset_configs: Available preset configurations

    Returns:
        Parameters with preset references resolved
    """

    # Handle None or empty params
    if params is None:
        return {}

    def resolve_value(value):
        if isinstance(value, str) and value.startswith("preset:"):
            preset_path = value.replace("preset:", "")
            path_parts = preset_path.split(".")

            current = preset_configs
            try:
                for part in path_parts:
                    current = current[part]
                return current
            except KeyError:
                raise ValueError(f"Preset '{preset_path}' not found in presets.")

        elif isinstance(value, list):
            return [resolve_value(item) for item in value]

        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}

        else:
            return value

    return {k: resolve_value(v) for k, v in params.items()}


def merge_sweep_with_template(
    sweep_config: dict,
    template: SweepTemplate,
    sweep_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Merge user sweep configuration with template.

    Args:
        sweep_config: User's sweep configuration
        template: Loaded template
        sweep_path: Path to the sweep file (for loading user/experiment presets)

    Returns:
        Merged configuration
    """
    from ..config import deep_update

    merged = {
        "template_name": template.name,
        "base_config": template.base_config,
        "fixed": copy.deepcopy(template.fixed_params),
        "grid": copy.deepcopy(template.default_grid),
        "tracking": copy.deepcopy(template.tracking),
        "executor": copy.deepcopy(template.executor),
        "accelerator": copy.deepcopy(template.accelerator),
    }

    # Load and merge all preset sources
    # Priority: global (lowest) -> user -> experiment -> template (highest)
    all_presets = {}
    all_presets.update(load_global_presets())  # Global presets
    all_presets.update(load_user_presets(sweep_path))  # User-level presets (optional)
    all_presets.update(
        load_experiment_presets(sweep_path)
    )  # Experiment-level presets (optional)
    if template.preset_configs:  # Template presets override all (only if not None/null)
        all_presets.update(template.preset_configs)

    # Resolve preset references in template fixed parameters
    merged["fixed"] = resolve_preset_references(merged["fixed"], all_presets)

    # Merge fixed parameters
    if "fixed" in sweep_config:
        user_fixed = resolve_preset_references(sweep_config["fixed"], all_presets)
        merged["fixed"] = deep_update(merged["fixed"], user_fixed)

    # Merge/override grid parameters
    if "grid" in sweep_config:
        user_grid = sweep_config["grid"]

        # Resolve preset references first
        user_grid = resolve_preset_references(user_grid, all_presets)

        # User grid takes priority - start with template defaults and override with user values
        # For conflicting parameter paths, user values completely replace template values
        template_grid = merged.get("grid") or {}

        # Check for conflicting parameter paths
        user_param_prefixes = set()
        for user_key in user_grid.keys():
            # Extract parameter path prefixes (e.g., "model.0" from "model.0.lora_alpha")
            parts = user_key.split(".")
            for i in range(1, len(parts) + 1):
                user_param_prefixes.add(".".join(parts[:i]))

        # Remove conflicting template parameters
        filtered_template_grid = {}
        for template_key, template_value in template_grid.items():
            # Check if this template parameter conflicts with any user parameter
            template_parts = template_key.split(".")
            conflicts = False
            for i in range(1, len(template_parts) + 1):
                template_prefix = ".".join(template_parts[:i])
                if template_prefix in user_param_prefixes:
                    conflicts = True
                    break

            if not conflicts:
                filtered_template_grid[template_key] = template_value

        # Merge filtered template grid with user grid (user takes priority)
        merged["grid"] = deep_update(filtered_template_grid, user_grid)

    # Merge tracking configuration (user overrides template)
    if "tracking" in sweep_config:
        user_tracking = sweep_config["tracking"]
        merged["tracking"] = deep_update(merged["tracking"], user_tracking)

    # Merge executor configuration (user overrides template)
    if "executor" in sweep_config:
        user_executor = sweep_config["executor"]
        merged["executor"] = deep_update(merged["executor"], user_executor)

    # Merge accelerator configuration (user overrides template)
    if "accelerator" in sweep_config:
        user_accelerator = sweep_config["accelerator"]
        merged["accelerator"] = deep_update(merged["accelerator"], user_accelerator)

    # Add user-specific fields
    for key in ["user", "description", "seeds", "custom_tags", "parameter_constraints"]:
        if key in sweep_config:
            merged[key] = sweep_config[key]

    # Extract executor and accelerator from fixed section to top-level
    # Presets get resolved into fixed with dotted keys like "executor.name", "executor.compute_target"
    # Extract them to top-level executor/accelerator dicts for runner to use
    _extract_component_from_fixed(merged, "executor")
    _extract_component_from_fixed(merged, "accelerator")

    return merged


def _extract_component_from_fixed(config: dict, component_name: str) -> None:
    """
    Extract component config from fixed section to top-level.

    Presets like "preset:executors.azure" get resolved into fixed with nested dotted keys:
    fixed:
      executor:
        executor.name: azure
        executor.compute_target: NC24s-v3-V100-low-pri

    This function extracts them to:
    executor:
      name: azure
      compute_target: NC24s-v3-V100-low-pri

    Args:
        config: Sweep configuration to modify in-place
        component_name: Component to extract ("executor" or "accelerator")
    """
    fixed = config.get("fixed", {})

    # Check if component exists in fixed as a nested dict
    if component_name not in fixed:
        return

    component_from_fixed = fixed[component_name]
    if not isinstance(component_from_fixed, dict):
        return

    prefix = f"{component_name}."
    component_dict = {}

    # Extract params with dotted keys like "executor.name" -> "name"
    for key, value in component_from_fixed.items():
        if key.startswith(prefix):
            # Remove prefix to get the actual parameter name
            param_name = key[len(prefix) :]
            component_dict[param_name] = value
        else:
            # Keep keys without prefix as-is (shouldn't happen, but be safe)
            component_dict[key] = value

    # If we found component params, update top-level component
    if component_dict:
        # Update top-level component (merge with existing if any)
        existing = config.get(component_name, {})
        if existing:
            # Deep merge - component_dict takes priority
            from ..config import deep_update

            config[component_name] = deep_update(existing, component_dict)
        else:
            config[component_name] = component_dict

        # Remove the component from fixed section (it's now at top-level)
        del fixed[component_name]


def load_user_sweep(
    sweep_path: Union[str, Path], template_dirs: Optional[List[Union[str, Path]]] = None
) -> dict:
    """
    Load and process a user sweep configuration.

    Args:
        sweep_path: Path to user's sweep YAML file
        template_dirs: List of directories to search for templates

    Returns:
        Processed sweep configuration ready for execution
    """
    from ..config import resolve_conditional_values

    sweep_path = Path(sweep_path)
    if not sweep_path.exists():
        raise TemplateError(f"Sweep file not found: {sweep_path}")

    try:
        with open(sweep_path, "r", encoding="utf-8") as f:
            sweep_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise TemplateError(f"Invalid YAML in sweep file {sweep_path}: {e}")

    if not isinstance(sweep_config, dict):
        raise TemplateError(f"Sweep configuration must be a dictionary: {sweep_path}")

    # Check if sweep extends a template
    if "extends_template" in sweep_config:
        template_ref = sweep_config["extends_template"]
        template = load_template(template_ref, template_dirs, sweep_path)
        merged_config = merge_sweep_with_template(sweep_config, template, sweep_path)
    else:
        # Standalone sweep
        merged_config = copy.deepcopy(sweep_config)

    # Validate merged config (after template inheritance)
    errors = validate_sweep_config(merged_config)
    if errors:
        raise TemplateError(f"Invalid sweep configuration: {errors}")

    # Resolve conditional values and presets
    if "grid" in merged_config:
        grid = merged_config["grid"]

        # Load base config for conditional resolution
        base_config_path = merged_config.get("base_config")
        if base_config_path:
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f)

            grid = resolve_conditional_values(grid, base_config)

        merged_config["grid"] = grid

    return merged_config
