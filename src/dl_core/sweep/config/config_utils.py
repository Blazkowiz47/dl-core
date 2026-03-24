"""
Configuration utility functions.

Pure utility functions for configuration manipulation:
- Type conversions
- Dictionary navigation (deep_get, deep_set, deep_update)
- Preset loading
- Conditional value resolution
- Validation
- Tag extraction
"""

from __future__ import annotations

import copy
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def convert_numeric_strings(obj: Any) -> Any:
    """
    Recursively convert string representations of numbers to int/float.

    Converts strings like "1e-3", "0.001", "42" to their numeric types.
    Leaves other strings unchanged.

    Args:
        obj: Any object (dict, list, str, etc.)

    Returns:
        Object with numeric strings converted to numbers
    """
    if isinstance(obj, dict):
        return {k: convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numeric_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert to number
        try:
            # Check if it's scientific notation or contains decimal point
            if "e" in obj.lower() or "." in obj:
                return float(obj)
            else:
                # Try int first, fall back to float if needed
                return int(obj)
        except ValueError:
            # Not a number, return as-is
            return obj
    else:
        # Return other types unchanged
        return obj


def deep_get(config: dict, dotted_key: str) -> Any:
    """
    Get value from nested dict using dotted key notation.

    Supports wildcard (*) for dynamic key matching:
    - Use '*' to match any single key at that level
    - Returns value from first matching key (deterministic in Python 3.7+)
    - Multiple wildcards supported: a.*.b.*.c

    Args:
        config: Configuration dictionary
        dotted_key: Key in dotted notation (e.g., "model.0.name", "models.*.variant")

    Returns:
        Value at the dotted key path

    Raises:
        KeyError: If key path doesn't exist, wildcard matches nothing, or
                  wildcard applied to non-dict

    Examples:
        >>> # Standard dotted path
        >>> config = {'models': {'dino': {'variant': 'v1'}}}
        >>> deep_get(config, 'models.dino.variant')
        'v1'

        >>> # Wildcard matching
        >>> deep_get(config, 'models.*.variant')
        'v1'

        >>> # Wildcard with list indices
        >>> config = {'models': {'m1': {'layers': [{'size': 10}]}}}
        >>> deep_get(config, 'models.*.layers.0.size')
        10
    """
    cur = config
    parts = dotted_key.split(".")
    for part in parts:
        if part == "*":
            # Wildcard: match first key in current dictionary
            if not isinstance(cur, dict):
                raise KeyError(
                    f"Cannot apply wildcard '*' to non-dict in path '{dotted_key}'"
                )
            if not cur:  # Empty dict
                raise KeyError(
                    f"Wildcard '*' in path '{dotted_key}' matched no keys (empty dict)"
                )
            # Get first key (dict iteration order guaranteed in Python 3.7+)
            first_key = next(iter(cur))
            cur = cur[first_key]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                raise KeyError(f"List index {idx} out of range for key {dotted_key}")
        else:
            raise KeyError(f"Key {dotted_key} not found in config")
    return cur


def deep_set(config: dict, dotted_key: str, value: Any) -> None:
    """
    Set value in nested dict using dotted key notation.

    Supports wildcard (*) for applying values to all existing keys:
    - Use '*' to apply to all keys at that level
    - Example: "models.*.num_frames" sets num_frames on all models

    Args:
        config: Configuration dictionary (modified in-place)
        dotted_key: Key in dotted notation (e.g., "model.0.name", "models.*.lr")
        value: Value to set
    """
    cur = config
    parts = dotted_key.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "*":
            # Wildcard: recursively set on all existing dict keys
            if isinstance(cur, dict):
                remaining_key = ".".join(parts[i + 1 :])
                for key in cur:
                    if isinstance(cur[key], dict):
                        deep_set(cur[key], remaining_key, value)
                return
            else:
                raise ValueError(f"Cannot use wildcard on non-dict: {dotted_key}")
        elif part.isdigit():
            # Handle list indices
            idx = int(part)
            if isinstance(cur, list):
                while len(cur) <= idx:
                    cur.append({})
                if not isinstance(cur[idx], dict):
                    cur[idx] = {}
                cur = cur[idx]
            else:
                raise ValueError(f"Cannot index non-list with {part}")
        else:
            # Handle dict keys
            if part not in cur:
                cur[part] = {}
            elif not isinstance(cur[part], (dict, list)):
                cur[part] = {}
            cur = cur[part]

    final_key = parts[-1]
    if final_key == "*":
        # Wildcard at end: apply to all existing keys
        if isinstance(cur, dict):
            for key in cur:
                cur[key] = value
        return
    elif final_key.isdigit() and isinstance(cur, list):
        idx = int(final_key)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    else:
        cur[final_key] = value


def deep_update(base: dict, update: dict) -> dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        update: Dictionary with updates to apply

    Returns:
        New dictionary with updates merged
    """
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_centralized_presets() -> Dict[str, Any]:
    """
    Load preset configurations from centralized presets.yaml file.

    DEPRECATED: Use load_global_presets() instead.
    Kept for backward compatibility.

    Returns:
        Dictionary of preset configurations, or empty dict if not found
    """
    presets_file = Path(__file__).parent.parent / "presets.yaml"
    if presets_file.exists():
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            # Fall back to empty presets if loading fails
            pass
    return {}


def _find_parent_preset_files(sweep_path: Path) -> List[Path]:
    """Find all parent ``presets.yaml`` files from nearest to farthest."""
    preset_files: List[Path] = []
    for parent in sweep_path.parents:
        candidate = parent / "presets.yaml"
        if candidate.exists():
            preset_files.append(candidate)
    return preset_files


def _find_project_root_from_sweep(sweep_path: Path) -> Optional[Path]:
    """Find the nearest project root for a sweep file."""
    current = sweep_path if sweep_path.is_dir() else sweep_path.parent
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    return None


def load_global_presets() -> Dict[str, Any]:
    """
    Load bundled preset configurations shipped with dl-core, if available.

    This keeps the package decoupled from any repository layout. Experiment-specific
    repositories can layer their own presets via ``load_user_presets`` and
    ``load_experiment_presets``.

    Returns:
        Dictionary of global preset configurations, or empty dict if not found
    """
    logger = logging.getLogger(__name__)

    presets_file = Path(__file__).parents[2] / "templates" / "presets.yaml"

    if presets_file.exists():
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                presets = yaml.safe_load(f) or {}
                logger.debug(f"Loaded global presets from {presets_file}")
                return presets
        except Exception as e:
            logger.warning(f"Failed to load global presets from {presets_file}: {e}")
            return {}
    return {}


def load_user_presets(sweep_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load repository-level preset configurations (OPTIONAL).

    Args:
        sweep_path: Path to the sweep file

    Returns:
        Dictionary of user preset configurations, or empty dict if not found
    """
    if not sweep_path:
        return {}

    logger = logging.getLogger(__name__)
    sweep_path = Path(sweep_path)

    try:
        preset_files = _find_parent_preset_files(sweep_path)
        if len(preset_files) >= 2:
            user_presets_file = preset_files[-1]
            with open(user_presets_file, "r", encoding="utf-8") as f:
                presets = yaml.safe_load(f) or {}
                logger.debug(f"Loaded user presets from {user_presets_file}")
                return presets
    except Exception as e:
        logger.debug(f"Could not load user presets: {e}")

    return {}


def load_experiment_presets(sweep_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment-level preset configurations (OPTIONAL).

    Args:
        sweep_path: Path to the sweep file

    Returns:
        Dictionary of experiment preset configurations, or empty dict if not found
    """
    if not sweep_path:
        return {}

    logger = logging.getLogger(__name__)
    sweep_path = Path(sweep_path)

    try:
        project_root = _find_project_root_from_sweep(sweep_path)
        if project_root is not None:
            config_presets_file = project_root / "configs" / "presets.yaml"
            if config_presets_file.exists():
                with open(config_presets_file, "r", encoding="utf-8") as f:
                    presets = yaml.safe_load(f) or {}
                    logger.debug(
                        f"Loaded experiment presets from {config_presets_file}"
                    )
                    return presets

        preset_files = _find_parent_preset_files(sweep_path)
        if preset_files:
            exp_presets_file = preset_files[0]
            with open(exp_presets_file, "r", encoding="utf-8") as f:
                presets = yaml.safe_load(f) or {}
                logger.debug(f"Loaded experiment presets from {exp_presets_file}")
                return presets
    except Exception as e:
        logger.debug(f"Could not load experiment presets: {e}")

    return {}


def resolve_conditional_values(
    grid: Dict[str, Any], base_config: dict
) -> Dict[str, Any]:
    """
    Resolve conditional values in grid parameters.

    Supports:
    - "same_as:other_param" - copy value from another parameter
    - "from_config:path.to.value" - extract value from base config

    Args:
        grid: Grid with potential conditional values
        base_config: Base configuration to extract values from

    Returns:
        Grid with conditional values resolved
    """
    resolved_grid = {}

    for key, value in grid.items():
        if isinstance(value, str):
            if value.startswith("same_as:"):
                ref_key = value.replace("same_as:", "")
                if ref_key in grid:
                    resolved_grid[key] = grid[ref_key]
                else:
                    raise ValueError(f"Reference key '{ref_key}' not found in grid")
            elif value.startswith("from_config:"):
                config_path = value.replace("from_config:", "")
                try:
                    resolved_grid[key] = deep_get(base_config, config_path)
                except KeyError:
                    raise ValueError(
                        f"Config path '{config_path}' not found in base config"
                    )
            else:
                resolved_grid[key] = value
        else:
            resolved_grid[key] = value

    return resolved_grid


def validate_grid_syntax(grid: Dict[str, Any]) -> List[str]:
    """
    Validate grid parameter syntax.

    Args:
        grid: Grid parameter dictionary

    Returns:
        List of error messages (empty if valid)
    """
    if not grid:
        return []

    errors = []

    for key, value in grid.items():
        if not key or not isinstance(key, str):
            errors.append(f"Invalid parameter key: {key}")
            continue

        if ".." in key or key.startswith(".") or key.endswith("."):
            errors.append(f"Invalid dotted notation in key: {key}")

        if isinstance(value, str):
            if value.startswith("same_as:") or value.startswith("from_config:"):
                continue

        if not isinstance(value, (list, str, int, float, dict)):
            errors.append(f"Invalid value type for {key}: {type(value)}")

    return errors


def extract_tags_from_config(
    config: dict, tag_mapping: Dict[str, str]
) -> Dict[str, str]:
    """
    Extract tags from config using dotted paths.

    Args:
        config: Configuration dictionary
        tag_mapping: Maps tag names to config paths

    Returns:
        Dictionary of extracted tags
    """
    tags = {}

    for tag_name, config_path in tag_mapping.items():
        try:
            value = deep_get(config, config_path)
            tags[tag_name] = str(value)
        except KeyError:
            continue

    return tags
