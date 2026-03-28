"""
Tracking metadata utilities.

This module handles:
- Experiment name generation
- Tracker experiment resolution
- Generic tracking metadata extraction
- Tag extraction and template substitution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.project import find_project_root


def _as_non_empty_string(value: Any) -> str | None:
    """Return a stripped string when the input is a non-empty string."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_project_root_name(*paths: Path) -> str | None:
    """Resolve the nearest project root name from one or more candidate paths."""
    for path in paths:
        project_root = find_project_root(path)
        if project_root is None:
            continue
        return project_root.name
    return None


def resolve_tracking_experiment_name(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """
    Resolve the canonical tracker experiment name for a config.

    Resolution order:
    1. ``tracking.experiment_name``
    2. The nearest experiment repository root name
    3. ``experiment.name``
    4. ``template_name`` without common suffixes
    5. ``"experiment"``

    Args:
        config: Run or sweep configuration
        config_path: Optional path to the config file on disk

    Returns:
        Resolved tracker experiment name.
    """
    tracking_config = config.get("tracking", {})
    if isinstance(tracking_config, dict):
        configured_name = _as_non_empty_string(tracking_config.get("experiment_name"))
        if configured_name is not None:
            return configured_name

    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path))

    sweep_file = _as_non_empty_string(config.get("sweep_file"))
    if sweep_file is not None:
        candidate_paths.append(Path(sweep_file))

    project_root_name = _resolve_project_root_name(*candidate_paths)
    if project_root_name is not None:
        return project_root_name

    experiment_config = config.get("experiment", {})
    if isinstance(experiment_config, dict):
        experiment_name = _as_non_empty_string(experiment_config.get("name"))
        if experiment_name is not None:
            return experiment_name

    template_name = _as_non_empty_string(config.get("template_name"))
    if template_name is not None:
        return template_name.replace("_template", "").replace("_sweep", "")

    return "experiment"


def ensure_tracking_experiment_name(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """
    Ensure ``tracking.experiment_name`` is populated in-place.

    Args:
        config: Run or sweep configuration
        config_path: Optional path to the config file on disk

    Returns:
        The resolved tracker experiment name.
    """
    tracking_config = config.setdefault("tracking", {})
    if not isinstance(tracking_config, dict):
        tracking_config = {}
        config["tracking"] = tracking_config

    experiment_name = resolve_tracking_experiment_name(
        config,
        config_path=config_path,
    )
    tracking_config.setdefault("experiment_name", experiment_name)
    return experiment_name


def generate_experiment_name(sweep_config: dict, timestamp: str) -> str:
    """
    Generate experiment name from sweep configuration.

    Args:
        sweep_config: Processed sweep configuration
        timestamp: Timestamp string (ignored - kept for compatibility)

    Returns:
        Generated experiment name
    """
    del timestamp
    sweep_file = sweep_config.get("sweep_file")
    config_path = Path(str(sweep_file)) if sweep_file else None
    return resolve_tracking_experiment_name(sweep_config, config_path=config_path)


def extract_tracking_config(sweep_config: dict, run_config: dict) -> dict:
    """
    Extract tracking metadata for a specific run.

    Args:
        sweep_config: Full sweep configuration
        run_config: Configuration for specific run

    Returns:
        Tracking metadata dictionary
    """
    from ..config import deep_get

    tracking_config = {}

    # Get template configuration
    tracking_template = sweep_config.get("tracking", {})

    # Extract auto-tags
    auto_tags = {}
    if "auto_tags" in tracking_template:
        for tag_name, config_path in tracking_template["auto_tags"].items():
            if isinstance(config_path, str) and config_path.startswith("from:"):
                path = config_path.replace("from:", "")
                try:
                    value = deep_get(run_config, path)
                    auto_tags[tag_name] = str(value)
                except KeyError:
                    continue
            else:
                auto_tags[tag_name] = str(config_path)

    # Add user-defined tags
    if "custom_tags" in sweep_config:
        auto_tags.update(sweep_config["custom_tags"])

    # Add user tag
    if "user" in sweep_config:
        auto_tags["user"] = sweep_config["user"]

    # Add template tag
    if "template_name" in sweep_config:
        auto_tags["template"] = sweep_config["template_name"]

    tracking_config["tags"] = auto_tags

    # Generate description
    description = sweep_config.get("description", "")
    if "description_template" in tracking_template:
        try:
            template_str = tracking_template["description_template"]
            # Extract values for template substitution
            template_vars = {}
            for key, value in auto_tags.items():
                template_vars[key] = value

            # Try to format with available variables
            description = template_str.format(**template_vars)
        except (KeyError, ValueError):
            # Fall back to original description
            pass

    tracking_config["description"] = description

    return tracking_config
