"""
Tracking metadata utilities.

This module handles:
- Experiment name generation
- Tracker experiment resolution
- Generic tracking metadata extraction
- Tag extraction and template substitution
"""

from __future__ import annotations

from typing import Any

from dl_core.utils.config_names import resolve_config_experiment_name


def resolve_tracking_experiment_name(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """
    Resolve the canonical tracker experiment name for a config.

    Resolution order:
    1. ``tracking.experiment_name``
    2. ``experiment.name``
    3. The nearest experiment repository root name
    4. ``template_name`` without common suffixes
    5. ``"experiment"``

    Args:
        config: Run or sweep configuration
        config_path: Optional path to the config file on disk

    Returns:
        Resolved tracker experiment name.
    """
    return resolve_config_experiment_name(config, config_path=config_path)


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
