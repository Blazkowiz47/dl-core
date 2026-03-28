"""
Tracking metadata utilities.

This module handles:
- Experiment name generation
- Generic tracking metadata extraction
- Tag extraction and template substitution
"""

from __future__ import annotations


def generate_experiment_name(sweep_config: dict, timestamp: str) -> str:
    """
    Generate experiment name from sweep configuration.

    Args:
        sweep_config: Processed sweep configuration
        timestamp: Timestamp string (ignored - kept for compatibility)

    Returns:
        Generated experiment name
    """
    tracking_config = sweep_config.get("tracking", {})
    if isinstance(tracking_config, dict):
        sweep_name = tracking_config.get("sweep_name")
        if isinstance(sweep_name, str) and sweep_name:
            return sweep_name

        experiment_name = tracking_config.get("experiment_name")
        if isinstance(experiment_name, str) and experiment_name:
            return experiment_name

    # Fallback to template name if no tracking config
    if "template_name" in sweep_config:
        template_name = sweep_config["template_name"]
        # Remove common suffixes
        return template_name.replace("_template", "").replace("_sweep", "")

    # Default fallback
    return "experiment"


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
