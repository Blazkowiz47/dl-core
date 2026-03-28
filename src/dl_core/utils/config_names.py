"""Helpers for resolving experiment and run names from config."""

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
    """Resolve the nearest project root name from candidate paths."""
    for path in paths:
        project_root = find_project_root(path)
        if project_root is None:
            continue
        return project_root.name
    return None


def resolve_config_experiment_name(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """
    Resolve the canonical experiment name for tracking and artifact grouping.

    Resolution order:
    1. ``tracking.experiment_name``
    2. ``experiment.name``
    3. The nearest experiment repository root name
    4. ``template_name`` without common suffixes
    5. ``"experiment"``

    Args:
        config: Run or sweep configuration.
        config_path: Optional path to the config file on disk.

    Returns:
        Resolved experiment name.
    """
    tracking_config = config.get("tracking", {})
    if isinstance(tracking_config, dict):
        configured_name = _as_non_empty_string(tracking_config.get("experiment_name"))
        if configured_name is not None:
            return configured_name

    experiment_config = config.get("experiment", {})
    if isinstance(experiment_config, dict):
        experiment_name = _as_non_empty_string(experiment_config.get("name"))
        if experiment_name is not None:
            return experiment_name

    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path))

    sweep_file = _as_non_empty_string(config.get("sweep_file"))
    if sweep_file is not None:
        candidate_paths.append(Path(sweep_file))

    project_root_name = _resolve_project_root_name(*candidate_paths)
    if project_root_name is not None:
        return project_root_name

    template_name = _as_non_empty_string(config.get("template_name"))
    if template_name is not None:
        return template_name.replace("_template", "").replace("_sweep", "")

    return "experiment"


def resolve_config_run_name(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    fallback: str = "run",
) -> str:
    """
    Resolve the concrete run name for artifacts and tracking.

    Resolution order:
    1. ``runtime.name``
    2. Config filename stem
    3. ``fallback``

    Args:
        config: Run configuration.
        config_path: Optional path to the config file on disk.
        fallback: Final fallback when no better name is available.

    Returns:
        Resolved run name.
    """
    runtime_config = config.get("runtime", {})
    if isinstance(runtime_config, dict):
        runtime_name = _as_non_empty_string(runtime_config.get("name"))
        if runtime_name is not None:
            return runtime_name

    if config_path is not None:
        return Path(config_path).stem

    return fallback
