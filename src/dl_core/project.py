"""Utilities for discovering local experiment repositories."""

from __future__ import annotations

import sys
from pathlib import Path

LOCAL_COMPONENT_SUBPACKAGES = (
    "accelerators",
    "augmentations",
    "callbacks",
    "criterions",
    "datasets",
    "executors",
    "metric_managers",
    "metrics",
    "models",
    "optimizers",
    "samplers",
    "schedulers",
    "trainers",
)


def find_project_root(start_path: Path) -> Path | None:
    """Find the nearest project root containing ``pyproject.toml`` and ``src``."""
    current = start_path if start_path.is_dir() else start_path.parent
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    return None


def find_local_component_root_dir(project_root: Path) -> Path:
    """Resolve the directory where local components should be created."""
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"No src directory found under {project_root}")
    return src_dir


def add_src_to_path(project_root: Path) -> None:
    """Ensure the project ``src`` directory is importable."""
    src_dir = project_root / "src"
    src_dir_str = str(src_dir)
    if src_dir.is_dir() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
