"""Utilities for discovering local experiment repositories and packages."""

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


def iter_local_package_dirs(project_root: Path) -> list[Path]:
    """Return importable local packages contained in ``project_root/src``."""
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        return []

    package_dirs: list[Path] = []
    for package_dir in sorted(src_dir.iterdir()):
        if not package_dir.is_dir():
            continue
        if package_dir.name.startswith(".") or package_dir.name == "dl_core":
            continue
        if not (package_dir / "__init__.py").exists():
            continue
        package_dirs.append(package_dir)
    return package_dirs


def add_src_to_path(project_root: Path) -> None:
    """Ensure the project ``src`` directory is importable."""
    src_dir = project_root / "src"
    src_dir_str = str(src_dir)
    if src_dir.is_dir() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def find_local_package_dir(
    project_root: Path,
    package_name: str | None = None,
) -> Path:
    """Resolve the local experiment package directory within a project."""
    package_dirs = iter_local_package_dirs(project_root)
    if not package_dirs:
        raise FileNotFoundError(
            f"No importable local package found under {project_root / 'src'}"
        )

    if package_name:
        for package_dir in package_dirs:
            if package_dir.name == package_name:
                return package_dir
        raise FileNotFoundError(
            f"Package '{package_name}' was not found under {project_root / 'src'}"
        )

    if len(package_dirs) == 1:
        return package_dirs[0]

    available = ", ".join(package_dir.name for package_dir in package_dirs)
    raise ValueError(
        "Multiple local packages found. "
        f"Use --package-name to choose one of: {available}"
    )
