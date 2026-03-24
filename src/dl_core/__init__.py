"""Reusable deep learning framework core."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from pathlib import Path
import sys

from dl_core.project import (
    LOCAL_COMPONENT_SUBPACKAGES,
    add_src_to_path,
    find_project_root,
)

_BUILTIN_COMPONENT_MODULES = (
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
    "schedulers",
    "trainers",
)

__version__ = "0.1.5"


def load_builtin_components() -> None:
    """Import built-in component modules so they register themselves."""
    for module_name in _BUILTIN_COMPONENT_MODULES:
        import_module(f"dl_core.{module_name}")


def _clear_local_modules() -> None:
    """Drop cached flat-layout modules before loading a project."""
    module_prefixes = (*LOCAL_COMPONENT_SUBPACKAGES, "bootstrap")
    for module_name in list(sys.modules):
        if module_name in module_prefixes:
            sys.modules.pop(module_name, None)
            continue

        if any(module_name.startswith(f"{prefix}.") for prefix in module_prefixes):
            sys.modules.pop(module_name, None)


def load_local_components(start_path: str | Path | None = None) -> list[str]:
    """Import local experiment modules so custom components register themselves."""
    search_root = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        return []

    add_src_to_path(project_root)
    invalidate_caches()
    _clear_local_modules()

    imported_modules: list[str] = []
    src_dir = project_root / "src"
    bootstrap_path = src_dir / "bootstrap.py"
    if bootstrap_path.exists():
        import_module("bootstrap")
        imported_modules.append("bootstrap")

    for subpackage_name in LOCAL_COMPONENT_SUBPACKAGES:
        subpackage_dir = src_dir / subpackage_name
        if not subpackage_dir.is_dir():
            continue
        if not (subpackage_dir / "__init__.py").exists():
            continue

        import_module(subpackage_name)
        imported_modules.append(subpackage_name)
        for module_path in sorted(subpackage_dir.glob("*.py")):
            if module_path.name == "__init__.py":
                continue
            import_module(f"{subpackage_name}.{module_path.stem}")

    return imported_modules


__all__ = ["__version__", "load_builtin_components", "load_local_components"]
