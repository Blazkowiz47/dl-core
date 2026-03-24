"""Reusable deep learning framework core."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from pathlib import Path

from dl_core.project import (
    LOCAL_COMPONENT_SUBPACKAGES,
    add_src_to_path,
    find_project_root,
    iter_local_package_dirs,
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

__version__ = "0.1.2"


def load_builtin_components() -> None:
    """Import built-in component modules so they register themselves."""
    for module_name in _BUILTIN_COMPONENT_MODULES:
        import_module(f"dl_core.{module_name}")


def load_local_components(start_path: str | Path | None = None) -> list[str]:
    """Import local experiment packages so custom components register themselves."""
    search_root = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        return []

    add_src_to_path(project_root)
    invalidate_caches()

    imported_packages: list[str] = []
    for package_dir in iter_local_package_dirs(project_root):
        package_name = package_dir.name
        import_module(package_name)
        imported_packages.append(package_dir.name)
        for subpackage_name in LOCAL_COMPONENT_SUBPACKAGES:
            subpackage_dir = package_dir / subpackage_name
            if not subpackage_dir.is_dir():
                continue
            if not (subpackage_dir / "__init__.py").exists():
                continue

            import_module(f"{package_name}.{subpackage_name}")
            for module_path in sorted(subpackage_dir.glob("*.py")):
                if module_path.name == "__init__.py":
                    continue
                import_module(f"{package_name}.{subpackage_name}.{module_path.stem}")

    return imported_packages


__all__ = ["__version__", "load_builtin_components", "load_local_components"]
