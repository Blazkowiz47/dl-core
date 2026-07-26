"""Reusable deep learning framework core."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

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
    "environments",
    "executors",
    "episode_managers",
    "metrics_sources",
    "metric_managers",
    "metrics",
    "models",
    "optimizers",
    "samplers",
    "schedulers",
    "trackers",
    "trainers",
)

__version__ = "0.0.29"

_LOCAL_MODULES: dict[str, ModuleType] = {}
_LOCAL_REGISTRATIONS: list[tuple[Any, str, type[Any]]] = []
_LOCAL_SRC_PATH: str | None = None
_LOCAL_SRC_PATH_INSERTED = False


def load_builtin_components() -> None:
    """Import built-in component modules so they register themselves."""
    for module_name in _BUILTIN_COMPONENT_MODULES:
        import_module(f"dl_core.{module_name}")


def _is_local_module_name(module_name: str) -> bool:
    """Return whether a module uses one of the generated flat-layout names."""
    module_prefixes = (*LOCAL_COMPONENT_SUBPACKAGES, "bootstrap")
    return module_name in module_prefixes or any(
        module_name.startswith(f"{prefix}.") for prefix in module_prefixes
    )


def _clear_previous_local_state() -> None:
    """Remove registrations, modules, and paths created by the previous load."""
    global _LOCAL_SRC_PATH, _LOCAL_SRC_PATH_INSERTED

    for registry, name, registered_class in _LOCAL_REGISTRATIONS:
        registry.unregister(name, expected_class=registered_class)
    _LOCAL_REGISTRATIONS.clear()

    for module_name, module in sorted(
        _LOCAL_MODULES.items(),
        key=lambda item: item[0].count("."),
        reverse=True,
    ):
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
    _LOCAL_MODULES.clear()

    if (
        _LOCAL_SRC_PATH_INSERTED
        and _LOCAL_SRC_PATH is not None
        and _LOCAL_SRC_PATH in sys.path
    ):
        sys.path.remove(_LOCAL_SRC_PATH)
    _LOCAL_SRC_PATH = None
    _LOCAL_SRC_PATH_INSERTED = False


def _module_is_from_src(module: ModuleType, src_dir: Path) -> bool:
    """Return whether a loaded module originated inside the project source tree."""
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    try:
        return Path(module_file).resolve().is_relative_to(src_dir)
    except OSError:
        return False


def load_local_components(start_path: str | Path | None = None) -> list[str]:
    """Import local experiment modules so custom components register themselves."""
    global _LOCAL_SRC_PATH, _LOCAL_SRC_PATH_INSERTED

    search_root = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        return []

    from dl_core.core.registry import COMPONENT_REGISTRIES

    _clear_previous_local_state()

    src_dir = (project_root / "src").resolve()
    src_dir_str = str(src_dir)
    _LOCAL_SRC_PATH = src_dir_str
    _LOCAL_SRC_PATH_INSERTED = src_dir_str not in sys.path
    add_src_to_path(project_root)
    invalidate_caches()

    shadowed_modules = {
        module_name: module
        for module_name, module in list(sys.modules.items())
        if _is_local_module_name(module_name)
    }
    for module_name in shadowed_modules:
        sys.modules.pop(module_name, None)

    registry_snapshots = {
        registry: registry.registered_items() for registry in COMPONENT_REGISTRIES
    }

    imported_modules: list[str] = []
    try:
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
    except Exception:
        for registry, snapshot in registry_snapshots.items():
            current_items = registry.registered_items()
            for name, registered_class in current_items.items():
                if snapshot.get(name) is not registered_class:
                    registry.unregister(name, expected_class=registered_class)
        for module_name, module in list(sys.modules.items()):
            if _is_local_module_name(module_name) and _module_is_from_src(
                module, src_dir
            ):
                sys.modules.pop(module_name, None)
        sys.modules.update(shadowed_modules)
        if _LOCAL_SRC_PATH_INSERTED and src_dir_str in sys.path:
            sys.path.remove(src_dir_str)
        _LOCAL_SRC_PATH = None
        _LOCAL_SRC_PATH_INSERTED = False
        raise

    loaded_local_modules = {
        module_name: module
        for module_name, module in list(sys.modules.items())
        if _is_local_module_name(module_name) and _module_is_from_src(module, src_dir)
    }

    shadowed_prefixes = {name.split(".", 1)[0] for name in shadowed_modules}
    for module_name, module in list(loaded_local_modules.items()):
        if module_name.split(".", 1)[0] in shadowed_prefixes:
            if sys.modules.get(module_name) is module:
                sys.modules.pop(module_name, None)
            loaded_local_modules.pop(module_name)
    sys.modules.update(shadowed_modules)

    _LOCAL_MODULES.update(loaded_local_modules)
    for registry, snapshot in registry_snapshots.items():
        for name, registered_class in registry.registered_items().items():
            if snapshot.get(name) is not registered_class:
                _LOCAL_REGISTRATIONS.append((registry, name, registered_class))

    return imported_modules


__all__ = ["__version__", "load_builtin_components", "load_local_components"]
