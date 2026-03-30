"""Init extension discovery and scaffold hooks for dl-init."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "dl_core.init_extensions"


@dataclass(frozen=True)
class ProjectNames:
    """Normalized project names used while generating a scaffold."""

    project_name: str
    project_slug: str
    component_name: str
    dataset_name: str
    dataset_class_name: str
    model_name: str
    model_class_name: str
    trainer_name: str
    trainer_class_name: str


@dataclass
class ScaffoldContext:
    """Mutable scaffold state that init extensions can update."""

    target_dir: Path
    templates_dir: Path
    project: ProjectNames
    files: dict[Path, str]
    enabled_extensions: set[str] = field(default_factory=set)

    def get_file(self, relative_path: str | Path) -> str:
        """Return file content for a generated relative path."""
        return self.files[Path(relative_path)]

    def set_file(self, relative_path: str | Path, content: str) -> None:
        """Add or replace a generated file."""
        self.files[Path(relative_path)] = content

    def replace_in_file(self, relative_path: str | Path, old: str, new: str) -> None:
        """Replace text inside a generated file."""
        relative = Path(relative_path)
        self.files[relative] = self.files[relative].replace(old, new)

    def append_line(self, relative_path: str | Path, line: str) -> None:
        """Append a line to a generated text file if it is not already present."""
        relative = Path(relative_path)
        content = self.files[relative]
        if line in content:
            return

        suffix = "\n" if content.endswith("\n") else "\n\n"
        self.files[relative] = f"{content}{suffix}{line}\n"

    def add_dependency(self, requirement: str) -> None:
        """Append a dependency to the generated project pyproject."""
        relative = Path("pyproject.toml")
        content = self.files[relative]
        dependency_line = f'    "{requirement}",\n'
        if dependency_line in content:
            return

        marker = "dependencies = [\n"
        self.files[relative] = content.replace(
            marker,
            f"{marker}{dependency_line}",
            1,
        )

    def append_bootstrap_import(self, import_line: str) -> None:
        """Append an import to the generated bootstrap module."""
        self.append_line(Path("src") / "bootstrap.py", import_line)

    def append_readme_note(self, note: str) -> None:
        """Append a note to the generated README."""
        self.append_line("README.md", note)


class InitExtension:
    """Base interface for scaffold extensions."""

    name = ""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register extension-specific CLI arguments."""

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Return whether this extension should be enabled."""
        return False

    def apply(self, context: ScaffoldContext) -> None:
        """Apply extension-specific scaffold mutations."""
        raise NotImplementedError


def _builtin_init_extensions() -> dict[str, InitExtension]:
    """Return bundled init extensions."""
    return {}


def _iter_entry_points(group: str) -> list[EntryPoint]:
    """Return entry points for the requested group across Python versions."""
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    if isinstance(discovered, dict):
        return list(discovered.get(group, ()))
    return [entry_point for entry_point in discovered if entry_point.group == group]


def _normalize_loaded_extension(candidate: Any) -> InitExtension:
    """Convert a loaded entry-point target to an init extension instance."""
    if isinstance(candidate, InitExtension):
        return candidate

    if isinstance(candidate, type) and issubclass(candidate, InitExtension):
        return candidate()

    if callable(candidate):
        produced = candidate()
        if isinstance(produced, InitExtension):
            return produced

    raise TypeError(
        "Init extension entry points must resolve to an InitExtension instance, "
        "InitExtension subclass, or zero-argument factory."
    )


def discover_init_extensions(
    include_builtin: bool = True,
) -> dict[str, InitExtension]:
    """Discover bundled and installed init extensions."""
    discovered: dict[str, InitExtension] = {}
    if include_builtin:
        discovered.update(_builtin_init_extensions())

    for entry_point in _iter_entry_points(ENTRY_POINT_GROUP):
        discovered[entry_point.name] = _normalize_loaded_extension(entry_point.load())
    return discovered


def resolve_enabled_extensions(
    args: argparse.Namespace,
    discovered_extensions: dict[str, InitExtension],
) -> set[str]:
    """Resolve enabled extensions from parsed CLI arguments."""
    enabled: set[str] = set()
    for name, extension in discovered_extensions.items():
        if extension.is_enabled(args, discovered_extensions):
            enabled.add(name)
    return enabled
