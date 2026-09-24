"""Init extension discovery and scaffold hooks for dl-init."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
import logging
from pathlib import Path
import re
import sys
from typing import Any

from dl_core.core.registry import COMPONENT_REGISTRIES

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ENTRY_POINT_GROUP = "dl_core.init_extensions"
_LOGGER = logging.getLogger(__name__)


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
    tracking_backend: str | None = None

    def get_file(self, relative_path: str | Path) -> str:
        """Return file content for a generated relative path."""
        return self.files[Path(relative_path)]

    def set_file(self, relative_path: str | Path, content: str) -> None:
        """Add or replace a generated file."""
        self.files[Path(relative_path)] = content

    def replace_in_file(self, relative_path: str | Path, old: str, new: str) -> None:
        """Replace text inside a generated file."""
        relative = Path(relative_path)
        content = self.files[relative]
        count = content.count(old) if old else 0
        if count == 0:
            raise ValueError(f"Scaffold anchor not found in {relative}: {old!r}")
        if count != 1:
            raise ValueError(
                f"Expected one scaffold anchor in {relative}, found {count}: {old!r}"
            )
        self.files[relative] = content.replace(old, new, 1)

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
        try:
            existing = tomllib.loads(content)["project"].get("dependencies", [])
        except (KeyError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read project dependencies in {relative}") from exc
        requirement_name = re.split(r"[\[<>=!~;\s]", requirement, 1)[0]
        if any(
            re.split(r"[\[<>=!~;\s]", item, 1)[0].lower().replace("_", "-")
            == requirement_name.lower().replace("_", "-")
            for item in existing
        ):
            return
        dependency_line = f'    "{requirement}",\n'
        project_header = re.search(r"(?m)^\[project\][ \t]*$", content)
        if project_header is None:
            raise ValueError(f"Scaffold [project] table not found in {relative}")
        section_start = project_header.end()
        next_table = re.search(r"(?m)^\[[^\n]+\][ \t]*$", content[section_start:])
        section_end = (
            section_start + next_table.start() if next_table else len(content)
        )
        section = content[section_start:section_end]
        marker = re.search(
            r"(?m)^[ \t]*dependencies[ \t]*=[ \t]*\[([ \t]*\n|[ \t]*\])",
            section,
        )
        if marker is None:
            raise ValueError(f"Scaffold dependency anchor not found in {relative}")
        if marker.group(1).endswith("\n"):
            insert_at = section_start + marker.end()
            self.files[relative] = (
                f"{content[:insert_at]}{dependency_line}{content[insert_at:]}"
            )
        else:
            start = section_start + marker.start()
            end = section_start + marker.end()
            opening = content[start:end].split("[", 1)[0]
            self.files[relative] = (
                f"{content[:start]}{opening}[\n{dependency_line}]{content[end:]}"
            )

    def add_gitignore_patterns(self, *patterns: str) -> None:
        """Append missing ignore patterns to the generated project gitignore."""
        relative = Path(".gitignore")
        content = self.files.get(relative, "")
        existing_lines = set(content.splitlines())
        missing = [pattern for pattern in patterns if pattern not in existing_lines]
        if not missing:
            return
        prefix = content.rstrip()
        suffix = "\n".join(missing)
        self.files[relative] = f"{prefix}\n{suffix}\n" if prefix else f"{suffix}\n"

    def append_bootstrap_import(self, import_line: str) -> None:
        """Append an import to the generated bootstrap module."""
        self.append_line(Path("src") / "bootstrap.py", import_line)

    def append_readme_note(self, note: str) -> None:
        """Append a note to the generated README."""
        self.append_line("README.md", note)


class InitExtension:
    """Base interface for scaffold extensions."""

    name = ""
    tracking_backend: str | None = None
    tracking_priority = 1

    def display_name(self) -> str:
        """Return a user-facing extension name for prompts and help text."""
        return self.name

    def selection_state(self, args: argparse.Namespace) -> bool | None:
        """Return explicit CLI selection state for this extension, if any."""
        attribute_name = f"with_{self.name.replace('-', '_')}"
        value = getattr(args, attribute_name, None)
        if value is None:
            return None
        return bool(value)

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
        registered_before = {
            registry: registry.registered_items() for registry in COMPONENT_REGISTRIES
        }
        modules_before = set(sys.modules)
        try:
            discovered[entry_point.name] = _normalize_loaded_extension(
                entry_point.load()
            )
        except Exception as error:
            for registry, previous in registered_before.items():
                for name, registered_class in registry.registered_items().items():
                    if previous.get(name) is not registered_class:
                        registry.unregister(name, expected_class=registered_class)
            module_root = entry_point.value.partition(":")[0].split(".", 1)[0]
            for module_name in set(sys.modules) - modules_before:
                if module_name == module_root or module_name.startswith(
                    f"{module_root}."
                ):
                    sys.modules.pop(module_name, None)
            _LOGGER.warning(
                "Skipping init extension %r from %r: %s",
                entry_point.name,
                entry_point.value,
                error,
            )
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
