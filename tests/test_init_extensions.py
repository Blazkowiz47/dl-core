"""Tests for init extension discovery and scaffold application."""

from __future__ import annotations

import argparse
from pathlib import Path

from pytest import MonkeyPatch

from dl_core.init_experiment import create_experiment_scaffold
from dl_core.init_extensions import (
    ENTRY_POINT_GROUP,
    InitExtension,
    ScaffoldContext,
    discover_init_extensions,
    resolve_enabled_extensions,
)


class FakeEntryPoint:
    """Small entry-point test double."""

    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self.group = ENTRY_POINT_GROUP
        self._target = target

    def load(self) -> object:
        """Return the configured target."""
        return self._target


class FakeEntryPoints:
    """Entry-point selection container for tests."""

    def __init__(self, values: list[FakeEntryPoint]) -> None:
        self.values = values

    def select(self, *, group: str) -> list[FakeEntryPoint]:
        """Return the fake entry points for the requested group."""
        return [value for value in self.values if value.group == group]


class FakeWandbExtension(InitExtension):
    """Test init extension used to validate dynamic discovery."""

    name = "wandb"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register a test W&B flag."""
        parser.add_argument("--with-wandb", action="store_true")

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable W&B when the test flag is present."""
        return bool(getattr(args, "with_wandb", False))

    def apply(self, context: ScaffoldContext) -> None:
        """Write a small marker file into the scaffold."""
        context.set_file(
            Path("configs") / "wandb.yaml",
            "tracking:\n  backend: wandb\n",
        )
        context.append_readme_note("This scaffold includes test W&B wiring.")


def test_discover_init_extensions_returns_empty_without_entry_points(
    monkeypatch: MonkeyPatch,
) -> None:
    """Plain dl-core should not ship vendor-specific init extensions."""
    monkeypatch.setattr(
        "dl_core.init_extensions.entry_points",
        lambda: FakeEntryPoints([]),
    )

    discovered = discover_init_extensions(include_builtin=False)

    assert discovered == {}


def test_discover_init_extensions_loads_entry_points(
    monkeypatch: MonkeyPatch,
) -> None:
    """Installed entry points should be exposed through the discovery helper."""
    monkeypatch.setattr(
        "dl_core.init_extensions.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint("wandb", FakeWandbExtension)]),
    )

    discovered = discover_init_extensions(include_builtin=False)
    parser = argparse.ArgumentParser(add_help=False)
    for extension in discovered.values():
        extension.add_arguments(parser)

    args = parser.parse_args(["--with-wandb"])

    assert "wandb" in discovered
    assert resolve_enabled_extensions(args, discovered) == {"wandb"}


def test_scaffold_applies_selected_init_extension(tmp_path: Path) -> None:
    """Selected init extensions should be able to add generated files."""
    target_dir = create_experiment_scaffold(
        "wandb-demo",
        root_dir=str(tmp_path),
        enabled_extensions={"wandb"},
        discovered_extensions={"wandb": FakeWandbExtension()},
    )

    assert (target_dir / "configs" / "wandb.yaml").exists()
    assert "test W&B wiring" in (target_dir / "README.md").read_text()
