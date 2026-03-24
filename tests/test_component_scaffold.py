"""Tests for local component scaffold generation."""

from __future__ import annotations

from pathlib import Path

from dl_core import load_builtin_components, load_local_components
from dl_core.cli import main as cli_main
from dl_core.core import AUGMENTATION_REGISTRY, SAMPLER_REGISTRY
from dl_core.init_experiment import create_experiment_scaffold


def test_cli_add_augmentation_registers_component(tmp_path: Path) -> None:
    """The add command should create and register local augmentation stubs."""
    target_dir = create_experiment_scaffold("component-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "augmentation",
            "Custom1",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = (
        target_dir
        / "src"
        / "component_demo"
        / "augmentations"
        / "custom1.py"
    )
    assert component_path.exists()
    assert '@register_augmentation(["custom1", "Custom1"])' in (
        component_path.read_text()
    )

    load_builtin_components()
    imported_packages = load_local_components(target_dir / "configs" / "base.yaml")

    assert "component_demo" in imported_packages
    assert AUGMENTATION_REGISTRY.get_class("custom1").__name__ == (
        "Custom1Augmentation"
    )
    assert AUGMENTATION_REGISTRY.get_class("Custom1").__name__ == (
        "Custom1Augmentation"
    )


def test_cli_add_sampler_from_nested_path_creates_importable_package(
    tmp_path: Path,
) -> None:
    """Sampler scaffolds should work even when created from a nested project path."""
    target_dir = create_experiment_scaffold("sampler-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "sampler",
            "Passthrough1",
            "--root-dir",
            str(target_dir / "configs"),
        ]
    )

    assert exit_code == 0
    sampler_dir = target_dir / "src" / "sampler_demo" / "samplers"
    assert (sampler_dir / "__init__.py").exists()
    assert (sampler_dir / "passthrough1.py").exists()

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert SAMPLER_REGISTRY.get_class("passthrough1").__name__ == (
        "Passthrough1Sampler"
    )
