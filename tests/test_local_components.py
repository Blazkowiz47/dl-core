"""Tests for loading local experiment components."""

from __future__ import annotations

from pathlib import Path

from dl_core import load_builtin_components, load_local_components
from dl_core.core import DATASET_REGISTRY, MODEL_REGISTRY, TRAINER_REGISTRY
from dl_core.init_experiment import create_experiment_scaffold


def test_load_local_components_registers_scaffolded_components(
    tmp_path: Path,
) -> None:
    """Scaffolded local components should register under the expected names."""
    target_dir = create_experiment_scaffold("registry-demo", root_dir=str(tmp_path))
    config_path = target_dir / "configs" / "base.yaml"

    load_builtin_components()
    imported_modules = load_local_components(config_path)

    assert "datasets" in imported_modules
    assert "models" in imported_modules
    assert "trainers" in imported_modules
    assert DATASET_REGISTRY.get_class("registry_demo").__name__ == "RegistryDemoDataset"
    assert MODEL_REGISTRY.get_class("resnet_example").__name__ == "ResNetExample"
    assert TRAINER_REGISTRY.get_class("registry_demo").__name__ == "RegistryDemoTrainer"


def test_load_local_components_returns_empty_outside_project(tmp_path: Path) -> None:
    """Non-project paths should not import any local packages."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 2025\n", encoding="utf-8")

    assert load_local_components(config_path) == []
