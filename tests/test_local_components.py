"""Tests for loading local experiment components."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import ModuleType

from dl_core import load_builtin_components, load_local_components
from dl_core.core import (
    DATASET_REGISTRY,
    ENVIRONMENT_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
)
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


def test_load_local_components_registers_environment(tmp_path: Path) -> None:
    """Local environment packages should participate in normal discovery."""
    project_dir = tmp_path / "environment-project"
    environments_dir = project_dir / "src" / "environments"
    environments_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'environment-project'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    (environments_dir / "__init__.py").write_text(
        "from .local_environment import LocalEnvironment\n",
        encoding="utf-8",
    )
    (environments_dir / "local_environment.py").write_text(
        "from dl_core.core import register_environment\n\n"
        "@register_environment('local_environment')\n"
        "class LocalEnvironment:\n"
        "    pass\n",
        encoding="utf-8",
    )

    imported_modules = load_local_components(project_dir)

    assert imported_modules == ["environments"]
    assert ENVIRONMENT_REGISTRY.get_class("local_environment").__name__ == (
        "LocalEnvironment"
    )


def test_load_local_components_restores_shadowed_third_party_module(
    tmp_path: Path,
) -> None:
    """Flat local package names must not replace existing third-party modules."""
    target_dir = create_experiment_scaffold("collision-demo", root_dir=str(tmp_path))
    foreign_datasets = ModuleType("datasets")
    foreign_datasets.marker = "third-party"  # type: ignore[attr-defined]
    previous_datasets = sys.modules.get("datasets")
    sys.modules["datasets"] = foreign_datasets

    try:
        load_builtin_components()
        load_local_components(target_dir / "configs" / "base.yaml")

        assert sys.modules["datasets"] is foreign_datasets
        assert DATASET_REGISTRY.get_class("collision_demo").__name__ == (
            "CollisionDemoDataset"
        )
    finally:
        if previous_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = previous_datasets


def test_load_local_components_replaces_previous_project_registrations(
    tmp_path: Path,
) -> None:
    """Loading another project should not retain or collide with the first one."""
    first_dir = create_experiment_scaffold("project-one", root_dir=str(tmp_path))
    second_dir = create_experiment_scaffold("project-two", root_dir=str(tmp_path))

    load_builtin_components()
    load_local_components(first_dir / "configs" / "base.yaml")
    first_model_class = MODEL_REGISTRY.get_class("resnet_example")
    assert Path(inspect.getfile(first_model_class)).is_relative_to(first_dir)

    load_local_components(second_dir / "configs" / "base.yaml")
    second_model_class = MODEL_REGISTRY.get_class("resnet_example")

    assert "project_one" not in DATASET_REGISTRY.list_registered()
    assert "project_two" in DATASET_REGISTRY.list_registered()
    assert Path(inspect.getfile(second_model_class)).is_relative_to(second_dir)
