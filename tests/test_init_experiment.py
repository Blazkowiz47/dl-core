"""Tests for experiment scaffold generation."""

from __future__ import annotations

from pathlib import Path

import yaml

from dl_core.init_experiment import create_experiment_scaffold, main as init_main


def test_scaffold_uses_project_named_dataset_and_trainer(tmp_path: Path) -> None:
    """Generated wrapper files and base config should use the expected names."""
    target_dir = create_experiment_scaffold("named-demo", root_dir=str(tmp_path))
    package_name = "named_demo"

    assert (
        target_dir / "src" / package_name / "datasets" / "named_demo.py"
    ).exists()
    assert (
        target_dir / "src" / package_name / "models" / "resnet_example.py"
    ).exists()
    assert (target_dir / "src" / package_name / "trainers" / "named_demo.py").exists()

    config = yaml.safe_load((target_dir / "configs" / "base.yaml").read_text())
    assert list(config["models"].keys()) == ["resnet_example"]
    assert config["models"]["resnet_example"]["name"] == "resnet_example"
    assert config["dataset"]["name"] == package_name
    assert list(config["trainer"].keys()) == [package_name]
    assert config["trainer"][package_name]["name"] == package_name

    sweep_config = yaml.safe_load(
        (target_dir / "configs" / "base_sweep.yaml").read_text()
    )
    assert list(sweep_config["fixed"]["trainer"].keys()) == [package_name]
    assert sweep_config["fixed"]["trainer"][package_name]["name"] == package_name


def test_with_azure_scaffold_imports_adapter(tmp_path: Path) -> None:
    """Azure-enabled scaffolds should import the Azure adapter package."""
    target_dir = create_experiment_scaffold(
        "azure-demo",
        root_dir=str(tmp_path),
        with_azure=True,
    )
    package_init = (target_dir / "src" / "azure_demo" / "__init__.py").read_text()

    assert "import dl_mobai_azure" in package_init


def test_scaffold_without_name_initializes_root_dir_in_place(tmp_path: Path) -> None:
    """Omitting --name should initialize the provided directory in place."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "configs" / "base.yaml").exists()
    assert (
        created_dir / "src" / "custom_test" / "datasets" / "custom_test.py"
    ).exists()

    config = yaml.safe_load((created_dir / "configs" / "base.yaml").read_text())
    assert config["experiment"]["name"] == "custom-test"
    assert config["dataset"]["name"] == "custom_test"


def test_cli_allows_missing_name_with_root_dir(tmp_path: Path) -> None:
    """CLI should accept --root-dir without requiring --name."""
    target_dir = tmp_path / "cli_target"
    target_dir.mkdir()

    exit_code = init_main(["--root-dir", str(target_dir)])

    assert exit_code == 0
    assert (target_dir / "pyproject.toml").exists()
