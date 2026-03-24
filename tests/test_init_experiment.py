"""Tests for experiment scaffold generation."""

from __future__ import annotations

from pathlib import Path

import yaml

from dl_core.init_experiment import create_experiment_scaffold


def test_scaffold_uses_project_named_dataset_and_trainer(tmp_path: Path) -> None:
    """Generated wrapper files and base config should use the expected names."""
    target_dir = create_experiment_scaffold("named-demo", root_dir=str(tmp_path))
    package_name = "named_demo"

    assert (target_dir / "src" / package_name / "datasets" / "named_demo.py").exists()
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

    sweep_config = yaml.safe_load((target_dir / "configs" / "base_sweep.yaml").read_text())
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
