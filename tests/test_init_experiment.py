"""Tests for experiment scaffold generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dl_core.init_experiment import create_experiment_scaffold, main as init_main


def test_scaffold_uses_project_named_dataset_and_trainer(tmp_path: Path) -> None:
    """Generated wrapper files and base config should use the expected names."""
    target_dir = create_experiment_scaffold("named-demo", root_dir=str(tmp_path))
    component_name = "named_demo"

    assert (target_dir / "src" / "datasets" / "named_demo.py").exists()
    assert (target_dir / "src" / "models" / "resnet_example.py").exists()
    assert (target_dir / "src" / "trainers" / "named_demo.py").exists()
    assert not (target_dir / "presets.yaml").exists()
    assert (target_dir / "configs" / "presets.yaml").exists()
    assert (target_dir / "configs" / "base_sweep.yaml").exists()
    assert (target_dir / "experiments" / "lr_sweep.yaml").exists()
    assert (target_dir / "experiments" / "experiments.log").exists()
    assert (target_dir / "AGENTS.md").exists()
    assert (target_dir / "pyrightconfig.json").exists()
    assert not (target_dir / "configs" / "sweeps").exists()

    config = yaml.safe_load((target_dir / "configs" / "base.yaml").read_text())
    presets = yaml.safe_load((target_dir / "configs" / "presets.yaml").read_text())
    assert list(config["models"].keys()) == ["resnet_example"]
    assert config["models"]["resnet_example"]["name"] == "resnet_example"
    assert config["dataset"]["name"] == component_name
    assert list(config["trainer"].keys()) == [component_name]
    assert config["trainer"][component_name]["name"] == component_name
    assert list(presets["accelerators"].keys()) == [
        "cpu",
        "single_gpu",
        "multi_gpu_ddp_2",
        "multi_gpu_ddp_4",
        "multi_gpu_ddp_8",
    ]
    assert presets["accelerators"]["multi_gpu_ddp_4"]["accelerator.devices"] == [
        0,
        1,
        2,
        3,
    ]

    sweep_config = yaml.safe_load(
        (target_dir / "configs" / "base_sweep.yaml").read_text()
    )
    base_sweep_text = (target_dir / "configs" / "base_sweep.yaml").read_text()
    assert list(sweep_config["fixed"]["trainer"].keys()) == [component_name]
    assert sweep_config["fixed"]["trainer"][component_name]["name"] == component_name
    assert sweep_config["default_grid"] == {}
    assert "# sweep_name: my_custom_sweep" in base_sweep_text
    assert "Optional override. Defaults to the sweep filename." in base_sweep_text

    lr_sweep_text = (target_dir / "experiments" / "lr_sweep.yaml").read_text()
    lr_sweep = yaml.safe_load(lr_sweep_text)
    experiments_log = (target_dir / "experiments" / "experiments.log").read_text()
    agents_text = (target_dir / "AGENTS.md").read_text()
    assert 'extends_template: "../configs/base_sweep.yaml"' in lr_sweep_text
    assert lr_sweep["fixed"]["accelerators"] == "preset:accelerators.cpu"
    assert lr_sweep["fixed"]["executors"] == "preset:executors.local"
    assert "sweep=experiments/lr_sweep.yaml" in experiments_log
    assert "kind=new" in experiments_log
    assert "uv run dl-core add dataset ExtraDataset" in agents_text
    assert "uv run dl-core describe class dl_core.core.FrameWrapper" in agents_text

def test_scaffold_without_name_initializes_root_dir_in_place(tmp_path: Path) -> None:
    """Omitting --name should initialize the provided directory in place."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "configs" / "base.yaml").exists()
    assert (created_dir / "src" / "datasets" / "custom_test.py").exists()
    assert (created_dir / "src" / "bootstrap.py").exists()

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


def test_scaffold_allows_uv_init_bootstrap_files(tmp_path: Path) -> None:
    """In-place init should accept and replace a fresh uv-init bootstrap."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (target_dir / ".python-version").write_text("3.11\n", encoding="utf-8")
    (target_dir / "README.md").write_text("# temp\n", encoding="utf-8")
    (target_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (target_dir / "pyproject.toml").write_text(
        "[project]\nname='temp'\n",
        encoding="utf-8",
    )
    (target_dir / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert not (created_dir / "main.py").exists()
    assert not (created_dir / "uv.lock").exists()
    assert (created_dir / "pyproject.toml").exists()
    assert (created_dir / "README.md").exists()


def test_scaffold_allows_existing_agents_and_pyright_files(tmp_path: Path) -> None:
    """In-place init should replace scaffold-owned metadata helper files."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / "AGENTS.md").write_text("temp\n", encoding="utf-8")
    (target_dir / "pyrightconfig.json").write_text("{}", encoding="utf-8")

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert "uv run dl-core describe" in (
        created_dir / "AGENTS.md"
    ).read_text(encoding="utf-8")
    assert '"typeCheckingMode": "basic"' in (
        created_dir / "pyrightconfig.json"
    ).read_text(encoding="utf-8")


def test_scaffold_allows_existing_scripts_and_azure_config(tmp_path: Path) -> None:
    """In-place init should tolerate helper scripts and Azure config files."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / "scripts").mkdir()
    (target_dir / "scripts" / "test.py").write_text("print('ok')\n", encoding="utf-8")
    (target_dir / "azure-config.json").write_text(
        '{"workspace_name": "existing-workspace"}\n',
        encoding="utf-8",
    )
    (target_dir / "pyproject.toml").write_text(
        "[project]\nname='temp'\n",
        encoding="utf-8",
    )

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "scripts" / "test.py").exists()
    assert (created_dir / "azure-config.json").exists()
    assert (created_dir / "configs" / "base.yaml").exists()


def test_scaffold_preserves_existing_tool_uv_config(tmp_path: Path) -> None:
    """In-place init should keep existing uv index and source settings."""
    target_dir = tmp_path / "consumer_repo"
    target_dir.mkdir()
    (target_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "consumer-repo"',
                'version = "0.1.0"',
                "",
                "[tool.uv.sources]",
                '"deep-learning-core" = { index = "testpypi" }',
                "",
                "[[tool.uv.index]]",
                'name = "testpypi"',
                'url = "https://test.pypi.org/simple/"',
                "explicit = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))
    pyproject_text = (created_dir / "pyproject.toml").read_text(encoding="utf-8")

    assert '[tool.uv.sources]' in pyproject_text
    assert '"deep-learning-core" = { index = "testpypi" }' in pyproject_text
    assert '[[tool.uv.index]]' in pyproject_text
    assert 'url = "https://test.pypi.org/simple/"' in pyproject_text


def test_scaffold_rejects_unexpected_existing_files(tmp_path: Path) -> None:
    """In-place init should still reject directories with unrelated files."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / "notes.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_experiment_scaffold(root_dir=str(target_dir))
