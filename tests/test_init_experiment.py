"""Tests for experiment scaffold generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pytest import MonkeyPatch
import yaml

from dl_core.init_extensions import InitExtension, ScaffoldContext
from dl_core.init_experiment import create_experiment_scaffold, main as init_main


class FakePromptExtension(InitExtension):
    """Small init extension for interactive prompt tests."""

    name = "wandb"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register a tri-state flag pair for the fake extension."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--with-wandb",
            dest="with_wandb",
            action="store_true",
            default=None,
        )
        group.add_argument(
            "--without-wandb",
            dest="with_wandb",
            action="store_false",
            default=None,
        )

    def apply(self, context: ScaffoldContext) -> None:
        """Write a marker file into the scaffold."""
        context.set_file("wandb.txt", "enabled\n")


def test_scaffold_uses_project_named_dataset_and_trainer(tmp_path: Path) -> None:
    """Generated wrapper files and base config should use the expected names."""
    target_dir = create_experiment_scaffold("named-demo", root_dir=str(tmp_path))
    component_name = "named_demo"

    assert (target_dir / "src" / "datasets" / "named_demo.py").exists()
    assert (target_dir / "src" / "models" / "resnet_example.py").exists()
    assert (target_dir / "src" / "trainers" / "named_demo.py").exists()
    assert not (target_dir / "src" / "criterions").exists()
    assert not (target_dir / "src" / "optimizers").exists()
    assert not (target_dir / "src" / "schedulers").exists()
    assert not (target_dir / "presets.yaml").exists()
    assert (target_dir / "configs" / "presets.yaml").exists()
    assert (target_dir / "configs" / "base_sweep.yaml").exists()
    assert (target_dir / "experiments" / "lr_sweep.yaml").exists()
    assert (target_dir / "experiments" / "experiments.log").exists()
    assert (target_dir / "AGENTS.md").exists()
    assert (target_dir / "CLAUDE.md").exists()
    assert (target_dir / "pyrightconfig.json").exists()
    assert (target_dir / "scripts" / "temporary" / "README.md").exists()
    assert (target_dir / "scripts" / "temporary" / "test_dataset.py").exists()
    assert (target_dir / "scripts" / "temporary" / "test_model.py").exists()
    assert not (target_dir / "configs" / "sweeps").exists()

    config = yaml.safe_load((target_dir / "configs" / "base.yaml").read_text())
    base_text = (target_dir / "configs" / "base.yaml").read_text()
    presets = yaml.safe_load((target_dir / "configs" / "presets.yaml").read_text())
    readme_text = (target_dir / "README.md").read_text()
    pyproject_text = (target_dir / "pyproject.toml").read_text()
    helper_readme_text = (
        target_dir / "scripts" / "temporary" / "README.md"
    ).read_text()
    gitignore_text = (target_dir / ".gitignore").read_text()
    assert list(config["models"].keys()) == ["resnet_example"]
    assert "name" not in config["models"]["resnet_example"]
    assert config["dataset"]["name"] == component_name
    assert list(config["trainer"].keys()) == [component_name]
    assert "name" not in config["trainer"][component_name]
    assert config["criterions"]["crossentropy"] is None
    assert "name" not in config["metric_managers"]["standard"]
    assert "name" not in config["runtime"]
    assert "name" not in config["experiment"]
    assert config["deterministic"] is True
    assert "# name: named-demo" in base_text
    assert "# ema:" in base_text
    assert "#   models: [main]" in base_text
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
    assert "name" not in sweep_config["fixed"]["trainer"][component_name]
    assert sweep_config["default_grid"] == {}
    assert "# experiment_name: my_project" in base_sweep_text
    assert "Defaults to experiment.name or the" in base_sweep_text
    assert "# sweep_name: my_custom_sweep" in base_sweep_text
    assert "Optional sweep grouping override. Defaults to the sweep filename." in (
        base_sweep_text
    )

    lr_sweep_text = (target_dir / "experiments" / "lr_sweep.yaml").read_text()
    lr_sweep = yaml.safe_load(lr_sweep_text)
    experiments_log = (target_dir / "experiments" / "experiments.log").read_text()
    agents_text = (target_dir / "AGENTS.md").read_text()
    claude_text = (target_dir / "CLAUDE.md").read_text()
    assert 'extends_template: "../configs/base_sweep.yaml"' in lr_sweep_text
    assert lr_sweep["fixed"]["accelerators"] == "preset:accelerators.cpu"
    assert lr_sweep["fixed"]["executors"] == "preset:executors.local"
    assert lr_sweep["tracking"]["run_name_template"] == "lr_{optimizers.lr}"
    assert "sweep=experiments/lr_sweep.yaml" in experiments_log
    assert "kind=new" in experiments_log
    assert "uv run dl-run --config configs/base.yaml --validate-only" in agents_text
    assert "uv run dl-run --config experiments/<run_name>.yaml" in agents_text
    assert "uv run python scripts/temporary/test_dataset.py" in agents_text
    assert "uv run python scripts/temporary/test_model.py" in agents_text
    assert "uv run dl-sweep experiments/lr_sweep.yaml --dry-run" in agents_text
    assert "uv run dl-analyze --sweep experiments/lr_sweep.yaml" in agents_text
    assert "`experiments/experiments.log` automatically when it exists" in agents_text
    assert "# named-demo Experiment Repository Guidelines" in agents_text
    assert "## Execution Safety" in agents_text
    assert "## Config Rules" in agents_text
    assert "## Code Hygiene" in agents_text
    assert "Even one-off single-run configs belong under `experiments/`." in agents_text
    assert "Do not extract one-off logic into a separate function" in agents_text
    assert "unless the logic is used more than twice" in agents_text
    assert "`CLAUDE.md` should only point at this file with `@AGENTS.md`" in (
        agents_text
    )
    assert "## Sweep Safety Rules" in agents_text
    assert "Never delete `experiments/<sweep_name>/sweep_tracking.json`." in agents_text
    assert "Never run `rm -rf experiments/<sweep_name>` or any equivalent cleanup" in (
        agents_text
    )
    assert "uv run dl-core add dataset ExtraDataset" in agents_text
    assert "uv run dl-core add optimizer MyOptimizer" in agents_text
    assert "uv run dl-core add scheduler MyScheduler" in agents_text
    assert "uv run dl-core describe class dl_core.core.FrameWrapper" in agents_text
    assert "<agent_spec>" not in agents_text
    assert claude_text == "@AGENTS.md\n"
    assert "`CLAUDE.md`: Claude-compatible pointer to `AGENTS.md`" in readme_text
    assert "concrete single-run and sweep experiment configs" in readme_text
    assert '"deep-learning-core>=0.0.33,<0.1"' in pyproject_text
    assert "   - `CLAUDE.md`" in readme_text
    assert "scripts/temporary/test_dataset.py" in readme_text
    assert "scripts/temporary/test_model.py" in readme_text
    assert "uv run python scripts/temporary/test_dataset.py" in helper_readme_text
    assert "uv run python scripts/temporary/test_model.py" in helper_readme_text
    assert ".env\n" in gitignore_text
    assert "!.env.example" in gitignore_text
    assert "mlruns/" in gitignore_text
    assert "wandb/" in gitignore_text
    assert "outputs/" in gitignore_text

def test_scaffold_without_name_initializes_root_dir_in_place(tmp_path: Path) -> None:
    """Omitting --name should initialize the provided directory in place."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "configs" / "base.yaml").exists()
    assert (created_dir / "src" / "datasets" / "custom_test.py").exists()
    assert (created_dir / "src" / "bootstrap.py").exists()

    base_text = (created_dir / "configs" / "base.yaml").read_text()
    config = yaml.safe_load(base_text)
    assert "name" not in config["experiment"]
    assert "# name: custom-test" in base_text
    assert "name" not in config["runtime"]
    assert config["dataset"]["name"] == "custom_test"


def test_cli_allows_missing_name_with_root_dir(tmp_path: Path) -> None:
    """CLI should accept --root-dir without requiring --name."""
    target_dir = tmp_path / "cli_target"
    target_dir.mkdir()

    exit_code = init_main(["--root-dir", str(target_dir)])

    assert exit_code == 0
    assert (target_dir / "pyproject.toml").exists()


def test_cli_prompts_for_discovered_extensions_in_interactive_mode(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Interactive dl-init should prompt for installed extensions."""
    target_dir = tmp_path / "cli_prompt_target"
    target_dir.mkdir()
    monkeypatch.setattr(
        "dl_core.init_experiment.discover_init_extensions",
        lambda: {"wandb": FakePromptExtension()},
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "wandb")

    exit_code = init_main(["--root-dir", str(target_dir)])

    assert exit_code == 0
    assert (target_dir / "wandb.txt").read_text(encoding="utf-8") == "enabled\n"


def test_cli_skips_prompt_when_extension_flag_is_explicit(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Explicit extension flags should bypass the interactive prompt."""
    target_dir = tmp_path / "cli_explicit_target"
    target_dir.mkdir()
    monkeypatch.setattr(
        "dl_core.init_experiment.discover_init_extensions",
        lambda: {"wandb": FakePromptExtension()},
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    def fail_input(_: str) -> str:
        raise AssertionError("input() should not be called for explicit flags")

    monkeypatch.setattr("builtins.input", fail_input)

    exit_code = init_main(
        ["--root-dir", str(target_dir), "--without-wandb"]
    )

    assert exit_code == 0
    assert not (target_dir / "wandb.txt").exists()


def test_scaffold_allows_uv_init_bootstrap_files(tmp_path: Path) -> None:
    """In-place init should preserve existing bootstrap files by default."""
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
    assert (created_dir / "main.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert (created_dir / "uv.lock").read_text(encoding="utf-8") == "version = 1\n"
    assert (created_dir / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname='temp'\n"
    )
    assert (created_dir / "README.md").read_text(encoding="utf-8") == "# temp\n"
    assert (created_dir / "configs" / "base.yaml").exists()


def test_scaffold_allows_existing_agents_and_pyright_files(tmp_path: Path) -> None:
    """In-place init should preserve existing scaffold-owned helper files."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / "AGENTS.md").write_text("temp\n", encoding="utf-8")
    (target_dir / "CLAUDE.md").write_text("custom claude\n", encoding="utf-8")
    (target_dir / "pyrightconfig.json").write_text("{}", encoding="utf-8")

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "AGENTS.md").read_text(encoding="utf-8") == "temp\n"
    assert (created_dir / "CLAUDE.md").read_text(encoding="utf-8") == (
        "custom claude\n"
    )
    assert (created_dir / "pyrightconfig.json").read_text(encoding="utf-8") == "{}"
    assert (created_dir / "configs" / "base.yaml").exists()


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
    """In-place init should leave an existing pyproject untouched by default."""
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


def test_scaffold_allows_unrelated_existing_files(tmp_path: Path) -> None:
    """In-place init should add missing scaffold files without rejecting extras."""
    target_dir = tmp_path / "custom_test"
    target_dir.mkdir()
    (target_dir / "notes.txt").write_text("keep me\n", encoding="utf-8")

    created_dir = create_experiment_scaffold(root_dir=str(target_dir))

    assert created_dir == target_dir.resolve()
    assert (created_dir / "notes.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (created_dir / "configs" / "base.yaml").exists()
