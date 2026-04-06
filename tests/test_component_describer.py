"""Tests for CLI component description output."""

from __future__ import annotations

import json
from pathlib import Path

from dl_core.cli import main as cli_main
from dl_core.init_experiment import create_experiment_scaffold


def test_cli_list_metric_managers_prints_registered_names(capsys) -> None:
    """The list command should print built-in metric manager names."""
    exit_code = cli_main(["list", "metric_manager"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Registered metric manager" in output
    assert "- standard" in output
    assert "- standard_act" in output


def test_cli_list_supports_json_output(capsys) -> None:
    """Registry listings should be serializable for tooling and agents."""
    exit_code = cli_main(["list", "sampler", "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    listing = json.loads(output)

    assert "sampler" in listing
    assert "attack" in listing["sampler"]


def test_cli_describe_dataset_prints_registered_component_details(
    tmp_path: Path,
    capsys,
) -> None:
    """Dataset descriptions should include registry and class details."""
    target_dir = create_experiment_scaffold("describe-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "ExtraDataset",
            "--root-dir",
            str(target_dir),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli_main(
        [
            "describe",
            "dataset",
            "extradataset",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Target type: dataset" in output
    assert "Requested name: extradataset" in output
    assert "Resolved class: datasets.extradataset.ExtraDatasetDataset" in output
    assert "Registered names: extradataset, ExtraDataset" in output
    assert "Constructor:" in output
    assert "config: dict" in output
    assert "get_file_list" in output
    assert "transform" in output


def test_cli_describe_class_prints_base_class_details(capsys) -> None:
    """Base classes should be describable through an import path."""
    exit_code = cli_main(["describe", "class", "dl_core.core.FrameWrapper"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Target type: class" in output
    assert "Requested name: dl_core.core.FrameWrapper" in output
    assert "Resolved class: dl_core.core.base_dataset.FrameWrapper" in output
    assert "Config fields:" in output
    assert "batch_size" in output
    assert "convert_groups_to_files" in output
    assert "get_video_groups" in output


def test_cli_describe_supports_json_output(tmp_path: Path, capsys) -> None:
    """Descriptions should be serializable for machine-readable tooling."""
    target_dir = create_experiment_scaffold("json-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "TextSet",
            "--base",
            "text_sequence",
            "--root-dir",
            str(target_dir),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = cli_main(
        [
            "describe",
            "dataset",
            "textset",
            "--root-dir",
            str(target_dir),
            "--json",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    description = json.loads(output)

    assert description["component_type"] == "dataset"
    assert description["requested_name"] == "textset"
    assert description["class_name"] == "TextSetDataset"
    assert description["registered_names"] == ["textset", "TextSet"]
    assert any(field["name"] == "batch_size" for field in description["config_fields"])
    assert any(method["name"] == "transform" for method in description["methods"])
