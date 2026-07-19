"""Tests for user-facing YAML configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dl_core import load_builtin_components
from dl_core.utils.config_validator import ConfigValidator


def test_validator_accepts_keyed_components_without_redundant_names(
    tmp_path: Path,
) -> None:
    """Outer component keys should be sufficient for keyed mappings."""
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        (
            "models:\n"
            "  resnet:\n"
            "    variant: resnet18\n"
            "dataset:\n"
            "  classes: [class0, class1]\n"
            "optimizers:\n"
            "  name: adamw\n"
            "schedulers:\n"
            "  name: cosine\n"
            "accelerator:\n"
            "  type: cpu\n"
            "experiment:\n"
            "  description: demo\n"
        ),
        encoding="utf-8",
    )
    validator = ConfigValidator(str(config_path))

    assert validator.validate() is True
    assert validator.errors == []
    assert validator.warnings == []


@pytest.mark.parametrize("content", ["", "- list-root\n", "42\n"])
def test_validator_rejects_non_mapping_yaml_roots(
    tmp_path: Path,
    content: str,
) -> None:
    """Empty, list, and scalar YAML roots should produce validation errors."""
    config_path = tmp_path / "invalid-root.yaml"
    config_path.write_text(content, encoding="utf-8")
    validator = ConfigValidator(str(config_path))

    assert validator.validate() is False
    assert validator.errors == ["Config root must be a YAML mapping"]


def test_validator_reports_invalid_section_types_without_crashing(
    tmp_path: Path,
) -> None:
    """Malformed optional and required sections should be reported together."""
    config_path = tmp_path / "invalid-sections.yaml"
    config_path.write_text(
        (
            "models:\n"
            "  resnet: {}\n"
            "dataset: []\n"
            "optimizers:\n"
            "  name: adamw\n"
            "runtime:\n"
            "experiment: []\n"
        ),
        encoding="utf-8",
    )
    validator = ConfigValidator(str(config_path))

    assert validator.validate() is False
    assert "'dataset' must be a dict" in validator.errors
    assert "Missing required top-level 'accelerator' configuration" in validator.errors
    assert "'experiment' must be a dict" in validator.errors


def test_validator_rejects_non_string_dataset_paths(tmp_path: Path) -> None:
    """Malformed dataset path values should be reported instead of raising."""
    config_path = tmp_path / "invalid-path.yaml"
    config_path.write_text(
        (
            "models:\n"
            "  resnet: {}\n"
            "dataset:\n"
            "  classes: [class0]\n"
            "  train_root: []\n"
            "optimizers:\n"
            "  name: adamw\n"
            "accelerator: cpu\n"
        ),
        encoding="utf-8",
    )
    validator = ConfigValidator(str(config_path))

    assert validator.validate() is False
    assert "'dataset.train_root' must be a path string" in validator.errors


def test_validator_accepts_environment_instead_of_supervised_components_for_rl(
    tmp_path: Path,
) -> None:
    load_builtin_components()
    config_path = tmp_path / "q-learning.yaml"
    config_path.write_text(
        (
            "environment:\n"
            "  name: gymnasium\n"
            "  id: FrozenLake-v1\n"
            "trainer:\n"
            "  q_learning:\n"
            "    total_timesteps: 100\n"
            "accelerator: cpu\n"
        ),
        encoding="utf-8",
    )
    validator = ConfigValidator(str(config_path))

    assert validator.validate() is True
    assert validator.errors == []
