"""Tests for local component scaffold generation."""

from __future__ import annotations

from pathlib import Path

import dl_core.component_scaffold as component_scaffold
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
    component_path = target_dir / "src" / "augmentations" / "custom1.py"
    assert component_path.exists()
    assert '@register_augmentation(["custom1", "Custom1"])' in (
        component_path.read_text()
    )
    init_text = (target_dir / "src" / "augmentations" / "__init__.py").read_text()
    assert "from .custom1 import Custom1Augmentation" in init_text
    assert '"Custom1Augmentation"' in init_text

    load_builtin_components()
    imported_modules = load_local_components(target_dir / "configs" / "base.yaml")

    assert "augmentations" in imported_modules
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
    sampler_dir = target_dir / "src" / "samplers"
    assert (sampler_dir / "__init__.py").exists()
    assert (sampler_dir / "passthrough1.py").exists()

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert SAMPLER_REGISTRY.get_class("passthrough1").__name__ == (
        "Passthrough1Sampler"
    )


def test_cli_add_dataset_defaults_to_base_wrapper(tmp_path: Path) -> None:
    """Dataset scaffolds should default to the plain BaseWrapper contract."""
    target_dir = create_experiment_scaffold("dataset-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "LocalBase",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "datasets" / "localbase.py"
    component_text = component_path.read_text()

    assert 'class LocalBaseDataset(BaseWrapper):' in component_text
    assert "def get_file_list(self, split: str)" in component_text
    assert "def transform(self, file_dict: dict[str, Any], split: str)" in (
        component_text
    )
    assert "Thin local wrapper around the built-in standard dataset." not in (
        component_text
    )

    init_text = (target_dir / "src" / "datasets" / "__init__.py").read_text()
    assert "from .localbase import LocalBaseDataset" in init_text
    assert '"LocalBaseDataset"' in init_text


def test_cli_add_frame_dataset_uses_frame_wrapper(tmp_path: Path) -> None:
    """Frame dataset scaffolds should expose the grouped-frame contract."""
    target_dir = create_experiment_scaffold("frame-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "frame_set",
            "--base",
            "frame",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "datasets" / "frame_set.py"
    component_text = component_path.read_text()

    assert 'class FrameSetDataset(FrameWrapper):' in component_text
    assert "def get_video_groups(self, split: str)" in component_text
    assert "def convert_groups_to_files(" in component_text
    assert "def transform(self, file_dict: dict[str, Any], split: str)" in (
        component_text
    )


def test_cli_add_text_sequence_dataset_uses_text_sequence_wrapper(
    tmp_path: Path,
) -> None:
    """Text dataset scaffolds should expose the tokenized sequence contract."""

    target_dir = create_experiment_scaffold("text-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "text_set",
            "--base",
            "text-sequence",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "datasets" / "text_set.py"
    component_text = component_path.read_text()

    assert 'class TextSetDataset(TextSequenceWrapper):' in component_text
    assert "def get_file_list(self, split: str)" in component_text
    assert "def transform(self, file_dict: dict[str, Any], split: str)" in (
        component_text
    )
    assert "input_ids" in component_text


def test_cli_add_adaptive_dataset_uses_adaptive_base(tmp_path: Path) -> None:
    """Adaptive dataset scaffolds should expose class-stream helper guidance."""

    target_dir = create_experiment_scaffold("adaptive-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "act_set",
            "--base",
            "adaptive-computation",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "datasets" / "act_set.py"
    component_text = component_path.read_text()

    assert 'class ActSetDataset(AdaptiveComputationDataset):' in component_text
    assert "AdaptiveComputationDataset will group them per class for you." in (
        component_text
    )


def test_cli_add_dataset_supports_optional_azure_bases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Azure dataset bases should be accepted when an extension exposes them."""
    target_dir = create_experiment_scaffold("azure-demo", root_dir=str(tmp_path))

    monkeypatch.setattr(
        component_scaffold,
        "_load_optional_dataset_base_specs",
        lambda: {
            "azure_compute_multiframe": component_scaffold.DatasetBaseSpec(
                canonical_name="azure_compute_multiframe",
                import_path="dl_azure.datasets",
                base_class="AzureComputeMultiFrameWrapper",
                class_docstring=(
                    "Dataset scaffold based on AzureComputeMultiFrameWrapper."
                ),
                template_kind="multiframe",
            )
        },
    )

    exit_code = cli_main(
        [
            "add",
            "dataset",
            "azure_seq",
            "--base",
            "azure-compute-multiframe",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "datasets" / "azure_seq.py"
    component_text = component_path.read_text()

    assert "from dl_azure.datasets import AzureComputeMultiFrameWrapper" in (
        component_text
    )
    assert (
        "class AzureSeqDataset(AzureComputeMultiFrameWrapper):" in component_text
    )
    assert "def get_video_groups(self, split: str)" in component_text
    assert "def build_frame_record(" in component_text


def test_cli_add_component_updates_existing_package_exports(tmp_path: Path) -> None:
    """New component scaffolds should extend the package-level re-exports."""

    target_dir = create_experiment_scaffold("exports-demo", root_dir=str(tmp_path))

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
    init_text = (target_dir / "src" / "datasets" / "__init__.py").read_text()

    assert "from .exports_demo import ExportsDemoDataset" in init_text
    assert "from .extradataset import ExtraDatasetDataset" in init_text
    assert '"ExportsDemoDataset"' in init_text
    assert '"ExtraDatasetDataset"' in init_text
