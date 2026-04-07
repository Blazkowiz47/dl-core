"""Tests for local component scaffold generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dl_core.component_scaffold as component_scaffold
from dl_core import load_builtin_components, load_local_components
from dl_core.cli import main as cli_main
from dl_core.core import (
    AUGMENTATION_REGISTRY,
    CALLBACK_REGISTRY,
    CRITERION_REGISTRY,
    EXECUTOR_REGISTRY,
    METRIC_MANAGER_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SAMPLER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
)
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
    component_text = component_path.read_text()
    assert '@register_augmentation(["custom1", "Custom1"])' in component_text
    assert "def _create_train_transforms(self) -> A.Compose:" in component_text
    assert "def _create_test_transforms(self) -> A.Compose:" in component_text
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


def test_cli_add_callback_defaults_to_plain_base(tmp_path: Path) -> None:
    """Callback scaffolds should default to the plain Callback base class."""
    target_dir = create_experiment_scaffold("callback-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "callback",
            "ArtifactLogger",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "callbacks" / "artifactlogger.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import config_field" in component_text
    assert "from dl_core.core import register_callback" in component_text
    assert "from dl_core.core import Callback" in component_text
    assert "CONFIG_FIELDS = Callback.CONFIG_FIELDS + [" in component_text
    assert "class ArtifactLoggerCallback(Callback):" in component_text
    assert "MetricLoggerCallback" not in component_text


def test_cli_add_callback_supports_registered_base(tmp_path: Path) -> None:
    """Callback scaffolds should support explicit built-in bases via --base."""
    target_dir = create_experiment_scaffold(
        "callback-base-demo",
        root_dir=str(tmp_path),
    )

    exit_code = cli_main(
        [
            "add",
            "callback",
            "MetricMirror",
            "--base",
            "metric_logger",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "callbacks" / "metricmirror.py"
    component_text = component_path.read_text()

    assert (
        "from dl_core.callbacks.metric_logger import MetricLoggerCallback"
        in component_text
    )
    assert "CONFIG_FIELDS = MetricLoggerCallback.CONFIG_FIELDS + [" in component_text
    assert "class MetricMirrorCallback(MetricLoggerCallback):" in component_text

    load_builtin_components()
    spec = importlib.util.spec_from_file_location("metricmirror_scaffold", component_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert CALLBACK_REGISTRY.get_class("metricmirror").__name__ == (
        "MetricMirrorCallback"
    )


def test_cli_add_metric_manager_defaults_to_plain_base(tmp_path: Path) -> None:
    """Metric manager scaffolds should default to BaseMetricManager."""
    target_dir = create_experiment_scaffold(
        "metric-manager-demo",
        root_dir=str(tmp_path),
    )

    exit_code = cli_main(
        [
            "add",
            "metric_manager",
            "PadManager",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "metric_managers" / "padmanager.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import BaseMetricManager" in component_text
    assert "class PadManagerMetricManager(BaseMetricManager):" in component_text
    assert "def setup_metrics(self) -> None:" in component_text
    assert "StandardMetricManager" not in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert METRIC_MANAGER_REGISTRY.get_class("padmanager").__name__ == (
        "PadManagerMetricManager"
    )


def test_cli_add_metric_defaults_to_editable_compute_template(tmp_path: Path) -> None:
    """Metric scaffolds should start with a pure compute method."""
    target_dir = create_experiment_scaffold("metric-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "metric",
            "eer_metric",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "metrics" / "eer_metric.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import BaseMetric" in component_text
    assert "def compute(" in component_text
    assert "np.ndarray" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert METRIC_REGISTRY.get_class("eer_metric").__name__ == "EerMetricMetric"


def test_cli_add_criterion_defaults_to_compute_loss_template(tmp_path: Path) -> None:
    """Criterion scaffolds should start with compute_loss guidance."""
    target_dir = create_experiment_scaffold("criterion-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "criterion",
            "my_loss",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "criterions" / "my_loss.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import BaseCriterion" in component_text
    assert "def compute_loss(" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert CRITERION_REGISTRY.get_class("my_loss").__name__ == "MyLossCriterion"


def test_cli_add_model_defaults_to_compute_forward_template(tmp_path: Path) -> None:
    """Model scaffolds should start with compute_forward guidance."""
    target_dir = create_experiment_scaffold("model-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "model",
            "toy_classifier",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "models" / "toy_classifier.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import BaseModel" in component_text
    assert "def compute_forward(" in component_text
    assert "'probabilities': ..., 'logits': ..., 'features': ..." in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert MODEL_REGISTRY.get_class("toy_classifier").__name__ == (
        "ToyClassifierModel"
    )


def test_cli_add_executor_defaults_to_lifecycle_template(tmp_path: Path) -> None:
    """Executor scaffolds should expose setup/execute/teardown hooks."""
    target_dir = create_experiment_scaffold("executor-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "executor",
            "batch_debug",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "executors" / "batch_debug.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import BaseExecutor" in component_text
    assert "def setup(self, total_runs: int) -> None:" in component_text
    assert "def execute_run(self, run_index: int, config_path: Path)" in component_text
    assert "def teardown(self) -> None:" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert EXECUTOR_REGISTRY.get_class("batch_debug").__name__ == (
        "BatchDebugExecutor"
    )


def test_cli_add_optimizer_defaults_to_plain_base(tmp_path: Path) -> None:
    """Optimizer scaffolds should default to the plain Optimizer base class."""
    target_dir = create_experiment_scaffold("optimizer-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "optimizer",
            "MyOptimizer",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "optimizers" / "myoptimizer.py"
    component_text = component_path.read_text()

    assert "from torch.optim import Optimizer" in component_text
    assert "class MyOptimizerOptimizer(Optimizer):" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert OPTIMIZER_REGISTRY.get_class("myoptimizer").__name__ == (
        "MyOptimizerOptimizer"
    )


def test_cli_add_scheduler_defaults_to_plain_base(tmp_path: Path) -> None:
    """Scheduler scaffolds should default to the plain LRScheduler base class."""
    target_dir = create_experiment_scaffold("scheduler-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "scheduler",
            "MyScheduler",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "schedulers" / "myscheduler.py"
    component_text = component_path.read_text()

    assert "from torch.optim.lr_scheduler import LRScheduler" in component_text
    assert "class MySchedulerScheduler(LRScheduler):" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert SCHEDULER_REGISTRY.get_class("myscheduler").__name__ == (
        "MySchedulerScheduler"
    )


def test_supported_trainer_scaffold_bases_are_available() -> None:
    """Trainer scaffolds should expose the built-in peer base flavors."""

    assert component_scaffold.list_supported_trainer_bases() == [
        "epochtrainer",
        "nlptrainer",
        "acttrainer",
    ]


def test_cli_add_trainer_defaults_to_epoch_trainer(tmp_path: Path) -> None:
    """Trainer scaffolds should default to the epoch-based trainer base."""
    target_dir = create_experiment_scaffold("trainer-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "trainer",
            "ArcVein",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "trainers" / "arcvein.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import EpochTrainer" in component_text
    assert "class ArcVeinTrainer(EpochTrainer):" in component_text
    assert "def setup_model(self) -> None:" in component_text
    assert "def train_step(" in component_text
    assert "def validation_step(" in component_text

    load_builtin_components()
    load_local_components(target_dir / "configs" / "base.yaml")

    assert TRAINER_REGISTRY.get_class("arcvein").__name__ == "ArcVeinTrainer"


def test_cli_add_trainer_supports_explicit_epoch_base(tmp_path: Path) -> None:
    """Trainer scaffolds should accept the explicit epochtrainer base."""
    target_dir = create_experiment_scaffold(
        "trainer-epoch-demo",
        root_dir=str(tmp_path),
    )

    exit_code = cli_main(
        [
            "add",
            "trainer",
            "EpochArc",
            "--base",
            "epochtrainer",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "trainers" / "epocharc.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import EpochTrainer" in component_text
    assert "class EpochArcTrainer(EpochTrainer):" in component_text
    assert "def train_step(" in component_text


def test_cli_add_trainer_supports_nlp_base(tmp_path: Path) -> None:
    """Trainer scaffolds should support the sequence-oriented NLP base."""
    target_dir = create_experiment_scaffold(
        "trainer-nlp-demo",
        root_dir=str(tmp_path),
    )

    exit_code = cli_main(
        [
            "add",
            "trainer",
            "TextFlow",
            "--base",
            "nlptrainer",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "trainers" / "textflow.py"
    component_text = component_path.read_text()

    assert "from dl_core.core import SequenceTrainer" in component_text
    assert "from dl_core.core import SequenceStepOutput, register_trainer" in (
        component_text
    )
    assert "class TextFlowTrainer(SequenceTrainer):" in component_text
    assert "def sequence_train_step(" in component_text
    assert "-> SequenceStepOutput:" in component_text


def test_cli_add_trainer_supports_act_base(tmp_path: Path) -> None:
    """Trainer scaffolds should support the adaptive-computation base."""
    target_dir = create_experiment_scaffold(
        "trainer-act-demo",
        root_dir=str(tmp_path),
    )

    exit_code = cli_main(
        [
            "add",
            "trainer",
            "AdaptiveFlow",
            "--base",
            "acttrainer",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    component_path = target_dir / "src" / "trainers" / "adaptiveflow.py"
    component_text = component_path.read_text()

    assert "AdaptiveComputationStepOutput" in component_text
    assert "CarryState" in component_text
    assert "from dl_core.core import (" in component_text
    assert "class AdaptiveFlowTrainer(AdaptiveComputationTrainer):" in (
        component_text
    )
    assert "def adaptive_train_step(" in component_text
    assert "-> AdaptiveComputationStepOutput:" in component_text


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


def test_cli_add_sweep_creates_local_sweep_file(tmp_path: Path) -> None:
    """The add command should create a sweep scaffold with local tracking."""
    target_dir = create_experiment_scaffold("sweep-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "sweep",
            "DebugSweep",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    sweep_path = target_dir / "experiments" / "debugsweep.yaml"
    sweep_text = sweep_path.read_text()

    assert sweep_path.exists()
    assert 'extends_template: "../configs/base_sweep.yaml"' in sweep_text
    assert "fixed:" in sweep_text
    assert "  accelerators: preset:accelerators.cpu" in sweep_text
    assert "  executors: preset:executors.local" in sweep_text
    assert "grid: {}" in sweep_text
    assert "run_name_template: null" in sweep_text
    assert "# experiment_name: my_project" in sweep_text
    assert "Defaults to experiment.name or the" in sweep_text
    assert "# sweep_name: custom_sweep_name" in sweep_text
    assert "backend: local" in sweep_text


def test_cli_add_sweep_supports_tracking_backend(tmp_path: Path) -> None:
    """Sweep scaffolds should support alternate tracking backend blocks."""
    target_dir = create_experiment_scaffold("tracking-demo", root_dir=str(tmp_path))

    exit_code = cli_main(
        [
            "add",
            "sweep",
            "AzureEval",
            "--tracking",
            "azure_mlflow",
            "--root-dir",
            str(target_dir),
        ]
    )

    assert exit_code == 0
    sweep_path = target_dir / "experiments" / "azureeval.yaml"
    sweep_text = sweep_path.read_text()

    assert "backend: azure_mlflow" in sweep_text
    assert "run_name_template: null" in sweep_text
    assert "# experiment_name: my_project" in sweep_text
    assert "Defaults to experiment.name or the" in sweep_text
    assert "# sweep_name: custom_sweep_name" in sweep_text
    assert "project:" not in sweep_text
