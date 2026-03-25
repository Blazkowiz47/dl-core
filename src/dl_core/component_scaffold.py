"""Helpers for generating local experiment component scaffolds."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import re
from pathlib import Path

from dl_core.project import find_local_component_root_dir, find_project_root


@dataclass(frozen=True)
class ComponentSpec:
    """Configuration for a scaffoldable local component type."""

    canonical_name: str
    package_dir: str
    class_suffix: str
    init_docstring: str


@dataclass(frozen=True)
class DatasetBaseSpec:
    """Configuration for a scaffoldable dataset base class."""

    canonical_name: str
    import_path: str
    base_class: str
    class_docstring: str
    template_kind: str
    requires_file_extensions: bool = False


_COMPONENT_SPECS = {
    "augmentation": ComponentSpec(
        canonical_name="augmentation",
        package_dir="augmentations",
        class_suffix="Augmentation",
        init_docstring="Local augmentation extensions.",
    ),
    "callback": ComponentSpec(
        canonical_name="callback",
        package_dir="callbacks",
        class_suffix="Callback",
        init_docstring="Local callback extensions.",
    ),
    "criterion": ComponentSpec(
        canonical_name="criterion",
        package_dir="criterions",
        class_suffix="Criterion",
        init_docstring="Local criterion extensions.",
    ),
    "dataset": ComponentSpec(
        canonical_name="dataset",
        package_dir="datasets",
        class_suffix="Dataset",
        init_docstring="Local dataset extensions.",
    ),
    "executor": ComponentSpec(
        canonical_name="executor",
        package_dir="executors",
        class_suffix="Executor",
        init_docstring="Local executor extensions.",
    ),
    "metric": ComponentSpec(
        canonical_name="metric",
        package_dir="metrics",
        class_suffix="Metric",
        init_docstring="Local metric extensions.",
    ),
    "metric_manager": ComponentSpec(
        canonical_name="metric_manager",
        package_dir="metric_managers",
        class_suffix="MetricManager",
        init_docstring="Local metric manager extensions.",
    ),
    "model": ComponentSpec(
        canonical_name="model",
        package_dir="models",
        class_suffix="Model",
        init_docstring="Local model extensions.",
    ),
    "sampler": ComponentSpec(
        canonical_name="sampler",
        package_dir="samplers",
        class_suffix="Sampler",
        init_docstring="Local sampler extensions.",
    ),
    "trainer": ComponentSpec(
        canonical_name="trainer",
        package_dir="trainers",
        class_suffix="Trainer",
        init_docstring="Local trainer extensions.",
    ),
}

_COMPONENT_TYPE_ALIASES = {
    "augmentation": "augmentation",
    "augmentations": "augmentation",
    "callback": "callback",
    "callbacks": "callback",
    "criterion": "criterion",
    "criterions": "criterion",
    "dataset": "dataset",
    "datasets": "dataset",
    "executor": "executor",
    "executors": "executor",
    "metric": "metric",
    "metrics": "metric",
    "metricmanager": "metric_manager",
    "metric_manager": "metric_manager",
    "metric_managers": "metric_manager",
    "metricmanagers": "metric_manager",
    "model": "model",
    "models": "model",
    "sampler": "sampler",
    "samplers": "sampler",
    "trainer": "trainer",
    "trainers": "trainer",
}

_CORE_DATASET_BASE_SPECS = {
    "base": DatasetBaseSpec(
        canonical_name="base",
        import_path="dl_core.core.base_dataset",
        base_class="BaseWrapper",
        class_docstring="Dataset scaffold based on BaseWrapper.",
        template_kind="sample",
        requires_file_extensions=True,
    ),
    "frame": DatasetBaseSpec(
        canonical_name="frame",
        import_path="dl_core.core.base_dataset",
        base_class="FrameWrapper",
        class_docstring="Dataset scaffold based on FrameWrapper.",
        template_kind="frame",
        requires_file_extensions=True,
    ),
}

_DATASET_BASE_ALIASES = {
    "base": "base",
    "base_wrapper": "base",
    "basewrapper": "base",
    "frame": "frame",
    "frame_wrapper": "frame",
    "framewrapper": "frame",
}


def list_supported_component_types() -> list[str]:
    """Return the canonical component types supported by the scaffold CLI."""
    return sorted(_COMPONENT_SPECS.keys())


def list_supported_dataset_bases() -> list[str]:
    """Return dataset scaffold bases supported in the current environment."""
    return list(_dataset_base_specs().keys())


def normalize_component_type(component_type: str) -> str:
    """Normalize a user-provided component type string."""
    key = component_type.strip().lower().replace("-", "_").replace(" ", "_")
    canonical_name = _COMPONENT_TYPE_ALIASES.get(key)
    if canonical_name is None:
        supported = ", ".join(list_supported_component_types())
        raise ValueError(
            f"Unsupported component type '{component_type}'. Supported types: "
            f"{supported}"
        )
    return canonical_name


def normalize_dataset_base(dataset_base: str | None) -> str:
    """Normalize a user-provided dataset base string."""
    if dataset_base is None:
        return "base"

    key = dataset_base.strip().lower().replace("-", "_").replace(" ", "_")
    canonical_name = _DATASET_BASE_ALIASES.get(key, key)
    if canonical_name not in _dataset_base_specs():
        supported = ", ".join(list_supported_dataset_bases())
        raise ValueError(
            f"Unsupported dataset base '{dataset_base}'. Supported bases: "
            f"{supported}"
        )
    return canonical_name


def create_component_scaffold(
    component_type: str,
    name: str,
    root_dir: str = ".",
    dataset_base: str | None = None,
    force: bool = False,
) -> Path:
    """Create a new local component scaffold inside an experiment repository."""
    canonical_type = normalize_component_type(component_type)
    spec = _COMPONENT_SPECS[canonical_type]
    canonical_dataset_base = _resolve_dataset_base(canonical_type, dataset_base)

    search_root = Path(root_dir).resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        raise FileNotFoundError(
            "Could not find an experiment repository from the provided root. "
            "Run this inside a repository created by dl-init-experiment or pass "
            "--root-dir."
        )

    component_root_dir = find_local_component_root_dir(project_root)
    package_component_dir = component_root_dir / spec.package_dir
    package_component_dir.mkdir(parents=True, exist_ok=True)

    init_path = package_component_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            f'"""{spec.init_docstring}"""\n',
            encoding="utf-8",
        )

    module_name = _to_module_name(name)
    class_name = f"{_to_class_name(name)}{spec.class_suffix}"
    registry_names = _registry_names(name, module_name)
    component_path = package_component_dir / f"{module_name}.py"

    if component_path.exists() and not force:
        raise FileExistsError(f"Component already exists: {component_path}")

    component_path.write_text(
        _render_component(
            spec,
            registry_names,
            class_name,
            dataset_base=canonical_dataset_base,
        ),
        encoding="utf-8",
    )
    return component_path


def _slugify(value: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    if not normalized:
        raise ValueError("Name must contain at least one alphanumeric character")
    return normalized


def _to_module_name(value: str) -> str:
    """Convert text to a Python module name."""
    module_name = _slugify(value).replace("-", "_")
    if module_name[0].isdigit():
        module_name = f"exp_{module_name}"
    return module_name


def _to_class_name(value: str) -> str:
    """Convert text to a PascalCase class-friendly name."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.strip())
    parts = re.split(r"[^a-zA-Z0-9]+|_", normalized)
    class_name = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not class_name:
        raise ValueError("Name must contain at least one alphanumeric character")
    if class_name[0].isdigit():
        class_name = f"Exp{class_name}"
    return class_name


def _registry_names(name: str, module_name: str) -> list[str]:
    """Build registry aliases for the generated component."""
    raw_name = name.strip()
    names = [module_name]
    if raw_name and raw_name != module_name:
        names.append(raw_name)
    return names


def _render_component(
    spec: ComponentSpec,
    registry_names: list[str],
    class_name: str,
    *,
    dataset_base: str | None,
) -> str:
    """Render the generated component source code."""
    registry_literal = _registry_literal(registry_names)

    if spec.canonical_name == "augmentation":
        return _wrapper_component(
            module_docstring="Local augmentation scaffold.",
            register_name="register_augmentation",
            registry_literal=registry_literal,
            import_path="dl_core.augmentations.minimal",
            base_class="MinimalTransform",
            class_name=class_name,
            class_docstring=(
                "Thin local wrapper around the built-in minimal augmentation."
            ),
        )
    if spec.canonical_name == "callback":
        return _wrapper_component(
            module_docstring="Local callback scaffold.",
            register_name="register_callback",
            registry_literal=registry_literal,
            import_path="dl_core.callbacks.metric_logger",
            base_class="MetricLoggerCallback",
            class_name=class_name,
            class_docstring=(
                "Thin local wrapper around the built-in metric logger callback."
            ),
        )
    if spec.canonical_name == "criterion":
        return _wrapper_component(
            module_docstring="Local criterion scaffold.",
            register_name="register_criterion",
            registry_literal=registry_literal,
            import_path="dl_core.criterions.crossentropy",
            base_class="CrossEntropy",
            class_name=class_name,
            class_docstring=(
                "Thin local wrapper around the built-in cross-entropy criterion."
            ),
        )
    if spec.canonical_name == "dataset":
        if dataset_base is None:
            raise ValueError("Dataset scaffolds require a resolved dataset base.")
        return _render_dataset_component(
            base_spec=_dataset_base_specs()[dataset_base],
            registry_literal=registry_literal,
            class_name=class_name,
        )
    if spec.canonical_name == "executor":
        return _wrapper_component(
            module_docstring="Local executor scaffold.",
            register_name="register_executor",
            registry_literal=registry_literal,
            import_path="dl_core.executors.local",
            base_class="LocalExecutor",
            class_name=class_name,
            class_docstring="Thin local wrapper around the built-in local executor.",
        )
    if spec.canonical_name == "metric":
        return _wrapper_component(
            module_docstring="Local metric scaffold.",
            register_name="register_metric",
            registry_literal=registry_literal,
            import_path="dl_core.metrics.accuracy",
            base_class="AccuracyMetric",
            class_name=class_name,
            class_docstring="Thin local wrapper around the built-in accuracy metric.",
        )
    if spec.canonical_name == "metric_manager":
        return _wrapper_component(
            module_docstring="Local metric manager scaffold.",
            register_name="register_metric_manager",
            registry_literal=registry_literal,
            import_path="dl_core.metric_managers.standard_manager",
            base_class="StandardMetricManager",
            class_name=class_name,
            class_docstring=(
                "Thin local wrapper around the built-in standard metric manager."
            ),
        )
    if spec.canonical_name == "model":
        return _wrapper_component(
            module_docstring="Local model scaffold.",
            register_name="register_model",
            registry_literal=registry_literal,
            import_path="dl_core.models.resnet",
            base_class="ResNet",
            class_name=class_name,
            class_docstring="Thin local wrapper around the built-in ResNet model.",
        )
    if spec.canonical_name == "trainer":
        return _wrapper_component(
            module_docstring="Local trainer scaffold.",
            register_name="register_trainer",
            registry_literal=registry_literal,
            import_path="dl_core.trainers.standard_trainer",
            base_class="StandardTrainer",
            class_name=class_name,
            class_docstring="Thin local wrapper around the built-in standard trainer.",
        )
    if spec.canonical_name == "sampler":
        return _sampler_component(
            registry_literal=registry_literal,
            class_name=class_name,
        )

    raise ValueError(f"Unsupported component type: {spec.canonical_name}")


def _registry_literal(registry_names: list[str]) -> str:
    """Render registry names as a decorator literal."""
    if len(registry_names) == 1:
        return f'"{registry_names[0]}"'
    rendered_names = ", ".join(f'"{name}"' for name in registry_names)
    return f"[{rendered_names}]"


def _wrapper_component(
    *,
    module_docstring: str,
    register_name: str,
    registry_literal: str,
    import_path: str,
    base_class: str,
    class_name: str,
    class_docstring: str,
) -> str:
    """Render a thin wrapper component around a built-in implementation."""
    return f"""\"\"\"{module_docstring}\"\"\"

from dl_core.core import {register_name}
from {import_path} import {base_class}


@{register_name}({registry_literal})
class {class_name}({base_class}):
    \"\"\"{class_docstring}\"\"\"

    # TODO: override the built-in behavior here when needed.
    pass
"""


def _resolve_dataset_base(
    component_type: str,
    dataset_base: str | None,
) -> str | None:
    """Validate dataset base usage for the requested component type."""
    if component_type != "dataset":
        if dataset_base is not None:
            raise ValueError("--base is only supported for dataset components.")
        return None
    return normalize_dataset_base(dataset_base)


def _dataset_base_specs() -> dict[str, DatasetBaseSpec]:
    """Return the dataset scaffold bases available in this environment."""
    return {
        **_CORE_DATASET_BASE_SPECS,
        **_load_optional_dataset_base_specs(),
    }


def _load_optional_dataset_base_specs() -> dict[str, DatasetBaseSpec]:
    """Return optional dataset bases exposed by installed extension packages."""
    try:
        azure_spec = importlib.util.find_spec("dl_azure")
    except ModuleNotFoundError:
        return {}

    if azure_spec is None:
        return {}

    return {
        "azure_compute": DatasetBaseSpec(
            canonical_name="azure_compute",
            import_path="dl_azure.datasets",
            base_class="AzureComputeWrapper",
            class_docstring="Dataset scaffold based on AzureComputeWrapper.",
            template_kind="sample",
        ),
        "azure_streaming": DatasetBaseSpec(
            canonical_name="azure_streaming",
            import_path="dl_azure.datasets",
            base_class="AzureStreamingWrapper",
            class_docstring="Dataset scaffold based on AzureStreamingWrapper.",
            template_kind="sample",
        ),
        "azure_compute_frame": DatasetBaseSpec(
            canonical_name="azure_compute_frame",
            import_path="dl_azure.datasets",
            base_class="AzureComputeFrameWrapper",
            class_docstring=(
                "Dataset scaffold based on AzureComputeFrameWrapper."
            ),
            template_kind="frame",
        ),
        "azure_streaming_frame": DatasetBaseSpec(
            canonical_name="azure_streaming_frame",
            import_path="dl_azure.datasets",
            base_class="AzureStreamingFrameWrapper",
            class_docstring=(
                "Dataset scaffold based on AzureStreamingFrameWrapper."
            ),
            template_kind="frame",
        ),
        "azure_compute_multiframe": DatasetBaseSpec(
            canonical_name="azure_compute_multiframe",
            import_path="dl_azure.datasets",
            base_class="AzureComputeMultiFrameWrapper",
            class_docstring=(
                "Dataset scaffold based on AzureComputeMultiFrameWrapper."
            ),
            template_kind="multiframe",
        ),
        "azure_streaming_multiframe": DatasetBaseSpec(
            canonical_name="azure_streaming_multiframe",
            import_path="dl_azure.datasets",
            base_class="AzureStreamingMultiFrameWrapper",
            class_docstring=(
                "Dataset scaffold based on AzureStreamingMultiFrameWrapper."
            ),
            template_kind="multiframe",
        ),
    }


def _render_dataset_component(
    *,
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render a dataset scaffold for the selected base class."""
    if base_spec.template_kind == "sample":
        return _dataset_sample_component(base_spec, registry_literal, class_name)
    if base_spec.template_kind == "frame":
        return _dataset_frame_component(base_spec, registry_literal, class_name)
    if base_spec.template_kind == "multiframe":
        return _dataset_multiframe_component(
            base_spec,
            registry_literal,
            class_name,
        )
    raise ValueError(
        f"Unsupported dataset template kind: {base_spec.template_kind}"
    )


def _dataset_module_docstring(base_spec: DatasetBaseSpec) -> str:
    """Render the module docstring for a generated dataset scaffold."""
    base_list = "\n".join(
        f"- {dataset_base}" for dataset_base in list_supported_dataset_bases()
    )
    return (
        f"""Local dataset scaffold based on {base_spec.base_class}.\n\n"""
        f"""Available scaffold bases in this environment:\n{base_list}"""
    )


def _dataset_sample_component(
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render a sample-level dataset scaffold."""
    file_extensions_block = ""
    if base_spec.requires_file_extensions:
        file_extensions_block = '''
    @property
    def file_extensions(self) -> list[str]:
        """Return the file extensions used by this dataset."""
        raise NotImplementedError(
            "TODO: return patterns like ['*.jpg', '*.png'] for this dataset."
        )

'''

    return f'''"""{_dataset_module_docstring(base_spec)}"""

from __future__ import annotations

from typing import Any

from dl_core.core import register_dataset
from {base_spec.import_path} import {base_spec.base_class}


@register_dataset({registry_literal})
class {class_name}({base_spec.base_class}):
    """{base_spec.class_docstring}"""
{file_extensions_block}    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return one record per sample for the requested split."""
        raise NotImplementedError(
            "TODO: scan this split and return records like "
            "{{'path': '...', 'label': 0}}."
        )

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Load one sample and return the model input dictionary."""
        raise NotImplementedError(
            "TODO: load file_dict['path'], apply preprocessing, and return "
            "{{'image': ..., 'label': ..., 'path': ...}}."
        )
'''


def _dataset_frame_component(
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render a frame-level dataset scaffold."""
    file_extensions_block = ""
    if base_spec.requires_file_extensions:
        file_extensions_block = '''
    @property
    def file_extensions(self) -> list[str]:
        """Return the frame file extensions used by this dataset."""
        raise NotImplementedError(
            "TODO: return patterns like ['*.jpg', '*.png'] for frame files."
        )

'''

    return f'''"""{_dataset_module_docstring(base_spec)}"""

from __future__ import annotations

from typing import Any

from dl_core.core import register_dataset
from {base_spec.import_path} import {base_spec.base_class}


@register_dataset({registry_literal})
class {class_name}({base_spec.base_class}):
    """{base_spec.class_docstring}"""
{file_extensions_block}    def get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        """Return grouped frame paths for the requested split."""
        raise NotImplementedError(
            "TODO: return {{dataset_name: {{video_id: [frame_path, ...]}}}}."
        )

    def convert_groups_to_files(
        self,
        video_groups: dict[str, dict[str, list[str]]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Flatten grouped frames into one record per frame."""
        raise NotImplementedError(
            "TODO: convert grouped frames into records with 'path', 'label', "
            "and 'video_id'."
        )

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Load one frame sample and return the model input dictionary."""
        raise NotImplementedError(
            "TODO: load file_dict['path'], apply preprocessing, and return "
            "{{'image': ..., 'label': ..., 'path': ...}}."
        )
'''


def _dataset_multiframe_component(
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render a multiframe dataset scaffold."""
    return f'''"""{_dataset_module_docstring(base_spec)}"""

from __future__ import annotations

from typing import Any

from dl_core.core import register_dataset
from {base_spec.import_path} import {base_spec.base_class}


@register_dataset({registry_literal})
class {class_name}({base_spec.base_class}):
    """{base_spec.class_docstring}"""

    def get_video_groups(self, split: str) -> dict[str, dict[str, list[str]]]:
        """Return grouped frame paths for the requested split."""
        raise NotImplementedError(
            "TODO: return {{dataset_name: {{video_id: [frame_path, ...]}}}}."
        )

    def build_frame_record(
        self,
        frame_path: str,
        dataset_name: str,
        video_id: str,
    ) -> dict[str, Any]:
        """Build the base metadata record for one multiframe sample."""
        raise NotImplementedError(
            "TODO: return a record with at least 'path' and 'label' plus any "
            "metadata needed for multiframe loading."
        )
'''


def _sampler_component(*, registry_literal: str, class_name: str) -> str:
    """Render a simple pass-through sampler scaffold."""
    return f"""\"\"\"Local sampler scaffold.\"\"\"

from dl_core.core import BaseSampler, register_sampler


@register_sampler({registry_literal})
class {class_name}(BaseSampler):
    \"\"\"Simple sampler scaffold that leaves the file list unchanged.\"\"\"

    def sample_data(self, files: list[dict], split: str) -> list[dict]:
        \"\"\"Return the incoming file list unchanged.\"\"\"
        return files
"""
