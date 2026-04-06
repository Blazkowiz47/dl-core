"""Helpers for generating local experiment component scaffolds."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from importlib import import_module
import importlib.util
import re
from pathlib import Path

from dl_core import load_builtin_components, load_local_components
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
    ComponentRegistry,
)
from dl_core.project import find_local_component_root_dir, find_project_root


@dataclass(frozen=True)
class ComponentSpec:
    """Configuration for a scaffoldable local component type."""

    canonical_name: str
    package_dir: str
    class_suffix: str
    init_docstring: str
    register_name: str


@dataclass(frozen=True)
class ComponentBaseSpec:
    """Resolved import metadata for one scaffold base class."""

    canonical_name: str
    import_path: str
    base_class: str
    class_docstring: str


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
        register_name="register_augmentation",
    ),
    "callback": ComponentSpec(
        canonical_name="callback",
        package_dir="callbacks",
        class_suffix="Callback",
        init_docstring="Local callback extensions.",
        register_name="register_callback",
    ),
    "criterion": ComponentSpec(
        canonical_name="criterion",
        package_dir="criterions",
        class_suffix="Criterion",
        init_docstring="Local criterion extensions.",
        register_name="register_criterion",
    ),
    "dataset": ComponentSpec(
        canonical_name="dataset",
        package_dir="datasets",
        class_suffix="Dataset",
        init_docstring="Local dataset extensions.",
        register_name="register_dataset",
    ),
    "executor": ComponentSpec(
        canonical_name="executor",
        package_dir="executors",
        class_suffix="Executor",
        init_docstring="Local executor extensions.",
        register_name="register_executor",
    ),
    "metric": ComponentSpec(
        canonical_name="metric",
        package_dir="metrics",
        class_suffix="Metric",
        init_docstring="Local metric extensions.",
        register_name="register_metric",
    ),
    "metric_manager": ComponentSpec(
        canonical_name="metric_manager",
        package_dir="metric_managers",
        class_suffix="MetricManager",
        init_docstring="Local metric manager extensions.",
        register_name="register_metric_manager",
    ),
    "model": ComponentSpec(
        canonical_name="model",
        package_dir="models",
        class_suffix="Model",
        init_docstring="Local model extensions.",
        register_name="register_model",
    ),
    "optimizer": ComponentSpec(
        canonical_name="optimizer",
        package_dir="optimizers",
        class_suffix="Optimizer",
        init_docstring="Local optimizer extensions.",
        register_name="register_optimizer",
    ),
    "sampler": ComponentSpec(
        canonical_name="sampler",
        package_dir="samplers",
        class_suffix="Sampler",
        init_docstring="Local sampler extensions.",
        register_name="register_sampler",
    ),
    "scheduler": ComponentSpec(
        canonical_name="scheduler",
        package_dir="schedulers",
        class_suffix="Scheduler",
        init_docstring="Local scheduler extensions.",
        register_name="register_scheduler",
    ),
    "trainer": ComponentSpec(
        canonical_name="trainer",
        package_dir="trainers",
        class_suffix="Trainer",
        init_docstring="Local trainer extensions.",
        register_name="register_trainer",
    ),
}

_COMPONENT_BASE_REGISTRIES: dict[str, ComponentRegistry] = {
    "augmentation": AUGMENTATION_REGISTRY,
    "callback": CALLBACK_REGISTRY,
    "criterion": CRITERION_REGISTRY,
    "executor": EXECUTOR_REGISTRY,
    "metric": METRIC_REGISTRY,
    "metric_manager": METRIC_MANAGER_REGISTRY,
    "model": MODEL_REGISTRY,
    "optimizer": OPTIMIZER_REGISTRY,
    "sampler": SAMPLER_REGISTRY,
    "scheduler": SCHEDULER_REGISTRY,
    "trainer": TRAINER_REGISTRY,
}

_DEFAULT_COMPONENT_BASE_SPECS = {
    "augmentation": ComponentBaseSpec(
        canonical_name="augmentation",
        import_path="dl_core.core",
        base_class="BaseTransform",
        class_docstring="Local augmentation scaffold based on BaseTransform.",
    ),
    "callback": ComponentBaseSpec(
        canonical_name="callback",
        import_path="dl_core.core",
        base_class="Callback",
        class_docstring="Local callback scaffold based on Callback.",
    ),
    "criterion": ComponentBaseSpec(
        canonical_name="criterion",
        import_path="dl_core.core",
        base_class="BaseCriterion",
        class_docstring="Local criterion scaffold based on BaseCriterion.",
    ),
    "executor": ComponentBaseSpec(
        canonical_name="executor",
        import_path="dl_core.core",
        base_class="BaseExecutor",
        class_docstring="Local executor scaffold based on BaseExecutor.",
    ),
    "metric": ComponentBaseSpec(
        canonical_name="metric",
        import_path="dl_core.core",
        base_class="BaseMetric",
        class_docstring="Local metric scaffold based on BaseMetric.",
    ),
    "metric_manager": ComponentBaseSpec(
        canonical_name="metric_manager",
        import_path="dl_core.core",
        base_class="BaseMetricManager",
        class_docstring=(
            "Local metric manager scaffold based on BaseMetricManager."
        ),
    ),
    "model": ComponentBaseSpec(
        canonical_name="model",
        import_path="dl_core.core",
        base_class="BaseModel",
        class_docstring="Local model scaffold based on BaseModel.",
    ),
    "optimizer": ComponentBaseSpec(
        canonical_name="optimizer",
        import_path="torch.optim",
        base_class="Optimizer",
        class_docstring="Local optimizer scaffold based on Optimizer.",
    ),
    "sampler": ComponentBaseSpec(
        canonical_name="sampler",
        import_path="dl_core.core",
        base_class="BaseSampler",
        class_docstring="Local sampler scaffold based on BaseSampler.",
    ),
    "scheduler": ComponentBaseSpec(
        canonical_name="scheduler",
        import_path="torch.optim.lr_scheduler",
        base_class="LRScheduler",
        class_docstring="Local scheduler scaffold based on LRScheduler.",
    ),
    "trainer": ComponentBaseSpec(
        canonical_name="trainer",
        import_path="dl_core.core",
        base_class="EpochTrainer",
        class_docstring="Local trainer scaffold based on EpochTrainer.",
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
    "optimizer": "optimizer",
    "optimizers": "optimizer",
    "sampler": "sampler",
    "samplers": "sampler",
    "scheduler": "scheduler",
    "schedulers": "scheduler",
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
    "text_sequence": DatasetBaseSpec(
        canonical_name="text_sequence",
        import_path="dl_core.core.base_dataset",
        base_class="TextSequenceWrapper",
        class_docstring="Dataset scaffold based on TextSequenceWrapper.",
        template_kind="text_sequence",
    ),
    "adaptive_computation": DatasetBaseSpec(
        canonical_name="adaptive_computation",
        import_path="dl_core.core.base_dataset",
        base_class="AdaptiveComputationDataset",
        class_docstring=(
            "Dataset scaffold based on AdaptiveComputationDataset."
        ),
        template_kind="adaptive_computation",
    ),
}

_CORE_TRAINER_BASE_SPECS = {
    "epochtrainer": ComponentBaseSpec(
        canonical_name="epochtrainer",
        import_path="dl_core.core",
        base_class="EpochTrainer",
        class_docstring="Local trainer scaffold based on EpochTrainer.",
    ),
    "nlptrainer": ComponentBaseSpec(
        canonical_name="nlptrainer",
        import_path="dl_core.core",
        base_class="SequenceTrainer",
        class_docstring="Local trainer scaffold based on SequenceTrainer.",
    ),
    "acttrainer": ComponentBaseSpec(
        canonical_name="acttrainer",
        import_path="dl_core.core",
        base_class="AdaptiveComputationTrainer",
        class_docstring=(
            "Local trainer scaffold based on AdaptiveComputationTrainer."
        ),
    ),
}

_DATASET_BASE_ALIASES = {
    "base": "base",
    "base_wrapper": "base",
    "basewrapper": "base",
    "frame": "frame",
    "frame_wrapper": "frame",
    "framewrapper": "frame",
    "text": "text_sequence",
    "text_sequence": "text_sequence",
    "textsequence": "text_sequence",
    "sequence": "text_sequence",
    "sequence_wrapper": "text_sequence",
    "textsequencewrapper": "text_sequence",
    "adaptive": "adaptive_computation",
    "adaptive_computation": "adaptive_computation",
    "adaptivecomputation": "adaptive_computation",
    "adaptive_computation_dataset": "adaptive_computation",
    "adaptivecomputationdataset": "adaptive_computation",
    "act": "adaptive_computation",
}

_TRAINER_BASE_ALIASES = {
    "base_trainer": "epochtrainer",
    "basetrainer": "epochtrainer",
    "epoch": "epochtrainer",
    "epoch_trainer": "epochtrainer",
    "epochtrainer": "epochtrainer",
    "nlp": "nlptrainer",
    "nlp_trainer": "nlptrainer",
    "nlptrainer": "nlptrainer",
    "sequence": "nlptrainer",
    "sequence_trainer": "nlptrainer",
    "sequencetrainer": "nlptrainer",
    "act": "acttrainer",
    "act_trainer": "acttrainer",
    "acttrainer": "acttrainer",
    "adaptive": "acttrainer",
    "adaptive_computation": "acttrainer",
    "adaptive_computation_trainer": "acttrainer",
    "adaptivecomputationtrainer": "acttrainer",
}


def list_supported_component_types() -> list[str]:
    """Return the canonical component types supported by the scaffold CLI."""
    return sorted(_COMPONENT_SPECS.keys())


def list_supported_dataset_bases() -> list[str]:
    """Return dataset scaffold bases supported in the current environment."""
    return list(_dataset_base_specs().keys())


def list_supported_trainer_bases() -> list[str]:
    """Return trainer scaffold bases supported in the current environment."""
    return list(_trainer_base_specs().keys())


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


def normalize_trainer_base(trainer_base: str | None) -> str:
    """Normalize a user-provided trainer base string."""
    if trainer_base is None:
        return "epochtrainer"

    key = _normalize_base_name(trainer_base)
    if key is None:
        return "epochtrainer"

    canonical_name = _TRAINER_BASE_ALIASES.get(key, key)
    if canonical_name not in _trainer_base_specs():
        supported = ", ".join(list_supported_trainer_bases())
        raise ValueError(
            f"Unsupported trainer base '{trainer_base}'. Supported bases: "
            f"{supported}"
        )
    return canonical_name


def create_component_scaffold(
    component_type: str,
    name: str,
    root_dir: str = ".",
    base_name: str | None = None,
    force: bool = False,
) -> Path:
    """Create a new local component scaffold inside an experiment repository."""
    canonical_type = normalize_component_type(component_type)
    spec = _COMPONENT_SPECS[canonical_type]

    search_root = Path(root_dir).resolve()
    project_root = find_project_root(search_root)
    if project_root is None:
        raise FileNotFoundError(
            "Could not find an experiment repository from the provided root. "
            "Run this inside a repository created by dl-init or pass "
            "--root-dir."
        )

    canonical_dataset_base = _resolve_dataset_base(canonical_type, base_name)
    component_base = _resolve_component_base(
        canonical_type,
        base_name,
        project_root,
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
            component_base=component_base,
        ),
        encoding="utf-8",
    )
    _update_package_init(
        init_path,
        module_name,
        class_name,
        default_docstring=spec.init_docstring,
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
    component_base: ComponentBaseSpec | None,
) -> str:
    """Render the generated component source code."""
    registry_literal = _registry_literal(registry_names)

    if spec.canonical_name == "callback":
        return _callback_component(
            import_path=_component_base_required(component_base).import_path,
            base_class=_component_base_required(component_base).base_class,
            registry_literal=registry_literal,
            class_name=class_name,
            class_docstring=_component_base_required(component_base).class_docstring,
        )
    if spec.canonical_name == "trainer":
        trainer_base = _component_base_required(component_base)
        return _trainer_component(
            registry_literal=registry_literal,
            class_name=class_name,
            import_path=trainer_base.import_path,
            base_class=trainer_base.base_class,
            class_docstring=trainer_base.class_docstring,
        )
    if spec.canonical_name == "dataset":
        if dataset_base is None:
            raise ValueError("Dataset scaffolds require a resolved dataset base.")
        return _render_dataset_component(
            base_spec=_dataset_base_specs()[dataset_base],
            registry_literal=registry_literal,
            class_name=class_name,
        )
    if spec.canonical_name == "sampler":
        sampler_base = _component_base_required(component_base)
        if _is_default_component_base(spec.canonical_name, sampler_base):
            return _sampler_component(
                registry_literal=registry_literal,
                class_name=class_name,
                import_path=sampler_base.import_path,
                base_class=sampler_base.base_class,
                class_docstring=sampler_base.class_docstring,
            )
    if spec.canonical_name == "optimizer":
        optimizer_base = _component_base_required(component_base)
        if _is_default_component_base(spec.canonical_name, optimizer_base):
            return _optimizer_component(
                registry_literal=registry_literal,
                class_name=class_name,
                import_path=optimizer_base.import_path,
                base_class=optimizer_base.base_class,
                class_docstring=optimizer_base.class_docstring,
            )
    if spec.canonical_name == "scheduler":
        scheduler_base = _component_base_required(component_base)
        if _is_default_component_base(spec.canonical_name, scheduler_base):
            return _scheduler_component(
                registry_literal=registry_literal,
                class_name=class_name,
                import_path=scheduler_base.import_path,
                base_class=scheduler_base.base_class,
                class_docstring=scheduler_base.class_docstring,
            )

    resolved_base = _component_base_required(component_base)
    return _wrapper_component(
        module_docstring=(
            f"Local {spec.canonical_name.replace('_', ' ')} scaffold."
        ),
        register_name=spec.register_name,
        registry_literal=registry_literal,
        import_path=resolved_base.import_path,
        base_class=resolved_base.base_class,
        class_name=class_name,
        class_docstring=resolved_base.class_docstring,
    )


def _update_package_init(
    init_path: Path,
    module_name: str,
    class_name: str,
    *,
    default_docstring: str,
) -> None:
    """Update a local component package ``__init__`` with the new export."""

    if not init_path.exists():
        init_path.write_text(
            _render_package_init(
                docstring=default_docstring,
                imports_by_module={module_name: {class_name}},
                exported_names=[class_name],
            ),
            encoding="utf-8",
        )
        return

    init_text = init_path.read_text(encoding="utf-8")
    package_init = _parse_package_init(init_text)
    if package_init is None:
        _append_package_export(init_path, init_text, module_name, class_name)
        return

    docstring, imports_by_module, exported_names = package_init
    imports_by_module.setdefault(module_name, set()).add(class_name)
    if class_name not in exported_names:
        exported_names.append(class_name)

    init_path.write_text(
        _render_package_init(
            docstring=docstring or default_docstring,
            imports_by_module=imports_by_module,
            exported_names=exported_names,
        ),
        encoding="utf-8",
    )


def _parse_package_init(
    init_text: str,
) -> tuple[str | None, dict[str, set[str]], list[str]] | None:
    """Parse a simple package ``__init__`` file into exports and imports."""

    try:
        module = ast.parse(init_text)
    except SyntaxError:
        return None

    docstring = ast.get_docstring(module)
    imports_by_module: dict[str, set[str]] = {}
    exported_names: list[str] = []

    for node in module.body:
        if isinstance(node, ast.Expr):
            if (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            return None

        if isinstance(node, ast.ImportFrom):
            if node.level != 1 or node.module is None:
                return None
            imports_by_module.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )
            continue

        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                return None
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            if not isinstance(value, (list, tuple)):
                return None
            exported_names.extend(str(item) for item in value)
            continue

        return None

    return docstring, imports_by_module, exported_names


def _render_package_init(
    *,
    docstring: str,
    imports_by_module: dict[str, set[str]],
    exported_names: list[str],
) -> str:
    """Render a local component package ``__init__`` file."""

    import_blocks: list[str] = []
    for module_name in sorted(imports_by_module):
        imported_names = ", ".join(sorted(imports_by_module[module_name]))
        import_blocks.append(f"from .{module_name} import {imported_names}")

    ordered_exports = sorted(dict.fromkeys(exported_names))
    exports_block = ",\n".join(f'    "{name}"' for name in ordered_exports)

    rendered = f'"""{docstring}"""\n'
    if import_blocks:
        rendered += "\n" + "\n".join(import_blocks) + "\n"
    if ordered_exports:
        rendered += f"\n__all__ = [\n{exports_block},\n]\n"
    return rendered


def _append_package_export(
    init_path: Path,
    init_text: str,
    module_name: str,
    class_name: str,
) -> None:
    """Append a best-effort export when the package init is non-standard."""

    appended_lines: list[str] = []
    import_line = f"from .{module_name} import {class_name}"
    if import_line not in init_text:
        appended_lines.append(import_line)

    if "__all__" in init_text:
        appended_lines.append(f'__all__.append("{class_name}")')
    else:
        appended_lines.append(f'__all__ = ["{class_name}"]')

    updated_text = init_text.rstrip()
    if updated_text:
        updated_text += "\n\n"
    updated_text += "\n".join(appended_lines) + "\n"
    init_path.write_text(updated_text, encoding="utf-8")


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


def _callback_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a callback scaffold with artifact layout guidance."""
    return f"""\"\"\"Local callback scaffold.\"\"\"

from dl_core.core import config_field
from dl_core.core import register_callback
from {import_path} import {base_class}


@register_callback({registry_literal})
class {class_name}({base_class}):
    \"\"\"{class_docstring}\"\"\"

    CONFIG_FIELDS = {base_class}.CONFIG_FIELDS + [
        # TODO: declare callback-specific config keys here, for example:
        # config_field("enabled", "bool", "Enable this callback.", default=True),
    ]

    # TODO: Override the base behavior here when needed.
    # TODO: Store epoch-scoped artifacts under
    # ``self.trainer.artifact_manager.get_epoch_dir(epoch)`` or a more specific
    # helper like ``get_epoch_training_plots_dir(epoch)``.
    # TODO: Store end-of-run artifacts under
    # ``self.trainer.artifact_manager.get_final_dir()``.
    # TODO: The MLflow callback uploads ``epoch_<n>/`` after each epoch and
    # ``final/`` plus ``config.yaml`` at training end.
    pass
"""


def _trainer_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a trainer scaffold tailored to the selected trainer base."""

    if base_class == "SequenceTrainer":
        return _sequence_trainer_component(
            registry_literal=registry_literal,
            class_name=class_name,
            import_path=import_path,
            base_class=base_class,
            class_docstring=class_docstring,
        )
    if base_class == "AdaptiveComputationTrainer":
        return _adaptive_trainer_component(
            registry_literal=registry_literal,
            class_name=class_name,
            import_path=import_path,
            base_class=base_class,
            class_docstring=class_docstring,
        )
    return _epoch_trainer_component(
        registry_literal=registry_literal,
        class_name=class_name,
        import_path=import_path,
        base_class=base_class,
        class_docstring=class_docstring,
    )


def _epoch_trainer_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render an epoch-oriented trainer scaffold."""

    return f'''"""Local trainer scaffold."""

from __future__ import annotations

import torch

from dl_core.core import register_trainer
from {import_path} import {base_class}


@register_trainer({registry_literal})
class {class_name}({base_class}):
    """{class_docstring}"""

    def setup_model(self) -> None:
        """Initialize models and store them in ``self.models``."""
        raise NotImplementedError("TODO: populate self.models in setup_model().")

    def setup_criterion(self) -> None:
        """Initialize criterions and store them in ``self.criterions``."""
        raise NotImplementedError(
            "TODO: populate self.criterions in setup_criterion()."
        )

    def setup_optimizer(self) -> None:
        """Initialize optimizers and store them in ``self.optimizers``."""
        raise NotImplementedError(
            "TODO: populate self.optimizers in setup_optimizer()."
        )

    def setup_scheduler(self) -> None:
        """Initialize schedulers and store them in ``self.schedulers``."""
        raise NotImplementedError(
            "TODO: populate self.schedulers in setup_scheduler()."
        )

    def train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        """Run one training step and return scalar metrics."""
        raise NotImplementedError("TODO: implement train_step().")

    def test_step(self, batch_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one test step and return scalar metrics."""
        raise NotImplementedError("TODO: implement test_step().")

    def validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Run one validation step and return scalar metrics."""
        raise NotImplementedError("TODO: implement validation_step().")
'''


def _sequence_trainer_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a sequence-oriented trainer scaffold."""

    return f'''"""Local sequence trainer scaffold."""

from __future__ import annotations

import torch

from dl_core.core import SequenceStepOutput, register_trainer
from {import_path} import {base_class}


@register_trainer({registry_literal})
class {class_name}({base_class}):
    """{class_docstring}"""

    def setup_model(self) -> None:
        """Initialize models and store them in ``self.models``."""
        raise NotImplementedError("TODO: populate self.models in setup_model().")

    def setup_criterion(self) -> None:
        """Initialize criterions and store them in ``self.criterions``."""
        raise NotImplementedError(
            "TODO: populate self.criterions in setup_criterion()."
        )

    def setup_optimizer(self) -> None:
        """Initialize optimizers and store them in ``self.optimizers``."""
        raise NotImplementedError(
            "TODO: populate self.optimizers in setup_optimizer()."
        )

    def setup_scheduler(self) -> None:
        """Initialize schedulers and store them in ``self.schedulers``."""
        raise NotImplementedError(
            "TODO: populate self.schedulers in setup_scheduler()."
        )

    def sequence_train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> SequenceStepOutput:
        """Run one sequence-specific training step."""
        raise NotImplementedError(
            "TODO: build model inputs with build_sequence_model_inputs(), "
            "compute probabilities, and return SequenceStepOutput(...)."
        )

    def sequence_test_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> SequenceStepOutput:
        """Run one sequence-specific test step."""
        raise NotImplementedError(
            "TODO: implement sequence_test_step() and return "
            "SequenceStepOutput(...)."
        )

    def sequence_validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> SequenceStepOutput:
        """Run one sequence-specific validation step."""
        raise NotImplementedError(
            "TODO: implement sequence_validation_step() and return "
            "SequenceStepOutput(...)."
        )
'''


def _adaptive_trainer_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render an adaptive-computation trainer scaffold."""

    return f'''"""Local adaptive-computation trainer scaffold."""

from __future__ import annotations

from dl_core.core import (
    AdaptiveComputationStepOutput,
    CarryState,
    register_trainer,
)
from {import_path} import {base_class}


@register_trainer({registry_literal})
class {class_name}({base_class}):
    """{class_docstring}"""

    def setup_model(self) -> None:
        """Initialize models and store them in ``self.models``."""
        raise NotImplementedError("TODO: populate self.models in setup_model().")

    def setup_criterion(self) -> None:
        """Initialize criterions and store them in ``self.criterions``."""
        raise NotImplementedError(
            "TODO: populate self.criterions in setup_criterion()."
        )

    def setup_optimizer(self) -> None:
        """Initialize optimizers and store them in ``self.optimizers``."""
        raise NotImplementedError(
            "TODO: populate self.optimizers in setup_optimizer()."
        )

    def setup_scheduler(self) -> None:
        """Initialize schedulers and store them in ``self.schedulers``."""
        raise NotImplementedError(
            "TODO: populate self.schedulers in setup_scheduler()."
        )

    def adaptive_train_step(
        self,
        carry_state: CarryState,
        batch_idx: int,
    ) -> AdaptiveComputationStepOutput:
        """Run one adaptive-computation training step."""
        raise NotImplementedError(
            "TODO: update carry_state, compute probabilities, and return "
            "AdaptiveComputationStepOutput(...)."
        )

    def adaptive_test_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """Run one adaptive-computation test step."""
        raise NotImplementedError(
            "TODO: implement adaptive_test_step() and return "
            "AdaptiveComputationStepOutput(...)."
        )

    def adaptive_validation_step(
        self,
        carry_state: CarryState,
    ) -> AdaptiveComputationStepOutput:
        """Run one adaptive-computation validation step."""
        raise NotImplementedError(
            "TODO: implement adaptive_validation_step() and return "
            "AdaptiveComputationStepOutput(...)."
        )
'''


def _resolve_dataset_base(
    component_type: str,
    dataset_base: str | None,
) -> str | None:
    """Validate dataset base usage for the requested component type."""
    if component_type != "dataset":
        return None
    return normalize_dataset_base(dataset_base)


def _resolve_component_base(
    component_type: str,
    base_name: str | None,
    root_dir: Path,
) -> ComponentBaseSpec | None:
    """Resolve the base class used by a generated scaffold."""
    if component_type == "dataset":
        return None
    if component_type == "trainer":
        return _resolve_trainer_component_base(base_name, root_dir)

    default_base = _DEFAULT_COMPONENT_BASE_SPECS[component_type]
    normalized_base_name = _normalize_base_name(base_name)
    if normalized_base_name in {None, "base"}:
        return default_base
    if normalized_base_name == _normalize_base_name(default_base.base_class):
        return default_base

    if base_name is None:
        return default_base

    if "." in base_name:
        return _resolve_importable_component_base(component_type, base_name)

    load_builtin_components()
    load_local_components(root_dir)
    registry = _COMPONENT_BASE_REGISTRIES[component_type]
    resolved_class = registry.get_class(base_name)
    return _resolve_registered_component_base(component_type, resolved_class)


def _resolve_trainer_component_base(
    base_name: str | None,
    root_dir: Path,
) -> ComponentBaseSpec:
    """Resolve the base class used by a generated trainer scaffold."""
    builtin_base = _resolve_builtin_trainer_base(base_name)
    if builtin_base is not None:
        return builtin_base

    if base_name is None:
        return _DEFAULT_COMPONENT_BASE_SPECS["trainer"]

    if "." in base_name:
        return _resolve_importable_component_base("trainer", base_name)

    load_builtin_components()
    load_local_components(root_dir)
    registry = _COMPONENT_BASE_REGISTRIES["trainer"]
    resolved_class = registry.get_class(base_name)
    return _resolve_registered_component_base("trainer", resolved_class)


def _resolve_builtin_trainer_base(
    base_name: str | None,
) -> ComponentBaseSpec | None:
    """Return a built-in trainer scaffold base when one is requested."""
    if base_name is None:
        return _DEFAULT_COMPONENT_BASE_SPECS["trainer"]

    normalized_base_name = _normalize_base_name(base_name)
    if normalized_base_name is None:
        return _DEFAULT_COMPONENT_BASE_SPECS["trainer"]

    if normalized_base_name == _normalize_base_name(
        _DEFAULT_COMPONENT_BASE_SPECS["trainer"].base_class
    ):
        return _DEFAULT_COMPONENT_BASE_SPECS["trainer"]

    trainer_base_name = _TRAINER_BASE_ALIASES.get(normalized_base_name)
    if trainer_base_name is None:
        return None

    return _trainer_base_specs()[trainer_base_name]


def _normalize_base_name(base_name: str | None) -> str | None:
    """Normalize a scaffold base name for comparison."""
    if base_name is None:
        return None
    return (
        base_name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _resolve_registered_component_base(
    component_type: str,
    resolved_class: type,
) -> ComponentBaseSpec:
    """Build scaffold base metadata from a registered component class."""
    return _build_component_base_spec(component_type, resolved_class)


def _resolve_importable_component_base(
    component_type: str,
    class_path: str,
) -> ComponentBaseSpec:
    """Build scaffold base metadata from an importable class path."""
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected package.module.ClassName."
        )

    module = import_module(module_path)
    resolved_class = getattr(module, class_name, None)
    if not inspect.isclass(resolved_class):
        raise ValueError(
            f"Class '{class_name}' could not be imported from '{module_path}'."
        )

    _validate_component_base_class(component_type, resolved_class)
    return _build_component_base_spec(component_type, resolved_class)


def _build_component_base_spec(
    component_type: str,
    resolved_class: type,
) -> ComponentBaseSpec:
    """Build scaffold base metadata for a resolved class."""
    _validate_component_base_class(component_type, resolved_class)
    class_docstring = inspect.getdoc(resolved_class)
    if not class_docstring:
        class_docstring = (
            "Local "
            f"{component_type.replace('_', ' ')} scaffold based on "
            f"{resolved_class.__name__}."
        )
    return ComponentBaseSpec(
        canonical_name=component_type,
        import_path=resolved_class.__module__,
        base_class=resolved_class.__name__,
        class_docstring=class_docstring,
    )


def _validate_component_base_class(
    component_type: str,
    resolved_class: type,
) -> None:
    """Validate that a resolved base class matches the scaffold type."""
    default_base = _DEFAULT_COMPONENT_BASE_SPECS[component_type]
    base_module = import_module(default_base.import_path)
    expected_base = getattr(base_module, default_base.base_class)
    if not issubclass(resolved_class, expected_base):
        raise ValueError(
            f"Base class '{resolved_class.__module__}.{resolved_class.__name__}' "
            f"is not a subclass of {expected_base.__module__}.{expected_base.__name__}."
        )


def _component_base_required(
    component_base: ComponentBaseSpec | None,
) -> ComponentBaseSpec:
    """Return a resolved component base or raise a helpful error."""
    if component_base is None:
        raise ValueError("A resolved scaffold base is required for this component.")
    return component_base


def _is_default_component_base(
    component_type: str,
    component_base: ComponentBaseSpec,
) -> bool:
    """Return whether a resolved scaffold base matches the default base."""
    default_base = _DEFAULT_COMPONENT_BASE_SPECS[component_type]
    return (
        component_base.import_path == default_base.import_path
        and component_base.base_class == default_base.base_class
    )


def _dataset_base_specs() -> dict[str, DatasetBaseSpec]:
    """Return the dataset scaffold bases available in this environment."""
    return {
        **_CORE_DATASET_BASE_SPECS,
        **_load_optional_dataset_base_specs(),
    }


def _trainer_base_specs() -> dict[str, ComponentBaseSpec]:
    """Return the trainer scaffold bases available in the current environment."""
    return dict(_CORE_TRAINER_BASE_SPECS)


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
    if base_spec.template_kind == "text_sequence":
        return _dataset_text_sequence_component(
            base_spec,
            registry_literal,
            class_name,
        )
    if base_spec.template_kind == "adaptive_computation":
        return _dataset_adaptive_component(
            base_spec,
            registry_literal,
            class_name,
        )
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


def _dataset_text_sequence_component(
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render a text-sequence dataset scaffold."""
    return f'''"""{_dataset_module_docstring(base_spec)}"""

from __future__ import annotations

from typing import Any

from dl_core.core import register_dataset
from {base_spec.import_path} import {base_spec.base_class}


@register_dataset({registry_literal})
class {class_name}({base_spec.base_class}):
    """{base_spec.class_docstring}"""

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return one record per text sample for the requested split."""
        raise NotImplementedError(
            "TODO: return records like {{'path': '...', 'text': '...', "
            "'label': 0}}."
        )

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Tokenize one text sample and return the model input dictionary."""
        raise NotImplementedError(
            "TODO: tokenize file_dict['text'] and return keys like "
            "{{'input_ids': ..., 'attention_mask': ..., 'label': ..., "
            "'path': ...}}."
        )
'''


def _dataset_adaptive_component(
    base_spec: DatasetBaseSpec,
    registry_literal: str,
    class_name: str,
) -> str:
    """Render an adaptive-computation dataset scaffold."""
    return f'''"""{_dataset_module_docstring(base_spec)}"""

from __future__ import annotations

from typing import Any

from dl_core.core import register_dataset
from {base_spec.import_path} import {base_spec.base_class}


@register_dataset({registry_literal})
class {class_name}({base_spec.base_class}):
    """{base_spec.class_docstring}"""

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return one record per sample for the requested split."""
        raise NotImplementedError(
            "TODO: return records like {{'path': '...', 'label': 0}}. "
            "AdaptiveComputationDataset will group them per class for you."
        )

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Load one sample and return the model input dictionary."""
        raise NotImplementedError(
            "TODO: load file_dict['path'] and return the tensors needed by "
            "your adaptive-computation trainer."
        )
'''


def _sampler_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a simple pass-through sampler scaffold."""
    return f"""\"\"\"Local sampler scaffold.\"\"\"

from dl_core.core import register_sampler
from {import_path} import {base_class}


@register_sampler({registry_literal})
class {class_name}({base_class}):
    \"\"\"{class_docstring}\"\"\"

    def sample_data(self, files: list[dict], split: str) -> list[dict]:
        \"\"\"Return the incoming file list unchanged.\"\"\"
        return files
"""


def _optimizer_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a minimal optimizer scaffold."""
    return f'''"""Local optimizer scaffold."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from dl_core.core import register_optimizer
from {import_path} import {base_class}


@register_optimizer({registry_literal})
class {class_name}({base_class}):
    """{class_docstring}"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
    ) -> None:
        defaults = {{"lr": lr}}
        super().__init__(params, defaults)

    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Perform one optimization step."""
        raise NotImplementedError("TODO: implement the optimizer update rule.")
'''


def _scheduler_component(
    *,
    registry_literal: str,
    class_name: str,
    import_path: str,
    base_class: str,
    class_docstring: str,
) -> str:
    """Render a minimal scheduler scaffold."""
    return f'''"""Local scheduler scaffold."""

from __future__ import annotations

from torch.optim import Optimizer

from dl_core.core import register_scheduler
from {import_path} import {base_class}


@register_scheduler({registry_literal})
class {class_name}({base_class}):
    """{class_docstring}"""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Return one learning rate per optimizer parameter group."""
        raise NotImplementedError("TODO: implement the scheduler update rule.")
'''
