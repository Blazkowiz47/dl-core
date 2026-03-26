"""Helpers for describing registered components and importable classes."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from dl_core import load_builtin_components, load_local_components
from dl_core.core import (
    ACCELERATOR_REGISTRY,
    AUGMENTATION_REGISTRY,
    CALLBACK_REGISTRY,
    CRITERION_REGISTRY,
    DATASET_REGISTRY,
    EXECUTOR_REGISTRY,
    FACE_DETECTOR_REGISTRY,
    METRIC_MANAGER_REGISTRY,
    METRICS_SOURCE_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    BIOMETRIC_PREPROCESSOR_REGISTRY,
    SAMPLER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRACKER_REGISTRY,
    TRAINER_REGISTRY,
    ComponentRegistry,
)

_COMPONENT_REGISTRIES: dict[str, ComponentRegistry] = {
    "accelerator": ACCELERATOR_REGISTRY,
    "augmentation": AUGMENTATION_REGISTRY,
    "callback": CALLBACK_REGISTRY,
    "criterion": CRITERION_REGISTRY,
    "dataset": DATASET_REGISTRY,
    "executor": EXECUTOR_REGISTRY,
    "face_detector": FACE_DETECTOR_REGISTRY,
    "metric": METRIC_REGISTRY,
    "metric_manager": METRIC_MANAGER_REGISTRY,
    "metrics_source": METRICS_SOURCE_REGISTRY,
    "model": MODEL_REGISTRY,
    "optimizer": OPTIMIZER_REGISTRY,
    "biometric_preprocessor": BIOMETRIC_PREPROCESSOR_REGISTRY,
    "sampler": SAMPLER_REGISTRY,
    "scheduler": SCHEDULER_REGISTRY,
    "tracker": TRACKER_REGISTRY,
    "trainer": TRAINER_REGISTRY,
}

_DESCRIBE_TYPE_ALIASES = {
    "accelerator": "accelerator",
    "accelerators": "accelerator",
    "augmentation": "augmentation",
    "augmentations": "augmentation",
    "callback": "callback",
    "callbacks": "callback",
    "class": "class",
    "classes": "class",
    "criterion": "criterion",
    "criterions": "criterion",
    "dataset": "dataset",
    "datasets": "dataset",
    "executor": "executor",
    "executors": "executor",
    "face_detector": "face_detector",
    "face_detectors": "face_detector",
    "facedetector": "face_detector",
    "facedetectors": "face_detector",
    "metric": "metric",
    "metrics": "metric",
    "metric_manager": "metric_manager",
    "metric_managers": "metric_manager",
    "metricmanager": "metric_manager",
    "metricmanagers": "metric_manager",
    "metrics_source": "metrics_source",
    "metrics_sources": "metrics_source",
    "metricssource": "metrics_source",
    "metricssources": "metrics_source",
    "model": "model",
    "models": "model",
    "optimizer": "optimizer",
    "optimizers": "optimizer",
    "biometric_preprocessor": "biometric_preprocessor",
    "biometric_preprocessors": "biometric_preprocessor",
    "biometricpreprocessor": "biometric_preprocessor",
    "biometricpreprocessors": "biometric_preprocessor",
    "sampler": "sampler",
    "samplers": "sampler",
    "scheduler": "scheduler",
    "schedulers": "scheduler",
    "tracker": "tracker",
    "trackers": "tracker",
    "trainer": "trainer",
    "trainers": "trainer",
}


def list_supported_describe_types() -> list[str]:
    """Return the describe target types supported by the CLI."""
    return sorted(set(_DESCRIBE_TYPE_ALIASES.values()))


def normalize_describe_type(component_type: str) -> str:
    """Normalize a user-provided describe target type string."""
    key = component_type.strip().lower().replace("-", "_").replace(" ", "_")
    canonical_name = _DESCRIBE_TYPE_ALIASES.get(key)
    if canonical_name is None:
        supported = ", ".join(list_supported_describe_types())
        raise ValueError(
            f"Unsupported describe target '{component_type}'. Supported types: "
            f"{supported}"
        )
    return canonical_name


def describe_component(
    component_type: str,
    name: str,
    *,
    root_dir: str = ".",
) -> dict[str, Any]:
    """Resolve and describe a registered component or importable class."""
    canonical_type = normalize_describe_type(component_type)

    load_builtin_components()
    load_local_components(root_dir)

    if canonical_type == "class":
        cls = _load_class_from_path(name)
        registered_names: list[str] = []
    else:
        registry = _COMPONENT_REGISTRIES[canonical_type]
        cls = registry.get_class(name)
        registered_names = registry.get_registered_names_for_class(cls)

    return _describe_class(
        cls,
        component_type=canonical_type,
        requested_name=name,
        registered_names=registered_names,
    )


def format_description(description: dict[str, Any]) -> str:
    """Format a component description as readable text."""
    lines = [
        "Component description",
        "=" * 21,
        f"Target type: {description['component_type']}",
        f"Requested name: {description['requested_name']}",
        f"Resolved class: {description['qualified_name']}",
    ]

    registered_names = description["registered_names"]
    if registered_names:
        lines.append(f"Registered names: {', '.join(registered_names)}")

    source_file = description["source_file"]
    if source_file:
        source_line = description["source_line"]
        if source_line is None:
            lines.append(f"Source: {source_file}")
        else:
            lines.append(f"Source: {source_file}:{source_line}")

    lines.extend(
        [
            f"Module: {description['module']}",
            f"Constructor: {description['constructor_signature']}",
            "",
            "Inheritance:",
        ]
    )
    for base_name in description["inheritance"]:
        lines.append(f"  - {base_name}")

    lines.extend(["", "Docstring:"])
    docstring = description["docstring"] or "(none)"
    for doc_line in docstring.splitlines():
        lines.append(f"  {doc_line}")

    lines.extend(["", "Properties:"])
    properties = description["properties"]
    if not properties:
        lines.append("  (none declared)")
    else:
        for prop in properties:
            lines.append(f"  - {prop['name']}")

    lines.extend(["", "Class attributes:"])
    class_attributes = description["class_attributes"]
    if not class_attributes:
        lines.append("  (none declared)")
    else:
        for item in class_attributes:
            lines.append(f"  - {item['name']} = {item['value_repr']}")

    lines.extend(["", "Public methods defined on class:"])
    methods = description["methods"]
    if not methods:
        lines.append("  (none declared)")
    else:
        for method in methods:
            lines.append(f"  - {method['name']}{method['signature']}")

    lines.extend(["", "Notes:"])
    for note in description["notes"]:
        lines.append(f"  - {note}")

    return "\n".join(lines)


def format_description_json(description: dict[str, Any]) -> str:
    """Serialize a component description as pretty JSON."""
    return json.dumps(description, indent=2, sort_keys=True)


def _load_class_from_path(class_path: str) -> type[Any]:
    """Import a class from a fully qualified module path."""
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            "Class paths must look like 'package.module.ClassName'. "
            f"Received: {class_path}"
        )

    module = import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Module '{module_path}' does not define class '{class_name}'."
        )
    if not inspect.isclass(cls):
        raise TypeError(
            f"'{class_path}' resolved to {type(cls).__name__}, not a class."
        )
    return cls


def _describe_class(
    cls: type[Any],
    *,
    component_type: str,
    requested_name: str,
    registered_names: list[str],
) -> dict[str, Any]:
    """Build a serializable description for a resolved class."""
    constructor_signature = _safe_signature(cls.__init__)
    source_file = inspect.getsourcefile(cls)
    source_line: int | None = None
    if source_file is not None:
        try:
            _, source_line = inspect.getsourcelines(cls)
        except OSError:
            source_line = None

    return {
        "component_type": component_type,
        "requested_name": requested_name,
        "registered_names": registered_names,
        "class_name": cls.__name__,
        "qualified_name": f"{cls.__module__}.{cls.__qualname__}",
        "module": cls.__module__,
        "source_file": str(Path(source_file).resolve()) if source_file else None,
        "source_line": source_line,
        "constructor_signature": constructor_signature,
        "inheritance": [_qualified_base_name(base) for base in cls.__mro__],
        "docstring": inspect.getdoc(cls),
        "properties": _collect_properties(cls),
        "class_attributes": _collect_class_attributes(cls),
        "methods": _collect_methods(cls),
        "notes": [
            (
                "Only class-level descriptors are shown here. Instance "
                "attributes created inside __init__ are not discoverable "
                "without constructing the class."
            )
        ],
    }


def _collect_properties(cls: type[Any]) -> list[dict[str, str | None]]:
    """Collect public property descriptors declared directly on a class."""
    properties: list[dict[str, str | None]] = []
    for name, value in sorted(cls.__dict__.items()):
        if name.startswith("_") or not isinstance(value, property):
            continue
        properties.append(
            {
                "name": name,
                "docstring": inspect.getdoc(value.fget) if value.fget else None,
            }
        )
    return properties


def _collect_class_attributes(cls: type[Any]) -> list[dict[str, str]]:
    """Collect public non-callable class attributes declared on a class."""
    class_attributes: list[dict[str, str]] = []
    for name, value in sorted(cls.__dict__.items()):
        if name.startswith("_") or isinstance(value, property):
            continue
        if _unwrap_callable(value) is not None:
            continue
        class_attributes.append({"name": name, "value_repr": repr(value)})
    return class_attributes


def _collect_methods(cls: type[Any]) -> list[dict[str, str | None]]:
    """Collect public methods declared directly on a class."""
    methods: list[dict[str, str | None]] = []
    for name, value in sorted(cls.__dict__.items()):
        if name.startswith("_"):
            continue
        function = _unwrap_callable(value)
        if function is None:
            continue
        methods.append(
            {
                "name": name,
                "signature": _safe_signature(function),
                "docstring": inspect.getdoc(function),
            }
        )
    return methods


def _unwrap_callable(value: Any) -> Callable[..., Any] | None:
    """Normalize a descriptor into a function-like object when possible."""
    if isinstance(value, (classmethod, staticmethod)):
        return value.__func__
    if inspect.isfunction(value) or inspect.ismethoddescriptor(value):
        return value
    return None


def _safe_signature(callable_obj: Callable[..., Any]) -> str:
    """Return a readable signature string for a callable."""
    try:
        return str(inspect.signature(callable_obj))
    except (TypeError, ValueError):
        return "(...)"


def _qualified_base_name(cls: type[Any]) -> str:
    """Return a fully qualified name for a class in the inheritance chain."""
    return f"{cls.__module__}.{cls.__qualname__}"
