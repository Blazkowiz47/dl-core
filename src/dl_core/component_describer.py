"""Helpers for describing registered components and importable classes."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any
import yaml

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


def list_supported_list_types() -> list[str]:
    """Return the registry target types supported by the CLI."""
    return ["all", *sorted(_COMPONENT_REGISTRIES.keys())]


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


def normalize_list_type(component_type: str) -> str:
    """Normalize a user-provided registry list target string."""
    key = component_type.strip().lower().replace("-", "_").replace(" ", "_")
    if key == "all":
        return "all"

    canonical_name = _DESCRIBE_TYPE_ALIASES.get(key)
    if canonical_name is None or canonical_name == "class":
        supported = ", ".join(list_supported_list_types())
        raise ValueError(
            f"Unsupported list target '{component_type}'. Supported types: "
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


def list_registered_components(
    component_type: str = "all",
    *,
    root_dir: str = ".",
) -> dict[str, list[str]]:
    """List registered component names for one registry or all registries."""
    canonical_type = normalize_list_type(component_type)

    load_builtin_components()
    load_local_components(root_dir)

    if canonical_type == "all":
        target_types = list(_COMPONENT_REGISTRIES.keys())
    else:
        target_types = [canonical_type]

    return {
        target_type: sorted(_COMPONENT_REGISTRIES[target_type].list_registered())
        for target_type in target_types
    }


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

    lines.extend(["", "Config fields:"])
    config_fields = description["config_fields"]
    if not config_fields:
        lines.append("  (none declared)")
    else:
        for field in config_fields:
            field_type = field.get("type") or "Any"
            details = [field_type]
            if field.get("required"):
                details.append("required")
            elif "default" in field:
                details.append(f"default={field['default_repr']}")
            lines.append(f"  - {field['name']} [{', '.join(details)}]")
            description_text = field.get("description")
            if isinstance(description_text, str) and description_text:
                lines.append(f"    {description_text}")

    lines.extend(["", "Example config:"])
    config_example = description.get("config_example")
    if not isinstance(config_example, str) or not config_example:
        lines.append("  (no direct snippet available)")
    else:
        lines.append("  ```yaml")
        for example_line in config_example.splitlines():
            lines.append(f"  {example_line}")
        lines.append("  ```")

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


def format_registry_listing(listing: dict[str, list[str]]) -> str:
    """Format a registry listing as readable text."""
    if len(listing) == 1:
        component_type, names = next(iter(listing.items()))
        label = component_type.replace("_", " ")
        lines = [
            f"Registered {label}",
            "=" * (11 + len(label)),
        ]
        if names:
            lines.extend(f"- {name}" for name in names)
        else:
            lines.append("(none registered)")
        return "\n".join(lines)

    total = sum(len(names) for names in listing.values())
    lines = [
        "Registered components",
        "=" * 21,
    ]
    for component_type, names in listing.items():
        lines.append(f"{component_type}:")
        if names:
            lines.extend(f"  - {name}" for name in names)
        else:
            lines.append("  (none registered)")
        lines.append("")
    lines.append(f"Total components: {total}")
    return "\n".join(lines).rstrip()


def format_registry_listing_json(listing: dict[str, list[str]]) -> str:
    """Serialize a registry listing as pretty JSON."""
    return json.dumps(listing, indent=2, sort_keys=True)


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
        "config_fields": _collect_config_fields(cls),
        "config_example": _build_config_example(
            component_type,
            requested_name,
            registered_names,
            _collect_config_fields(cls),
        ),
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


def _collect_config_fields(cls: type[Any]) -> list[dict[str, Any]]:
    """Collect explicitly declared user-facing config fields for a class."""
    fields_by_name: dict[str, dict[str, Any]] = {}

    for base in reversed(cls.__mro__):
        config_fields = base.__dict__.get("CONFIG_FIELDS")
        if not isinstance(config_fields, list):
            continue

        for item in config_fields:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue

            normalized = dict(item)
            if "default" in normalized:
                normalized["default_repr"] = repr(normalized["default"])
            fields_by_name[name] = normalized

    return list(fields_by_name.values())


def _collect_class_attributes(cls: type[Any]) -> list[dict[str, str]]:
    """Collect public non-callable class attributes declared on a class."""
    class_attributes: list[dict[str, str]] = []
    for name, value in sorted(cls.__dict__.items()):
        if name == "CONFIG_FIELDS":
            continue
        if name.startswith("_") or isinstance(value, property):
            continue
        if _unwrap_callable(value) is not None:
            continue
        class_attributes.append({"name": name, "value_repr": repr(value)})
    return class_attributes


def _build_config_example(
    component_type: str,
    requested_name: str,
    registered_names: list[str],
    config_fields: list[dict[str, Any]],
) -> str | None:
    """Build a minimal YAML snippet for directly configurable component types."""
    component_name = registered_names[0] if registered_names else requested_name
    field_values = _build_example_field_values(config_fields)

    if component_type == "dataset":
        payload = {"dataset": {"name": component_name, **field_values}}
    elif component_type == "accelerator":
        payload = {"accelerator": {"type": component_name, **field_values}}
    elif component_type == "executor":
        payload = {"executor": {"name": component_name, **field_values}}
    elif component_type == "model":
        payload = {"models": {component_name: field_values}}
    elif component_type == "trainer":
        payload = {"trainer": {component_name: field_values}}
    elif component_type == "criterion":
        payload = {"criterions": {component_name: field_values}}
    elif component_type == "metric_manager":
        payload = {"metric_managers": {component_name: field_values}}
    elif component_type == "callback":
        payload = {"callbacks": {component_name: field_values}}
    elif component_type == "optimizer":
        payload = {"optimizers": {"name": component_name, **field_values}}
    elif component_type == "scheduler":
        payload = {"schedulers": {"name": component_name, **field_values}}
    elif component_type == "augmentation":
        payload = {"dataset": {"augmentation": {component_name: field_values}}}
    elif component_type == "sampler":
        payload = {"dataset": {"sampler": {component_name: field_values}}}
    else:
        return None

    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()


def _build_example_field_values(
    config_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build example field values from declared config metadata."""
    field_values: dict[str, Any] = {}
    for field in config_fields:
        field_name = field.get("name")
        if not isinstance(field_name, str) or not field_name:
            continue
        if field_name in {"name", "type"}:
            continue
        if "default" in field:
            field_values[field_name] = field["default"]
            continue
        if field.get("required"):
            field_values[field_name] = "<required>"
    return field_values


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
