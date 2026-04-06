"""Helpers for declaring user-facing component configuration fields."""

from __future__ import annotations

from typing import Any, TypedDict


class ConfigFieldSpec(TypedDict, total=False):
    """One documented user-facing configuration field."""

    name: str
    type: str
    description: str
    required: bool
    default: Any


_MISSING = object()


def config_field(
    name: str,
    field_type: str,
    description: str,
    *,
    required: bool = False,
    default: Any = _MISSING,
) -> ConfigFieldSpec:
    """Build one config-field metadata entry for ``dl-core describe``."""

    field: ConfigFieldSpec = {
        "name": name,
        "type": field_type,
        "description": description,
        "required": required,
    }
    if default is not _MISSING:
        field["default"] = default
    return field
