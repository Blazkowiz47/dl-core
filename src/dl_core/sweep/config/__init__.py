"""
Sweep configuration package.

Focused modules for configuration building:
- config_utils: Pure utility functions
- config_builder: Main ConfigBuilder class
- grid_expansion: Parameter grid expansion
- preset_resolver: Preset handling
- name_generator: Run name generation
"""

from .config_utils import (
    convert_numeric_strings,
    deep_get,
    deep_set,
    deep_update,
    load_centralized_presets,
    load_global_presets,
    load_user_presets,
    load_experiment_presets,
    resolve_conditional_values,
    validate_grid_syntax,
    extract_tags_from_config,
)

from .config_builder import ConfigBuilder

__all__ = [
    # Utilities
    "convert_numeric_strings",
    "deep_get",
    "deep_set",
    "deep_update",
    "load_centralized_presets",
    "load_global_presets",
    "load_user_presets",
    "load_experiment_presets",
    "resolve_conditional_values",
    "validate_grid_syntax",
    "extract_tags_from_config",
    # Main class
    "ConfigBuilder",
]
