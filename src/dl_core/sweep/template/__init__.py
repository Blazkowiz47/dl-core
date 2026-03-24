"""
Sweep template package.

Focused modules for template handling:
- template_loader: SweepTemplate class and template loading
- template_merger: Merging user sweeps with templates
- template_validator: Validation functions
- tracking_utils: Generic tracking metadata extraction
"""

from .template_loader import (
    SweepTemplate,
    TemplateError,
    load_template,
    list_available_templates,
)

from .template_merger import (
    merge_sweep_with_template,
    load_user_sweep,
    resolve_preset_references,
)

from .template_validator import validate_sweep_config

from .tracking_utils import (
    extract_tracking_config,
    generate_experiment_name,
)

__all__ = [
    # Template loading
    "SweepTemplate",
    "TemplateError",
    "load_template",
    "list_available_templates",
    # Template merging
    "merge_sweep_with_template",
    "load_user_sweep",
    "resolve_preset_references",
    # Validation
    "validate_sweep_config",
    # Tracking utilities
    "extract_tracking_config",
    "generate_experiment_name",
]
