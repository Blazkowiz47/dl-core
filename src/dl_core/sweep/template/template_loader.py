"""
Template loading and SweepTemplate class.

This module handles:
- Loading sweep templates from YAML files
- Template validation
- Template property access
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import yaml


class TemplateError(Exception):
    """Exception raised for template-related errors."""

    pass


class SweepTemplate:
    """
    Represents a loaded sweep template with validation and inheritance support.
    """

    def __init__(self, template_path: Union[str, Path]):
        self.template_path = Path(template_path)
        self.template_data = self._load_template()
        self._validate_template()

    def _load_template(self) -> dict:
        """Load template from YAML file."""
        if not self.template_path.exists():
            raise TemplateError(f"Template file not found: {self.template_path}")

        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise TemplateError(
                    f"Template must be a dictionary: {self.template_path}"
                )

            return data
        except yaml.YAMLError as e:
            raise TemplateError(f"Invalid YAML in template {self.template_path}: {e}")

    def _validate_template(self) -> None:
        """Validate template structure and required fields."""
        from ..config import validate_grid_syntax

        required_fields = ["template_name", "base_config"]
        missing_fields = [
            field for field in required_fields if field not in self.template_data
        ]

        if missing_fields:
            raise TemplateError(
                f"Template missing required fields: {missing_fields} in {self.template_path}"
            )

        # Validate base_config path (resolve relative to template directory)
        base_config_rel = self.template_data["base_config"]
        base_config_path = Path(base_config_rel)

        # If relative path, resolve from template directory
        if not base_config_path.is_absolute():
            base_config_path = (self.template_path.parent / base_config_rel).resolve()

        if not base_config_path.exists():
            raise TemplateError(
                f"Base config file not found: {self.template_data['base_config']} "
                f"(resolved to {base_config_path})"
            )

        # Validate grid syntax if present
        if "default_grid" in self.template_data:
            errors = validate_grid_syntax(self.template_data["default_grid"])
            if errors:
                raise TemplateError(f"Invalid grid syntax in template: {errors}")

    @property
    def name(self) -> str:
        """Get template name."""
        return self.template_data["template_name"]

    @property
    def base_config(self) -> str:
        """Get base config path (resolved to absolute path)."""
        base_config_rel = self.template_data["base_config"]
        base_config_path = Path(base_config_rel)

        # If relative path, resolve from template directory
        if not base_config_path.is_absolute():
            base_config_path = (self.template_path.parent / base_config_rel).resolve()

        return str(base_config_path)

    @property
    def fixed_params(self) -> dict:
        """Get fixed parameters."""
        return self.template_data.get("fixed", {})

    @property
    def default_grid(self) -> dict:
        """Get default grid parameters."""
        return self.template_data.get("default_grid", {})

    @property
    def tracking(self) -> dict:
        """Get generic tracking configuration."""
        return self.template_data.get("tracking", {})

    @property
    def preset_configs(self) -> dict:
        """Get preset configurations."""
        return self.template_data.get("preset_configs", {})

    @property
    def executor(self) -> dict:
        """Get executor configuration."""
        return self.template_data.get("executor", {})

    @property
    def accelerator(self) -> dict:
        """Get accelerator configuration."""
        return self.template_data.get("accelerator", {})

    def get_auto_tags(self) -> dict:
        """Get auto-tag configuration."""
        tracking_config = self.tracking
        if isinstance(tracking_config, dict):
            return tracking_config.get("auto_tags", {})
        return {}

    def get_description_template(self) -> Optional[str]:
        """Get description template string."""
        tracking_config = self.tracking
        if isinstance(tracking_config, dict):
            return tracking_config.get("description_template")
        return None


def load_template(
    template_ref: str,
    template_dirs: Optional[List[Union[str, Path]]] = None,
    sweep_file_path: Optional[Union[str, Path]] = None,
) -> SweepTemplate:
    """
    Load a sweep template by reference with support for relative paths.

    Args:
        template_ref: Template reference (filename or path)
        template_dirs: List of directories to search for templates
        sweep_file_path: Path to the sweep file (for relative template resolution)

    Returns:
        Loaded SweepTemplate instance
    """
    template_path = Path(template_ref)

    # Handle relative paths (e.g., "./base_template.yaml" or "foundpad/base_template.yaml")
    if not template_path.is_absolute():
        if sweep_file_path:
            # Resolve relative to sweep file directory
            sweep_dir = Path(sweep_file_path).parent
            candidate = sweep_dir / template_ref
            if candidate.exists():
                return SweepTemplate(candidate)
        raise TemplateError(
            f"Relative template path '{template_ref}' requires sweep_file_path parameter"
        )

    # Handle absolute paths
    if template_path.is_absolute() and template_path.exists():
        return SweepTemplate(template_path)

    # Default template search directories (for backwards compatibility)
    if template_dirs is None:
        experiments_dir = Path(__file__).parent.parent
        template_dirs = [
            experiments_dir / "shared_templates",
            experiments_dir / "base_templates",  # Support new naming
            experiments_dir,
        ]

    # Convert to Path objects
    search_dirs = [Path(d) for d in template_dirs]

    # Search in template directories
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Try exact match first
        candidate = search_dir / template_ref
        if candidate.exists():
            return SweepTemplate(candidate)

        # Try with .yaml extension
        if not template_ref.endswith((".yaml", ".yml")):
            candidate = search_dir / f"{template_ref}.yaml"
            if candidate.exists():
                return SweepTemplate(candidate)

    raise TemplateError(f"Template not found: {template_ref}")


def list_available_templates(
    template_dirs: Optional[List[Union[str, Path]]] = None,
) -> List[dict]:
    """
    List all available templates with their metadata.

    Args:
        template_dirs: List of directories to search for templates

    Returns:
        List of template metadata dictionaries
    """
    if template_dirs is None:
        experiments_dir = Path(__file__).parent.parent
        template_dirs = [experiments_dir / "shared_templates"]

    templates = []

    for template_dir in template_dirs:
        template_dir = Path(template_dir)
        if not template_dir.exists():
            continue

        for template_file in template_dir.glob("*.yaml"):
            try:
                template = SweepTemplate(template_file)
                templates.append(
                    {
                        "name": template.name,
                        "path": str(template_file),
                        "description": template.template_data.get("description", ""),
                        "version": template.template_data.get("version", "unknown"),
                    }
                )
            except TemplateError:
                # Skip invalid templates
                continue

    return templates
