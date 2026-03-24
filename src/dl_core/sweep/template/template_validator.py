"""
Template and sweep configuration validation.

This module handles:
- Sweep configuration validation
- Grid syntax validation
"""

from __future__ import annotations

from typing import List


def validate_sweep_config(sweep_config: dict) -> List[str]:
    """
    Validate sweep configuration.

    Args:
        sweep_config: Sweep configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    from ..config import validate_grid_syntax

    errors = []

    # Check grid syntax
    if "grid" in sweep_config:
        grid_errors = validate_grid_syntax(sweep_config["grid"])
        errors.extend(grid_errors)

    # Validate seeds
    if "seeds" in sweep_config:
        seeds = sweep_config["seeds"]
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            errors.append("'seeds' must be a list of integers")

    return errors
