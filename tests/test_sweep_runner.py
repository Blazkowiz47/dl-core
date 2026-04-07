"""Tests for sweep runner helper behavior."""

from __future__ import annotations

from dl_core.sweep.runner import _filter_prepared_configs


def test_filter_prepared_configs_applies_only_and_skip_patterns() -> None:
    """Run-name filters should keep only the requested subset."""
    prepared_configs = [
        (0, {"seed": 2025}, "backbone_swin_s_seed_2025"),
        (1, {"seed": 2025}, "backbone_swin_b_seed_2025"),
        (2, {"seed": 2026}, "backbone_convnext_seed_2026"),
    ]

    filtered = _filter_prepared_configs(
        prepared_configs,
        ["backbone_swin_*"],
        ["*_b_*"],
    )

    assert filtered == [(0, {"seed": 2025}, "backbone_swin_s_seed_2025")]
