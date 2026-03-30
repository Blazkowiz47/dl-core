"""Tests for built-in sampler implementations."""

from __future__ import annotations

from dl_core import load_builtin_components
from dl_core.core import SAMPLER_REGISTRY


def test_attack_sampler_balances_bonafide_and_attack_groups() -> None:
    """The built-in attack sampler should balance groups from the attack key."""
    load_builtin_components()
    sampler = SAMPLER_REGISTRY.get("attack", seed=2025)

    files = [
        {"path": "real-0", "attack": "real"},
        {"path": "real-1", "attack": "real"},
        {"path": "real-2", "attack": "real"},
        {"path": "print-0", "attack": "print"},
        {"path": "screen-0", "attack": "screen"},
    ]

    sampled = sampler.sample(files, "train")
    attack_counts: dict[str, int] = {}
    for file_dict in sampled:
        attack_name = str(file_dict["attack"])
        attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1

    assert len(sampled) == 4
    assert attack_counts == {
        "real": 2,
        "print": 1,
        "screen": 1,
    }
