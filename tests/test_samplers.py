"""Tests for built-in sampler implementations."""

from __future__ import annotations

from dl_core import load_builtin_components
from dl_core.core import SAMPLER_REGISTRY


def test_label_sampler_undersamples_by_metadata_key() -> None:
    """The built-in label sampler should undersample all groups evenly."""
    load_builtin_components()
    sampler = SAMPLER_REGISTRY.get("label", key="attack", mode="undersample", seed=2025)

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

    assert len(sampled) == 3
    assert attack_counts == {
        "real": 1,
        "print": 1,
        "screen": 1,
    }


def test_label_sampler_oversamples_by_metadata_key() -> None:
    """The built-in label sampler should oversample all groups evenly."""
    load_builtin_components()
    sampler = SAMPLER_REGISTRY.get("label", key="label", mode="oversample", seed=2025)

    files = [
        {"path": "a-0", "label": 0},
        {"path": "b-0", "label": 1},
        {"path": "b-1", "label": 1},
        {"path": "b-2", "label": 1},
    ]

    sampled = sampler.sample(files, "train")
    label_counts: dict[str, int] = {}
    for file_dict in sampled:
        label_name = str(file_dict["label"])
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

    assert len(sampled) == 6
    assert label_counts == {"0": 3, "1": 3}
