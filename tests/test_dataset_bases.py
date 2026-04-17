"""Tests for reusable dataset base classes."""

from __future__ import annotations

from typing import Any

import torch

from dl_core.core import (
    AdaptiveComputationDataset,
    BaseSampler,
    TextSequenceWrapper,
)
from dl_core.core.base_dataset import BaseWrapper
from dl_core.core.registry import SAMPLER_REGISTRY


class _TextSequenceDataset(TextSequenceWrapper):
    """Small concrete text dataset for sequence padding tests."""

    def __init__(self) -> None:
        config = {
            "dataset": {
                "name": "text-sequence-demo",
                "batch_size": {"train": 2, "validation": 2, "test": 2},
                "num_workers": {"train": 0, "validation": 0, "test": 0},
                "shuffle": {"train": False, "validation": False, "test": False},
                "sequence_keys": ["input_ids", "attention_mask"],
                "sequence_padding_values": {"input_ids": 0, "attention_mask": 0},
            }
        }
        super().__init__(config)

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return a small fixed split for testing."""

        return [
            {"path": "sample-a", "label": 0, "tokens": [1, 2, 3]},
            {"path": "sample-b", "label": 1, "tokens": [4, 5]},
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Convert test records into token tensors."""

        input_ids = torch.tensor(file_dict["tokens"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "label": file_dict["label"],
            "path": file_dict["path"],
        }


class _AdaptiveDataset(AdaptiveComputationDataset):
    """Concrete adaptive dataset for class-stream tests."""

    @property
    def file_extensions(self) -> list[str]:
        """Return an empty extension list for the in-memory test dataset."""

        return []

    def __init__(self) -> None:
        config = {
            "dataset": {
                "name": "adaptive-demo",
                "class_stream_shuffle": {
                    "train": False,
                    "validation": False,
                    "test": False,
                },
            }
        }
        super().__init__(config)

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return a small fixed split for class-stream tests."""

        return [
            {"path": "a0", "label": 0, "value": 10},
            {"path": "b0", "label": 1, "value": 20},
            {"path": "a1", "label": 0, "value": 11},
            {"path": "b1", "label": 1, "value": 21},
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Return a simple transformed adaptive sample."""

        return {
            "path": file_dict["path"],
            "label": file_dict["label"],
            "value": torch.tensor(file_dict["value"]),
        }


class _RecordingSampler(BaseSampler):
    """Small sampler that drops the tail item and records call counts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def sample_data(self, files: list[dict], split: str) -> list[dict]:
        """Return a shortened list so repeated sampling is easy to detect."""

        self.calls += 1
        return list(files[:-1])


class _SamplerDataset(TextSequenceWrapper):
    """Concrete text dataset with a configurable sampler hook."""

    def __init__(self, sampler_name: str) -> None:
        config = {
            "dataset": {
                "name": "sampler-demo",
                "batch_size": {"train": 2, "validation": 2, "test": 2},
                "num_workers": {"train": 0, "validation": 0, "test": 0},
                "shuffle": {"train": False, "validation": False, "test": False},
                "sample_splits": {
                    "train": True,
                    "validation": False,
                    "test": False,
                },
                "sampler": {sampler_name: {}},
                "sequence_keys": ["input_ids"],
            }
        }
        super().__init__(config)

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return a small fixed split for sampling tests."""

        return [
            {"path": "sample-a", "label": 0, "tokens": [1]},
            {"path": "sample-b", "label": 1, "tokens": [2]},
            {"path": "sample-c", "label": 0, "tokens": [3]},
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Convert test records into minimal tensors."""

        return {
            "input_ids": torch.tensor(file_dict["tokens"], dtype=torch.long),
            "label": file_dict["label"],
            "path": file_dict["path"],
        }


class _OverrideDataset(BaseWrapper):
    """Small dataset used to verify falsy loader overrides."""

    @property
    def file_extensions(self) -> list[str]:
        """Return an empty extension list for the in-memory test dataset."""

        return []

    def __init__(self) -> None:
        config = {
            "dataset": {
                "name": "override-demo",
                "batch_size": {"train": 2, "validation": 2, "test": 2},
                "num_workers": {"train": 0, "validation": 0, "test": 0},
                "shuffle": {"train": False, "validation": False, "test": False},
                "drop_last": {"train": True, "validation": False, "test": False},
            }
        }
        super().__init__(config)

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return a small fixed split for override tests."""

        return [
            {"path": "sample-a", "label": 0, "value": 1},
            {"path": "sample-b", "label": 1, "value": 2},
            {"path": "sample-c", "label": 0, "value": 3},
        ]

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Return a minimal tensor payload."""

        return {
            "data": torch.tensor([file_dict["value"]], dtype=torch.float32),
            "label": file_dict["label"],
            "path": file_dict["path"],
        }


def test_text_sequence_wrapper_pads_variable_length_batches() -> None:
    """Text sequence batches should be padded on configured sequence keys."""

    dataset = _TextSequenceDataset()
    batch = next(iter(dataset.get_split("train")))

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [4, 5, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["label"].tolist() == [0, 1]


def test_adaptive_dataset_streams_samples_by_class() -> None:
    """Adaptive datasets should return class-specific samples in stream order."""

    dataset = _AdaptiveDataset()

    first = dataset.get_next_class_sample(0, "train", transform_sample=False)
    second = dataset.get_next_class_sample(0, "train", transform_sample=False)
    exhausted = dataset.get_next_class_sample(0, "train", transform_sample=False)

    assert first == {"path": "a0", "label": 0, "value": 10}
    assert second == {"path": "a1", "label": 0, "value": 11}
    assert exhausted is None


def test_adaptive_dataset_can_peek_and_wrap_streams() -> None:
    """Adaptive datasets should support peeking and wrap-around class streams."""

    dataset = _AdaptiveDataset()

    peeked = dataset.peek_next_class_sample(1, "train", transform_sample=False)
    first = dataset.get_next_class_sample(1, "train")
    second = dataset.get_next_class_sample(1, "train")
    wrapped = dataset.get_next_class_sample(1, "train", wrap_around=True)

    assert peeked == {"path": "b0", "label": 1, "value": 20}
    assert first["path"] == "b0"
    assert second["path"] == "b1"
    assert wrapped["path"] == "b0"


def test_dataset_reuses_sampled_files_on_repeated_split_access(
    monkeypatch: Any,
) -> None:
    """Repeated split access should keep using the sampled file cache."""

    sampler = _RecordingSampler(seed=2025)
    original_get = SAMPLER_REGISTRY.get

    def _get_sampler(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "recording":
            return sampler
        return original_get(name, *args, **kwargs)

    monkeypatch.setattr(SAMPLER_REGISTRY, "get", _get_sampler)

    dataset = _SamplerDataset("recording")

    first_loader = dataset.get_split("train")
    second_loader = dataset.get_split("train")

    assert first_loader is not None
    assert second_loader is not None
    assert len(first_loader.dataset) == 2
    assert len(second_loader.dataset) == 2
    assert sampler.calls == 1


def test_dataset_allows_falsey_loader_overrides() -> None:
    """Per-call loader overrides should honor explicit falsey values."""

    dataset = _OverrideDataset()

    loader = dataset.get_split("train", drop_last=False)

    assert loader is not None
    batches = list(loader)
    assert len(batches) == 2


def test_dataset_reproducibility_uses_configured_deterministic_flag(
    monkeypatch: Any,
) -> None:
    """Dataset seeding should pass the configured deterministic flag through."""

    calls: list[tuple[int, bool]] = []

    def _record_seed(seed: int, deterministic: bool = True) -> None:
        calls.append((seed, deterministic))

    monkeypatch.setattr("dl_core.core.base_dataset.set_seeds_local", _record_seed)

    dataset = _OverrideDataset()
    dataset.seed = 321
    dataset.deterministic = False

    dataset._ensure_reproducibility()

    assert calls == [(321, False)]
