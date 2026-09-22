"""Tests for reusable dataset base classes."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
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


class _AutoSplitDataset(_OverrideDataset):
    """Dataset with one raw split used to verify split-before-sampling."""

    def __init__(self, sampler_name: str) -> None:
        config = {
            "dataset": {
                "name": "auto-split-demo",
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": False,
                "validation_partition": 0.2,
                "test_split": 0.25,
                "stratify": False,
                "sample_splits": {
                    "train": True,
                    "validation": False,
                    "test": False,
                },
                "sampler": {sampler_name: {}},
            }
        }
        BaseWrapper.__init__(self, config)

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return ten unique training records and empty held-out splits."""

        if split != "train":
            return []
        return [
            {"path": f"sample-{index}", "label": index % 2, "value": index}
            for index in range(10)
        ]


class _DuplicatingSampler(BaseSampler):
    """Duplicate records so pre-split sampling would leak identities."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def sample_data(self, files: list[dict], split: str) -> list[dict]:
        """Return two references to every input record."""

        self.calls += 1
        return [*files, *files]


class _WorkerRandomDataset(_OverrideDataset):
    """Dataset whose transform exposes each worker RNG stream."""

    def __init__(self) -> None:
        config = {
            "dataset": {
                "name": "worker-rng-demo",
                "batch_size": 2,
                "num_workers": 1,
                "shuffle": False,
                "persistent_workers": False,
                "seed": 321,
            }
        }
        BaseWrapper.__init__(self, config)

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        """Return random values from Python, NumPy, and PyTorch."""

        return {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": torch.rand(()),
            "label": file_dict["label"],
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

    dataset.set_epoch(3)

    assert sampler.current_epoch == 3


def test_dataset_allows_falsey_loader_overrides() -> None:
    """Per-call loader overrides should honor explicit falsey values."""

    dataset = _OverrideDataset()

    loader = dataset.get_split("train", drop_last=False)

    assert loader is not None
    batches = list(loader)
    assert len(batches) == 2


def test_file_list_access_does_not_reset_global_rng_state() -> None:
    """Loading a split must not rewind model or augmentation randomness."""

    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    expected = (random.random(), float(np.random.random()), torch.rand(()).item())

    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    dataset = _OverrideDataset()
    dataset._get_file_list("train")
    actual = (random.random(), float(np.random.random()), torch.rand(()).item())

    assert actual == expected


def test_auto_split_uses_raw_records_before_sampling(monkeypatch: Any) -> None:
    """Oversampling must not duplicate identities across dataset partitions."""

    sampler = _DuplicatingSampler(seed=2025)
    original_get = SAMPLER_REGISTRY.get

    def _get_sampler(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "duplicating":
            return sampler
        return original_get(name, *args, **kwargs)

    monkeypatch.setattr(SAMPLER_REGISTRY, "get", _get_sampler)
    dataset = _AutoSplitDataset("duplicating")

    dataset.auto_generate_partitions()

    paths = {
        split: {record["path"] for record in records}
        for split, records in dataset.files_list.items()
    }
    assert sampler.calls == 0
    assert len(paths["train"] | paths["validation"] | paths["test"]) == 10
    assert paths["train"].isdisjoint(paths["validation"])
    assert paths["train"].isdisjoint(paths["test"])
    assert paths["validation"].isdisjoint(paths["test"])

    train_loader = dataset.get_split("train")

    assert train_loader is not None
    assert sampler.calls == 1
    assert len(train_loader.dataset) == 2 * len(dataset.files_list["train"])


def test_worker_randomness_changes_by_epoch_and_repeats_across_runs() -> None:
    """Respawned workers should vary by epoch and remain reproducible."""

    first_dataset = _WorkerRandomDataset()
    first_loader = first_dataset.get_split("train")
    assert first_loader is not None

    first_dataset.set_epoch(0)
    epoch_zero = next(iter(first_loader))
    first_dataset.set_epoch(1)
    epoch_one = next(iter(first_loader))

    second_dataset = _WorkerRandomDataset()
    second_loader = second_dataset.get_split("train")
    assert second_loader is not None
    second_dataset.set_epoch(0)
    repeated_epoch_zero = next(iter(second_loader))

    for key in ["python", "numpy", "torch"]:
        assert not torch.equal(epoch_zero[key], epoch_one[key])
        assert torch.equal(epoch_zero[key], repeated_epoch_zero[key])
