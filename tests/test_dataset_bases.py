"""Tests for reusable dataset base classes."""

from __future__ import annotations

from typing import Any

import torch

from dl_core.core import AdaptiveComputationDataset, TextSequenceWrapper


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
