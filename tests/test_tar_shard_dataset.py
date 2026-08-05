"""Tests for indexed tar datasets and distributed batch construction."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import DataLoader

from dl_core.accelerators.multi_gpu import MultiGPUAccelerator
from dl_core.datasets import (
    RoundRobinTarBatchSampler,
    TarShardIndex,
    TarShardWrapper,
)


def _write_tar(path: Path, samples: dict[str, dict[str, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for key, members in samples.items():
            for extension, payload in members.items():
                info = tarfile.TarInfo(f"{key}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


class _BytesTarWrapper(TarShardWrapper):
    """Concrete tar wrapper that returns grouped bytes unchanged."""

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return {**file_dict, "split": split}


def test_tar_index_and_dataset_read_grouped_members_without_extraction(
    tmp_path: Path,
) -> None:
    tar_path = tmp_path / "samples.tar"
    _write_tar(
        tar_path,
        {
            "nested/sample-1": {"png": b"image-one", "json": b'{"id":1}'},
            "nested/sample-2": {"png": b"image-two", "json": b'{"id":2}'},
        },
    )
    wrapper = _BytesTarWrapper(
        {
            "shards": {"train": [str(tar_path)]},
            "required_extensions": ["png", "json"],
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "auto_split": False,
        }
    )

    loader = wrapper.get_split("train")
    assert loader is not None
    batch = next(iter(loader))

    assert batch["key"] == ["nested/sample-1", "nested/sample-2"]
    assert [members["png"] for members in batch["members"]] == [
        b"image-one",
        b"image-two",
    ]
    assert [members["json"] for members in batch["members"]] == [
        b'{"id":1}',
        b'{"id":2}',
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "samples.tar",
        "samples.tar.idx.json",
    ]


def test_tar_index_rejects_changed_shard_size(tmp_path: Path) -> None:
    tar_path = tmp_path / "samples.tar"
    index_path = tmp_path / "samples.tar.idx.json"
    _write_tar(tar_path, {"sample": {"png": b"image", "json": b"{}"}})
    TarShardIndex.build(tar_path).write(index_path)

    with open(tar_path, "ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="size mismatch"):
        TarShardIndex.load(index_path, tar_path)


def test_tar_wrapper_rejects_incomplete_required_pairs(tmp_path: Path) -> None:
    tar_path = tmp_path / "incomplete.tar"
    _write_tar(tar_path, {"sample": {"png": b"image"}})
    wrapper = _BytesTarWrapper(
        {
            "shards": {"train": [str(tar_path)]},
            "required_extensions": ["png", "json"],
            "auto_split": False,
        }
    )

    with pytest.raises(ValueError, match="missing extensions"):
        wrapper.get_split("train")


def _sampler_records() -> list[dict[str, str]]:
    return [
        {
            "path": f"{group}-{shard}-{sample}",
            "shard_path": f"{group}-{shard}.tar",
            "group": group,
        }
        for group in ("attack", "real")
        for shard in range(2)
        for sample in range(4)
    ]


def test_round_robin_sampler_balances_batches_and_partitions_ranks() -> None:
    records = _sampler_records()
    common = {
        "records": records,
        "batch_size": 4,
        "group_pattern": ["attack", "real"],
        "shuffle": False,
        "shuffle_within_batch": False,
        "drop_last": True,
        "world_size": 2,
    }
    rank_zero = RoundRobinTarBatchSampler(rank=0, **common)
    rank_one = RoundRobinTarBatchSampler(rank=1, **common)

    zero_batches = list(rank_zero)
    one_batches = list(rank_one)
    assert len(zero_batches) == len(one_batches) == 2
    assert not ({index for batch in zero_batches for index in batch} & {
        index for batch in one_batches for index in batch
    })
    for batch in [*zero_batches, *one_batches]:
        assert [records[index]["group"] for index in batch] == [
            "attack",
            "real",
            "attack",
            "real",
        ]
        assert len({records[index]["shard_path"] for index in batch}) == 4


def test_round_robin_sampler_is_repeatable_and_changes_by_epoch() -> None:
    records = _sampler_records()
    sampler = RoundRobinTarBatchSampler(
        records,
        4,
        group_pattern=["attack", "real"],
        seed=7,
        drop_last=True,
    )

    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_multi_gpu_accelerator_preserves_distributed_tar_batch_sampler() -> None:
    records = _sampler_records()
    sampler = RoundRobinTarBatchSampler(
        records,
        4,
        group_pattern=["attack", "real"],
        world_size=2,
        rank=0,
        drop_last=True,
    )
    loader = DataLoader(records, batch_sampler=sampler)
    accelerator = object.__new__(MultiGPUAccelerator)
    accelerator.samplers = {}

    *_, prepared = accelerator.prepare(dataloaders={"train": loader})

    assert prepared["train"] is loader
    assert accelerator.samplers["train"] is sampler
    accelerator.set_sampler_epoch(3)
    assert sampler.epoch == 3
