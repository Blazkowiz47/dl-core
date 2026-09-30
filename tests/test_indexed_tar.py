"""Exercise indexed reads, source mixing, and completed-batch shard progress."""

from __future__ import annotations

import io
import os
import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from dl_core.datasets import (
    IndexedTarDataset,
    IndexedTarSampler,
    ShardProgress,
    TarShardWrapper,
)


def _write_tar(path: Path, count: int, prefix: str = "sample") -> dict[str, bytes]:
    payloads = {}
    with tarfile.open(path, "w:gz" if path.suffix == ".gz" else "w") as archive:
        for number in range(count):
            key = f"nested/{prefix}-{number}"
            payloads[key] = bytes([number % 256]) * (number * 513 + 1)
            for extension, payload in (
                ("png", payloads[key]),
                ("json", str(number).encode()),
            ):
                info = tarfile.TarInfo(f"{key}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return payloads


def _worker_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": sample["key"],
        "shard_id": sample["shard_id"],
        "image": sample["members"]["png"],
        "metadata": sample["members"]["json"],
        "pid": os.getpid(),
    }


class _IndexedWrapper(TarShardWrapper):
    def build_dataset(self, data: list[dict], split: str) -> IndexedTarDataset:
        return self.build_indexed_dataset(data, split)

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any] | None:
        if self.config.get("skip_first") and file_dict["key"].endswith("-0"):
            return None
        return {"key": file_dict["key"]}


def test_variable_members_and_long_names(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    payloads = _write_tar(path, 9, prefix="x" * 130)
    dataset = IndexedTarDataset([path], required_extensions=["png", "json"])
    for index in reversed(range(len(dataset))):
        sample = dataset[index]
        assert sample["members"]["png"] == payloads[sample["key"]]
        assert sample["members"]["json"] == str(index).encode()
    assert dataset.shard_totals == {str(path): 9}
    dataset.close()


def test_index_reuse_and_content_invalidation(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, 2)
    first = IndexedTarDataset([path], index_dir=tmp_path / "indexes")
    first[0]
    original_open = tarfile.open
    with monkeypatch.context() as scoped:
        scoped.setattr(
            tarfile, "open", lambda *args, **kwargs: pytest.fail("Index was rebuilt")
        )
        second = IndexedTarDataset([path], index_dir=tmp_path / "indexes")
        assert second[1]["key"] == "nested/sample-1"
        second.close()
    _write_tar(path, 3, prefix="replacement")
    rebuilt = IndexedTarDataset([path], index_dir=tmp_path / "indexes")
    assert len(rebuilt) == 3
    assert rebuilt[0]["key"] == "nested/replacement-0"
    with pytest.raises(RuntimeError, match="Tar changed"):
        first[0]
    assert tarfile.open is original_open
    first.close()
    rebuilt.close()


def test_bad_cached_index_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, 2)
    IndexedTarDataset([path], index_dir=tmp_path / "indexes").close()
    next((tmp_path / "indexes").glob("*.json")).write_text("broken")
    assert len(IndexedTarDataset([path], index_dir=tmp_path / "indexes")) == 2


def test_atomic_replacement_invalidates_an_open_reader(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, 2)
    dataset = IndexedTarDataset([path])
    dataset[0]
    replacement = tmp_path / "replacement.tar"
    _write_tar(replacement, 2, prefix="replacement")
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="Tar changed"):
        dataset[1]
    assert not dataset._handles


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_workers_share_one_shard_without_duplicate_samples(
    tmp_path: Path, workers: int
) -> None:
    path = tmp_path / "samples.tar"
    payloads = _write_tar(path, 24)
    dataset = IndexedTarDataset([path], transform=_worker_sample)
    # An already-open parent handle must not be copied into spawned workers.
    dataset[0]
    options = {"multiprocessing_context": "spawn", "timeout": 45} if workers else {}
    loader = DataLoader(dataset, batch_size=3, num_workers=workers, **options)
    seen = []
    pids = set()
    for batch in loader:
        pids.update(batch["pid"].tolist())
        for key, payload, metadata in zip(
            batch["key"], batch["image"], batch["metadata"]
        ):
            seen.append(key)
            assert payload == payloads[key]
            assert metadata == key.rsplit("-", 1)[1].encode()
    assert Counter(seen) == Counter(payloads.keys())
    assert len(pids) == (workers or 1)
    dataset.close()


def test_bounded_handles_and_pickling(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.tar" for index in range(3)]
    for path in paths:
        _write_tar(path, 1)
    dataset = IndexedTarDataset(paths, max_open_shards=1)
    for index in range(3):
        assert dataset[index]["key"] == "nested/sample-0"
        assert len(dataset._handles) == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._handles
    assert restored[0]["members"]["png"] == b"\0"
    dataset.close()
    restored.close()


def test_workers_mix_several_shards(tmp_path: Path) -> None:
    paths = [tmp_path / f"shard-{index}.tar" for index in range(4)]
    payloads = {}
    for index, path in enumerate(paths):
        payloads.update(_write_tar(path, 4, prefix=f"shard-{index}"))
    dataset = IndexedTarDataset(paths, transform=_worker_sample, max_open_shards=2)
    sampler = IndexedTarSampler(dataset, generator=torch.Generator().manual_seed(5))
    seen = []
    shards = set()
    for batch in DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        multiprocessing_context="spawn",
        timeout=45,
    ):
        seen.extend(batch["key"])
        shards.update(batch["shard_id"])
        for key, payload in zip(batch["key"], batch["image"]):
            assert payload == payloads[key]
    assert set(shards) == {str(path) for path in paths}
    assert Counter(seen) == Counter(payloads.keys())
    dataset.close()


def test_pair_validation_selection_and_compression(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, 3)
    selected = IndexedTarDataset(
        [{"path": str(path), "sample_keys": ["nested/sample-1"]}]
    )
    assert len(selected) == 1
    assert selected[0]["key"] == "nested/sample-1"
    with pytest.raises(ValueError, match="missing extensions"):
        IndexedTarDataset([path], required_extensions=["absent"])
    assert (
        len(
            IndexedTarDataset(
                [path], required_extensions=["absent"], strict_pairs=False
            )
        )
        == 0
    )
    compressed = tmp_path / "samples.tar.gz"
    _write_tar(compressed, 1)
    with pytest.raises(tarfile.ReadError):
        IndexedTarDataset([compressed])
    with pytest.raises(ValueError, match="max_open_shards"):
        IndexedTarDataset([path], max_open_shards=0)


def test_finite_and_weighted_repeated_sampling(tmp_path: Path) -> None:
    paths = [tmp_path / "small.tar", tmp_path / "large.tar"]
    _write_tar(paths[0], 1, prefix="small")
    _write_tar(paths[1], 9, prefix="large")
    dataset = IndexedTarDataset(
        [
            {"path": str(paths[0]), "source_name": "small", "source_weight": 0.7},
            {"path": str(paths[1]), "source_name": "large", "source_weight": 0.3},
        ]
    )
    finite = list(
        IndexedTarSampler(dataset, generator=torch.Generator().manual_seed(11))
    )
    assert sorted(finite) == list(range(10))
    repeated = list(
        IndexedTarSampler(
            dataset,
            replacement=True,
            num_samples=2000,
            generator=torch.Generator().manual_seed(11),
        )
    )
    small_fraction = repeated.count(0) / len(repeated)
    assert 0.65 < small_fraction < 0.75
    assert repeated == list(
        IndexedTarSampler(
            dataset,
            replacement=True,
            num_samples=2000,
            generator=torch.Generator().manual_seed(11),
        )
    )
    with pytest.raises(ValueError, match="requires num_samples"):
        IndexedTarSampler(dataset, replacement=True)
    with pytest.raises(ValueError, match="exceed"):
        IndexedTarSampler(dataset, num_samples=11)


def test_wrapper_progress_and_skipped_transforms(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, 3)
    wrapper = _IndexedWrapper(
        {
            "shards": {"train": [str(path)]},
            "auto_split": False,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "skip_first": True,
            "track_shard_progress": True,
            "indexed_tar": {"index_dir": str(tmp_path / "indexes")},
        }
    )
    wrapper.reset_shard_progress({str(path): 2})
    batches = list(wrapper.get_split("train"))
    assert batches[0] == {}
    for batch in batches[1:]:
        wrapper.record_shard_consumption(batch["shard_id"])
    assert wrapper.get_shard_progress(str(path)) == {
        "consumed": 2,
        "total": 2,
        "fraction": 1.0,
    }
    wrapper.reset_shard_progress({str(path): 2})
    assert wrapper.get_shard_progress(str(path))["consumed"] == 0


def test_progress_independent_shards_unknown_totals_and_validation() -> None:
    progress = ShardProgress({"a": 4, "b": 10, "unknown": None})
    progress.record(["a", "b", "a", "unknown"])
    assert progress.get("a")["fraction"] == 0.5
    assert progress.get("b")["fraction"] == 0.1
    assert progress.get("unknown")["fraction"] is None
    with pytest.raises(KeyError):
        progress.record(["a", "unplanned"])
    assert progress.get("a")["consumed"] == 2
    with pytest.raises(TypeError):
        progress.record("a")
    with pytest.raises(ValueError):
        ShardProgress({"a": 0})
    progress._owner_pid = -1
    with pytest.raises(RuntimeError, match="owning training process"):
        progress.record(["a"])
