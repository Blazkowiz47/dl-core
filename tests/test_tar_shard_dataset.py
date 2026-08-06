"""Tests for optional WebDataset-backed tar datasets."""

from __future__ import annotations

import builtins
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist

from dl_core.accelerators.multi_gpu import MultiGPUAccelerator
from dl_core.datasets import TarShardWrapper


def _write_tar(path: Path, samples: dict[str, dict[str, bytes]]) -> None:
    mode = "w:gz" if path.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(path, mode) as archive:
        for key, members in samples.items():
            for extension, payload in members.items():
                info = tarfile.TarInfo(f"{key}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


class _BytesTarWrapper(TarShardWrapper):
    """Concrete tar wrapper that returns grouped bytes unchanged."""

    def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
        return {**file_dict, "split": split}


def _wrapper(tar_path: Path, **config: Any) -> _BytesTarWrapper:
    return _BytesTarWrapper(
        {
            "shards": {
                "train": [{"path": str(tar_path), "group": "real"}]
            },
            "required_extensions": ["png", "json"],
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "auto_split": False,
            **config,
        }
    )


@pytest.mark.parametrize("suffix", [".tar", ".tar.gz"])
def test_tar_wrapper_streams_grouped_members_without_extraction(
    tmp_path: Path,
    suffix: str,
) -> None:
    tar_path = tmp_path / f"samples{suffix}"
    _write_tar(
        tar_path,
        {
            "nested/sample-1": {"png": b"image-one", "json": b'{"id":1}'},
            "nested/sample-2": {"png": b"image-two", "json": b'{"id":2}'},
        },
    )

    loader = _wrapper(tar_path).get_split("train")
    assert loader is not None
    batch = next(iter(loader))

    assert batch["key"] == ["nested/sample-1", "nested/sample-2"]
    assert batch["group"] == ["real", "real"]
    assert [members["png"] for members in batch["members"]] == [
        b"image-one",
        b"image-two",
    ]
    assert [members["json"] for members in batch["members"]] == [
        b'{"id":1}',
        b'{"id":2}',
    ]
    assert list(tmp_path.iterdir()) == [tar_path]


def test_tar_wrapper_rejects_incomplete_required_pairs(tmp_path: Path) -> None:
    tar_path = tmp_path / "incomplete.tar"
    _write_tar(tar_path, {"sample": {"png": b"image"}})
    loader = _wrapper(tar_path, batch_size=1).get_split("train")
    assert loader is not None

    with pytest.raises(ValueError, match="missing extensions"):
        next(iter(loader))


def test_tar_wrapper_skips_incomplete_pairs_when_not_strict(tmp_path: Path) -> None:
    tar_path = tmp_path / "mixed.tar"
    _write_tar(
        tar_path,
        {
            "complete": {"png": b"image", "json": b"{}"},
            "incomplete": {"png": b"image"},
        },
    )
    loader = _wrapper(tar_path, strict_pairs=False).get_split("train")
    assert loader is not None

    batch = next(iter(loader))
    assert batch["key"] == ["complete"]


def test_webdataset_is_required_only_when_tar_wrapper_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tar_path = tmp_path / "samples.tar"
    _write_tar(tar_path, {"sample": {"png": b"image", "json": b"{}"}})
    original_import = builtins.__import__

    def import_without_webdataset(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "webdataset":
            raise ModuleNotFoundError(name="webdataset")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_webdataset)
    with pytest.raises(ImportError, match=r"deep-learning-core\[webdataset\]"):
        _wrapper(tar_path).get_split("train")


def test_multi_gpu_accelerator_preserves_webdataset_loader(tmp_path: Path) -> None:
    tar_path = tmp_path / "samples.tar"
    _write_tar(tar_path, {"sample": {"png": b"image", "json": b"{}"}})
    loader = _wrapper(tar_path, batch_size=1).get_split("train")
    assert loader is not None
    accelerator = object.__new__(MultiGPUAccelerator)
    accelerator.samplers = {}

    *_, prepared = accelerator.prepare(dataloaders={"train": loader})

    assert prepared["train"] is loader
    assert accelerator.samplers == {}


def test_webdataset_splits_shards_between_distributed_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_paths = []
    for index in range(4):
        tar_path = tmp_path / f"samples-{index}.tar"
        _write_tar(
            tar_path,
            {f"sample-{index}": {"png": b"image", "json": b"{}"}},
        )
        shard_paths.append(str(tar_path))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: 2)
    rank_samples = []
    for rank in range(2):
        monkeypatch.setattr(
            dist,
            "get_rank",
            lambda *args, current_rank=rank, **kwargs: current_rank,
        )
        wrapper = _BytesTarWrapper(
            {
                "shards": {"train": shard_paths},
                "required_extensions": ["png", "json"],
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "auto_split": False,
            }
        )
        loader = wrapper.get_split("train")
        assert loader is not None
        rank_samples.append({batch["key"][0] for batch in loader})

    assert rank_samples[0].isdisjoint(rank_samples[1])
    assert rank_samples[0] | rank_samples[1] == {
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    }
