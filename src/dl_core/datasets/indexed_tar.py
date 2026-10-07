"""Indexed plain tar samples read by the normal DataLoader worker pool."""

from __future__ import annotations

import logging
import math
import os
import random
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Dataset, Sampler

from dl_core.datasets._tar_index import iter_tar_indexes


class IndexedTarDataset(Dataset):
    """Read grouped members from local, uncompressed tar files without extraction.

    Shard records accept path, shard_id, source_name, source_weight, and arbitrary
    metadata. Optional sample_keys selects an eligible subset of a shard. Each
    process opens its own bounded set of file handles. Call close() when finished.
    index_workers bounds startup builds separately from DataLoader workers.
    """

    def __init__(
        self,
        shards: Iterable[str | Path | dict[str, Any]],
        *,
        transform: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        required_extensions: Iterable[str] = (),
        strict_pairs: bool = True,
        index_dir: str | Path | None = None,
        index_workers: int = 4,
        max_open_shards: int = 8,
        track_shard_progress: bool = False,
        logger: logging.Logger | None = None,
        index_label: str = "dataset",
    ) -> None:
        if (
            isinstance(max_open_shards, bool)
            or not isinstance(max_open_shards, int)
            or max_open_shards < 1
        ):
            raise ValueError("max_open_shards must be a positive integer")
        if (
            isinstance(index_workers, bool)
            or not isinstance(index_workers, int)
            or index_workers < 1
        ):
            raise ValueError("index_workers must be a positive integer")
        self.transform = transform
        self.max_open_shards = max_open_shards
        self.track_shard_progress = track_shard_progress
        self.shards: list[dict[str, Any]] = []
        self.shard_totals: dict[str, int] = {}
        self.source_indices: dict[str, list[int]] = {}
        self.source_weights: dict[str, float] = {}
        self._samples: list[tuple[int, str, dict[str, list[int]]]] = []
        self._fingerprints: list[list[int]] = []
        self._handles: OrderedDict[int, BinaryIO] = OrderedDict()
        self._pid = os.getpid()
        self.index_stats: dict[str, Any] = {}
        required = {
            str(extension).lower().lstrip(".") for extension in required_extensions
        }
        index_root = Path(index_dir).expanduser() if index_dir is not None else None
        if index_root is not None:
            index_root.mkdir(parents=True, exist_ok=True)

        normalized = []
        for configured in shards:
            shard = (
                dict(configured)
                if isinstance(configured, dict)
                else {"path": str(configured)}
            )
            path = Path(shard["path"]).expanduser().resolve()
            source = str(shard.get("source_name", "default"))
            weight = float(shard.get("source_weight", 1))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid indexed source weight for {source}")
            if weight == 0:
                continue
            if source in self.source_weights and self.source_weights[source] != weight:
                raise ValueError(f"Conflicting indexed source weights for {source}")
            self.source_weights[source] = weight
            self.source_indices.setdefault(source, [])
            shard_id = str(
                shard.get(
                    "shard_id", shard.get("source_path", shard.get("public_url", path))
                )
            )
            shard.update(path=str(path), shard_id=shard_id)
            normalized.append(shard)

        indexes = iter_tar_indexes(
            [Path(shard["path"]) for shard in normalized],
            index_root,
            index_workers,
            logger if logger is not None else logging.getLogger(__name__),
            index_label,
            self.index_stats,
        )
        # Closing the iterator also cancels and joins workers on pair-validation
        # errors or interrupts raised while merging results in the parent.
        with closing(indexes):
            for shard_index, index in enumerate(indexes):
                shard = normalized[shard_index]
                path = Path(shard["path"])
                current = path.stat()
                if [
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                    current.st_mtime_ns,
                ] != index.fingerprint:
                    raise RuntimeError(f"Tar changed during indexing: {path}")
                source = str(shard.get("source_name", "default"))
                shard_id = shard["shard_id"]
                self.shards.append(shard)
                self._fingerprints.append(index.fingerprint)
                self.shard_totals.setdefault(shard_id, 0)
                selected = set(shard["sample_keys"]) if "sample_keys" in shard else None
                for key, members in index.samples:
                    if selected is not None and key not in selected:
                        continue
                    missing = required - set(members)
                    if missing:
                        if strict_pairs:
                            raise ValueError(
                                f"Sample {key!r} in {path} is missing extensions: {sorted(missing)}"
                            )
                        continue
                    self.source_indices[source].append(len(self._samples))
                    self._samples.append((shard_index, key, members))
                    self.shard_totals[shard_id] += 1

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard_index, key, offsets = self._samples[index]
        shard = self.shards[shard_index]
        current = os.stat(shard["path"])
        if [
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ] != self._fingerprints[shard_index]:
            self.close()
            raise RuntimeError(f"Tar changed after indexing: {shard['path']}")
        handle = self._handles.pop(shard_index, None)
        if handle is None:
            handle = open(shard["path"], "rb")
        stat = os.fstat(handle.fileno())
        if [
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        ] != self._fingerprints[shard_index]:
            handle.close()
            raise RuntimeError(f"Tar changed after indexing: {shard['path']}")
        self._handles[shard_index] = handle
        while len(self._handles) > self.max_open_shards:
            self._handles.popitem(last=False)[1].close()
        members = {}
        for extension, (offset, size) in offsets.items():
            handle.seek(offset)
            payload = handle.read(size)
            if len(payload) != size:
                raise OSError(f"Incomplete indexed tar member for sample {key!r}")
            members[extension] = payload
        metadata = {
            name: value
            for name, value in shard.items()
            if name not in {"path", "public_url", "sample_keys"}
        }
        logical_source = str(shard.get("source_path", shard["path"]))
        sample = {
            **metadata,
            "path": f"{logical_source}::{key}",
            "shard_path": str(shard.get("public_url", shard["path"])),
            "key": key,
            "members": members,
        }
        result = self.transform(sample) if self.transform is not None else sample
        if result is not None and self.track_shard_progress:
            result = {**result, "shard_id": shard["shard_id"]}
        return result

    def close(self) -> None:
        """Close this process's open tar handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()


class IndexedTarSampler(Sampler[int]):
    """Mix sources by source weight, then choose samples within each source.

    Finite sampling visits each selected sample at most once. Repeated sampling
    requires an explicit num_samples budget. This sampler targets one GPU.
    """

    def __init__(
        self,
        dataset: IndexedTarDataset,
        *,
        shuffle: bool = True,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if replacement and num_samples is None:
            raise ValueError("Repeated indexed sampling requires num_samples")
        budget = len(dataset) if num_samples is None else num_samples
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 0
            or (replacement and budget == 0)
        ):
            raise ValueError(
                "num_samples must be a nonnegative integer, or positive for repeated sampling"
            )
        if not replacement and budget > len(dataset):
            raise ValueError("Finite indexed sampling cannot exceed the dataset length")
        self.dataset = dataset
        self.shuffle = shuffle
        self.replacement = replacement
        self.num_samples = budget
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle and not self.replacement:
            yield from range(self.num_samples)
            return
        seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
        )
        rng = random.Random(seed)
        pools = {
            name: list(indices)
            for name, indices in self.dataset.source_indices.items()
            if indices
        }
        if self.num_samples and not pools:
            raise ValueError("Indexed sampler has no eligible samples")
        for indices in pools.values():
            if not self.replacement:
                rng.shuffle(indices)
        for _ in range(self.num_samples):
            sources = list(pools)
            source = rng.choices(
                sources, weights=[self.dataset.source_weights[name] for name in sources]
            )[0]
            if self.replacement:
                yield rng.choice(pools[source])
            else:
                yield pools[source].pop()
                if not pools[source]:
                    del pools[source]
