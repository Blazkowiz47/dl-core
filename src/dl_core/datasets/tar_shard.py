"""Indexed, grouped-sample datasets for uncompressed tar shards."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
import tempfile
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from dl_core.core.base_dataset import BaseWrapper
from dl_core.core.config_metadata import config_field


@dataclass(frozen=True)
class TarMemberReference:
    """Byte location of one regular member in an uncompressed tar file."""

    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class TarSampleReference:
    """Members sharing one WebDataset-style sample key."""

    key: str
    members: dict[str, TarMemberReference]


@dataclass(frozen=True)
class TarShardIndex:
    """Serializable random-access index for one uncompressed tar shard."""

    format_version: int
    tar_size: int
    tar_sha256: str | None
    samples: tuple[TarSampleReference, ...]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def build(cls, tar_path: str | Path, *, checksum: bool = False) -> TarShardIndex:
        """Scan an uncompressed tar and return its grouped member index."""

        path = Path(tar_path)
        if path.suffix.lower() != ".tar":
            raise ValueError(f"Indexed tar shards must use uncompressed .tar files: {path}")

        grouped: dict[str, dict[str, TarMemberReference]] = defaultdict(dict)
        try:
            archive = tarfile.open(path, mode="r:")
        except tarfile.ReadError as exc:
            raise ValueError(f"Could not read uncompressed tar shard: {path}") from exc

        with archive:
            for member in archive:
                if not member.isfile() or "." not in member.name:
                    continue
                key, extension = member.name.rsplit(".", 1)
                extension = extension.lower()
                if extension in grouped[key]:
                    raise ValueError(
                        f"Duplicate .{extension} member for sample {key!r} in {path}"
                    )
                grouped[key][extension] = TarMemberReference(
                    name=member.name,
                    offset=member.offset_data,
                    size=member.size,
                )

        samples = tuple(
            TarSampleReference(key=key, members=grouped[key])
            for key in sorted(grouped)
        )
        return cls(
            format_version=1,
            tar_size=path.stat().st_size,
            tar_sha256=cls._sha256(path) if checksum else None,
            samples=samples,
        )

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        tar_path: str | Path,
        *,
        validate_checksum: bool = False,
    ) -> TarShardIndex:
        """Load an index and validate it against its local tar shard."""

        index_file = Path(index_path)
        shard_path = Path(tar_path)
        with open(index_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported tar index format in {index_file}")

        members_by_sample: list[TarSampleReference] = []
        for sample in payload.get("samples", []):
            members = {
                extension: TarMemberReference(**member)
                for extension, member in sample["members"].items()
            }
            members_by_sample.append(
                TarSampleReference(key=sample["key"], members=members)
            )

        index = cls(
            format_version=payload["format_version"],
            tar_size=int(payload["tar_size"]),
            tar_sha256=payload.get("tar_sha256"),
            samples=tuple(members_by_sample),
        )
        actual_size = shard_path.stat().st_size
        if actual_size != index.tar_size:
            raise ValueError(
                f"Tar index size mismatch for {shard_path}: "
                f"expected {index.tar_size}, found {actual_size}"
            )
        if validate_checksum:
            if not index.tar_sha256:
                raise ValueError(f"Tar index has no checksum: {index_file}")
            actual_checksum = cls._sha256(shard_path)
            if actual_checksum != index.tar_sha256:
                raise ValueError(f"Tar index checksum mismatch for {shard_path}")
        return index

    def write(self, index_path: str | Path) -> Path:
        """Write the index atomically as JSON."""

        destination = Path(index_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.format_version,
            "tar_size": self.tar_size,
            "tar_sha256": self.tar_sha256,
            "samples": [
                {
                    "key": sample.key,
                    "members": {
                        extension: asdict(member)
                        for extension, member in sample.members.items()
                    },
                }
                for sample in self.samples
            ],
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return destination


class TarHandlePool:
    """Process-local LRU pool of open uncompressed tar file handles."""

    def __init__(self, max_open_shards: int = 8) -> None:
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be greater than zero")
        self.max_open_shards = max_open_shards
        self._process_id = os.getpid()
        self._handles: OrderedDict[Path, Any] = OrderedDict()

    def _handle(self, tar_path: Path) -> Any:
        if self._process_id != os.getpid():
            self.close()
            self._process_id = os.getpid()
        if tar_path in self._handles:
            handle = self._handles.pop(tar_path)
            self._handles[tar_path] = handle
            return handle

        handle = open(tar_path, "rb")
        self._handles[tar_path] = handle
        if len(self._handles) > self.max_open_shards:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        return handle

    def read(
        self,
        tar_path: str | Path,
        members: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, bytes]:
        """Read indexed members directly from one tar shard."""

        path = Path(tar_path)
        handle = self._handle(path)
        result: dict[str, bytes] = {}
        for extension, member in members.items():
            offset = int(member["offset"])
            size = int(member["size"])
            if offset < 0 or size < 0:
                raise ValueError(f"Invalid member range in tar shard {path}")
            handle.seek(offset)
            payload = handle.read(size)
            if len(payload) != size:
                raise OSError(
                    f"Short read for {member['name']!r} in {path}: "
                    f"expected {size} bytes, found {len(payload)}"
                )
            result[extension] = payload
        return result

    def close(self) -> None:
        """Close all handles owned by the current process."""

        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_process_id"] = os.getpid()
        state["_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()


class TarShardDataset(Dataset):
    """Map-style dataset that reads grouped bytes from indexed tar shards."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        transform: Callable[[dict[str, Any]], Any],
        *,
        max_open_shards: int = 8,
    ) -> None:
        self.records = records
        self.transform = transform
        self.handle_pool = TarHandlePool(max_open_shards=max_open_shards)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        record = self.records[index]
        sample = {
            key: value for key, value in record.items() if key != "member_refs"
        }
        sample["members"] = self.handle_pool.read(
            record["shard_path"], record["member_refs"]
        )
        return self.transform(sample)


class RoundRobinTarBatchSampler(Sampler[list[int]]):
    """Build deterministic batches while rotating reads across tar shards."""

    is_distributed = True

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        batch_size: int,
        *,
        group_pattern: Sequence[str] | None = None,
        group_key: str = "group",
        shard_key: str = "shard_path",
        shuffle: bool = True,
        shuffle_within_batch: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        distributed_drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be within the configured world_size")
        if group_pattern is not None and not group_pattern:
            raise ValueError("group_pattern cannot be empty")
        self.records = records
        self.batch_size = batch_size
        self.group_pattern = tuple(group_pattern) if group_pattern else None
        self.group_key = group_key
        self.shard_key = shard_key
        self.shuffle = shuffle
        self.shuffle_within_batch = shuffle_within_batch
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.distributed_drop_last = distributed_drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ordering for a data cycle or epoch."""

        self.epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        if self.group_pattern:
            for index, record in enumerate(self.records):
                group = record.get(self.group_key)
                if group in self.group_pattern:
                    grouped_indices[str(group)].append(index)
            missing_groups = set(self.group_pattern) - set(grouped_indices)
            if missing_groups:
                raise ValueError(
                    f"No tar samples found for groups: {sorted(missing_groups)}"
                )
        else:
            grouped_indices["__all__"] = list(range(len(self.records)))

        pools: dict[str, dict[str, list[int]]] = {}
        shard_cycles: dict[str, deque[str]] = {}
        remaining: Counter[str] = Counter()
        for group, indices in grouped_indices.items():
            by_shard: dict[str, list[int]] = defaultdict(list)
            for index in indices:
                by_shard[str(self.records[index][self.shard_key])].append(index)
            shard_names = sorted(by_shard)
            if self.shuffle:
                rng.shuffle(shard_names)
            for shard_indices in by_shard.values():
                if self.shuffle:
                    rng.shuffle(shard_indices)
                else:
                    shard_indices.reverse()
            pools[group] = by_shard
            shard_cycles[group] = deque(shard_names)
            remaining[group] = len(indices)

        def take(group: str) -> int:
            shards = shard_cycles[group]
            while shards:
                shard = shards[0]
                shard_indices = pools[group][shard]
                if shard_indices:
                    index = shard_indices.pop()
                    shards.rotate(-1)
                    remaining[group] -= 1
                    return index
                shards.popleft()
            raise RuntimeError(f"Tar sampler exhausted group {group!r} unexpectedly")

        batches: list[list[int]] = []
        if self.group_pattern:
            slots = list(islice(cycle(self.group_pattern), self.batch_size))
            needed = Counter(slots)
            while all(remaining[group] >= count for group, count in needed.items()):
                batch = [take(group) for group in slots]
                if self.shuffle_within_batch:
                    rng.shuffle(batch)
                batches.append(batch)

            if not self.drop_last:
                while any(remaining[group] for group in self.group_pattern):
                    batch = [take(group) for group in slots if remaining[group] > 0]
                    if not batch:
                        break
                    if self.shuffle_within_batch:
                        rng.shuffle(batch)
                    batches.append(batch)
        else:
            while remaining["__all__"] > 0:
                batch = [
                    take("__all__")
                    for _ in range(min(self.batch_size, remaining["__all__"]))
                ]
                if len(batch) < self.batch_size and self.drop_last:
                    break
                if self.shuffle_within_batch:
                    rng.shuffle(batch)
                batches.append(batch)

        return batches

    def _rank_batches(self) -> list[list[int]]:
        batches = self._global_batches()
        if self.world_size == 1:
            return batches
        remainder = len(batches) % self.world_size
        if remainder and self.distributed_drop_last:
            batches = batches[: len(batches) - remainder]
        elif remainder and batches:
            original = tuple(batches)
            for index in range(self.world_size - remainder):
                batches.append(list(original[index % len(original)]))
        return batches[self.rank :: self.world_size]

    def __iter__(self):
        return iter(self._rank_batches())

    def __len__(self) -> int:
        return len(self._rank_batches())


class TarShardWrapper(BaseWrapper):
    """BaseWrapper for grouped samples stored in indexed local tar shards."""

    CONFIG_FIELDS = BaseWrapper.CONFIG_FIELDS + [
        config_field(
            "shards",
            "list[str | dict] | dict[str, list[str | dict]]",
            "Explicit tar shards, optionally split-specific and annotated with metadata.",
        ),
        config_field(
            "shard_patterns",
            "str | list[str] | dict[str, str | list[str]]",
            "Glob patterns used to discover tar shards beneath rdir.",
        ),
        config_field(
            "required_extensions",
            "list[str]",
            "Member extensions required for every grouped sample.",
            default=[],
        ),
        config_field(
            "index_suffix",
            "str",
            "Suffix appended to each tar path for its sidecar index.",
            default=".idx.json",
        ),
        config_field(
            "create_index",
            "bool",
            "Create missing local tar indexes automatically.",
            default=True,
        ),
        config_field(
            "index_checksum",
            "bool",
            "Store and validate SHA-256 tar checksums in indexes.",
            default=False,
        ),
        config_field(
            "strict_pairs",
            "bool",
            "Fail when a grouped sample is missing a required extension.",
            default=True,
        ),
        config_field(
            "max_open_shards",
            "int",
            "Maximum tar handles retained inside each DataLoader worker.",
            default=8,
        ),
        config_field(
            "batch_sampler",
            "dict",
            "Optional round-robin tar batch sampler configuration.",
        ),
    ]

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.required_extensions = {
            str(extension).lower().lstrip(".")
            for extension in self.config.get("required_extensions", [])
        }
        self.index_suffix = str(self.config.get("index_suffix", ".idx.json"))
        self.create_index = bool(self.config.get("create_index", True))
        self.index_checksum = bool(self.config.get("index_checksum", False))
        self.strict_pairs = bool(self.config.get("strict_pairs", True))
        self.max_open_shards = int(self.config.get("max_open_shards", 8))
        self.batch_sampler_config = self.config.get("batch_sampler", {})
        self._tar_batch_samplers: dict[str, RoundRobinTarBatchSampler] = {}

    @property
    def file_extensions(self) -> list[str]:
        """Return the only archive format supported by indexed mode."""

        return ["*.tar"]

    @property
    def shard_root(self) -> Path:
        """Return the root used to resolve relative local shard paths."""

        return Path(self.rdir or ".").expanduser()

    def get_configured_shards(self, split: str) -> list[dict[str, Any]]:
        """Normalize explicit split-specific shard configuration."""

        configured = self.config.get("shards", [])
        if isinstance(configured, dict) and "path" not in configured:
            configured = configured.get(split, [])
        if isinstance(configured, (str, Path)) or (
            isinstance(configured, dict) and "path" in configured
        ):
            configured = [configured]
        return [
            dict(shard) if isinstance(shard, dict) else {"path": str(shard)}
            for shard in configured
        ]

    def get_shards(self, split: str) -> list[dict[str, Any]]:
        """Return resolved local shard records for one split."""

        configured = self.get_configured_shards(split)
        if configured:
            resolved = []
            for shard in configured:
                path = Path(shard["path"]).expanduser()
                if not path.is_absolute():
                    path = self.shard_root / path
                resolved.append({**shard, "path": str(path)})
            return resolved

        patterns = self.config.get("shard_patterns", {})
        if isinstance(patterns, dict):
            patterns = patterns.get(split, [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if not patterns:
            patterns = [f"{split}/**/*.tar"]
        paths = {
            path
            for pattern in patterns
            for path in self.shard_root.glob(pattern)
            if path.is_file() and path.suffix.lower() == ".tar"
        }
        return [{"path": str(path)} for path in sorted(paths)]

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Load indexes and return random-access sample records."""

        records: list[dict[str, Any]] = []
        for shard in self.get_shards(split):
            tar_path = Path(shard["path"])
            if tar_path.suffix.lower() != ".tar":
                raise ValueError(f"Indexed shards must be uncompressed .tar files: {tar_path}")
            index_path = Path(
                shard.get("index_path", f"{tar_path}{self.index_suffix}")
            )
            if not index_path.exists():
                if not self.create_index:
                    raise FileNotFoundError(f"Missing tar index: {index_path}")
                TarShardIndex.build(
                    tar_path, checksum=self.index_checksum
                ).write(index_path)
            index = TarShardIndex.load(
                index_path,
                tar_path,
                validate_checksum=self.index_checksum,
            )
            shard_metadata = {
                key: value
                for key, value in shard.items()
                if key not in {"path", "index_path"}
            }
            for sample in index.samples:
                available = set(sample.members)
                missing = self.required_extensions - available
                if missing:
                    if self.strict_pairs:
                        raise ValueError(
                            f"Sample {sample.key!r} in {tar_path} is missing "
                            f"extensions: {sorted(missing)}"
                        )
                    continue
                records.append(
                    {
                        **shard_metadata,
                        "path": f"{tar_path}::{sample.key}",
                        "shard_path": str(tar_path),
                        "key": sample.key,
                        "member_refs": {
                            extension: asdict(member)
                            for extension, member in sample.members.items()
                        },
                    }
                )
        return records

    def build_dataset(self, data: list[dict], split: str) -> Dataset:
        """Build the tar-aware map dataset for one split."""

        from functools import partial

        return TarShardDataset(
            data,
            partial(self.transform, split=split),
            max_open_shards=self.max_open_shards,
        )

    def build_batch_sampler(
        self,
        dataset: Dataset,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> RoundRobinTarBatchSampler | None:
        """Build the optional rank-aware tar batch sampler."""

        del dataset
        config = self.batch_sampler_config
        if not config or not config.get("enabled", True):
            return None
        splits = config.get("splits", ["train"])
        if split not in splits:
            return None
        sampler_type = config.get("type", "round_robin_tar")
        if sampler_type not in {"round_robin_tar", "round_robin"}:
            raise ValueError(f"Unsupported tar batch sampler: {sampler_type}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        sampler = RoundRobinTarBatchSampler(
            self.sampled_files_list[split],
            batch_size,
            group_pattern=config.get("group_pattern"),
            group_key=config.get("group_key", "group"),
            shard_key=config.get("shard_key", "shard_path"),
            shuffle=bool(config.get("shuffle", shuffle)),
            shuffle_within_batch=bool(config.get("shuffle_within_batch", True)),
            drop_last=bool(config.get("drop_last", drop_last)),
            seed=int(config.get("seed", self.seed)),
            rank=rank,
            world_size=world_size,
            distributed_drop_last=bool(config.get("distributed_drop_last", True)),
        )
        self._tar_batch_samplers[split] = sampler
        return sampler

    def set_epoch(self, epoch: int) -> None:
        """Advance the wrapper and any constructed tar batch samplers."""

        super().set_epoch(epoch)
        for sampler in self._tar_batch_samplers.values():
            sampler.set_epoch(epoch)


__all__ = [
    "RoundRobinTarBatchSampler",
    "TarHandlePool",
    "TarMemberReference",
    "TarSampleReference",
    "TarShardDataset",
    "TarShardIndex",
    "TarShardWrapper",
]
