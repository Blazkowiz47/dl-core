"""Optional WebDataset-backed tar shard datasets."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from dl_core.core.base_dataset import BaseWrapper
from dl_core.core.config_metadata import config_field


class TarShardWrapper(BaseWrapper):
    """BaseWrapper that streams grouped tar samples through WebDataset."""

    CONFIG_FIELDS = BaseWrapper.CONFIG_FIELDS + [
        config_field(
            "shards",
            "list[str | dict] | dict[str, list[str | dict]]",
            "Optional tar shards used by the default build_shard_sources hook.",
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
            "strict_pairs",
            "bool",
            "Fail instead of skipping samples missing a required extension.",
            default=True,
        ),
        config_field(
            "webdataset",
            "dict",
            "WebDataset shard shuffle, sample shuffle, resampling, and cache options.",
        ),
    ]

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.required_extensions = {
            str(extension).lower().lstrip(".")
            for extension in self.config.get("required_extensions", [])
        }
        self.strict_pairs = bool(self.config.get("strict_pairs", True))
        self.webdataset_config = self.config.get("webdataset", {})
        self._webdataset_shuffle: dict[str, bool] = {}

    @property
    def file_extensions(self) -> list[str]:
        """Return tar formats supported by WebDataset."""

        return ["*.tar", "*.tar.gz", "*.tgz"]

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

    def build_shard_sources(self, split: str) -> list[dict[str, Any]]:
        """Build weighted logical shard sources; subclasses may override."""

        shards = self.get_configured_shards(split)
        if not shards:
            patterns = self.config.get("shard_patterns", {})
            if isinstance(patterns, dict):
                patterns = patterns.get(split, [])
            if isinstance(patterns, str):
                patterns = [patterns]
            if not patterns:
                patterns = [
                    f"{split}/**/*.tar",
                    f"{split}/**/*.tar.gz",
                    f"{split}/**/*.tgz",
                ]
            paths = {
                path
                for pattern in patterns
                for path in self.shard_root.glob(pattern)
                if path.is_file()
            }
            shards = [{"path": str(path.absolute())} for path in sorted(paths)]
        return [{"name": split, "weight": 1.0, "shards": shards}]

    def get_shard_sources(self, split: str) -> list[dict[str, Any]]:
        """Resolve logical source shard records into local paths."""

        sources = []
        for source in self.build_shard_sources(split):
            resolved_shards = []
            for configured_shard in source.get("shards", []):
                shard = (
                    dict(configured_shard)
                    if isinstance(configured_shard, dict)
                    else {"path": str(configured_shard)}
                )
                path = Path(shard["path"]).expanduser()
                if not path.is_absolute():
                    path = self.shard_root / path
                resolved_shards.append({**shard, "path": str(path)})
            sources.append({**source, "shards": resolved_shards})
        return sources

    def get_file_list(self, split: str) -> list[dict[str, Any]]:
        """Return weighted shard sources for the WebDataset pipeline."""

        return self.get_shard_sources(split)

    def _webdataset_value(self, name: str, split: str, default: Any) -> Any:
        value = self.webdataset_config.get(name, default)
        if isinstance(value, dict):
            return value.get(split, default)
        return value

    def _transform_webdataset_sample(
        self,
        sample: dict[str, Any],
        *,
        split: str,
        metadata_by_shard: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        source = str(sample.get("__url__", ""))
        key = str(sample.get("__key__", ""))
        members = {
            extension: value
            for extension, value in sample.items()
            if not extension.startswith("__")
        }
        missing = self.required_extensions - set(members)
        if missing:
            if self.strict_pairs:
                raise ValueError(
                    f"Sample {key!r} in {source} is missing extensions: "
                    f"{sorted(missing)}"
                )
            return None

        metadata = metadata_by_shard.get(source, {})
        logical_source = str(metadata.get("source_path", source))
        return self.transform(
            {
                **metadata,
                "path": f"{logical_source}::{key}",
                "shard_path": source,
                "key": key,
                "members": members,
            },
            split,
        )

    def build_dataset(self, data: list[dict], split: str) -> Dataset:
        """Build an optional-dependency WebDataset pipeline for one split."""

        try:
            import webdataset as wds
        except ModuleNotFoundError as exc:
            if exc.name != "webdataset":
                raise
            raise ImportError(
                "TarShardWrapper requires WebDataset. Install it with "
                "`uv add 'deep-learning-core[webdataset]'`."
            ) from exc

        config = self.webdataset_config
        resampled = bool(self._webdataset_value("resampled", split, False))
        shuffle = self._webdataset_shuffle.get(split, self.shuffle[split])
        shard_shuffle = self._webdataset_value("shard_shuffle", split, 100)
        if not shuffle or resampled:
            shard_shuffle = False

        cache_dir = config.get("cache_dir")
        if cache_dir:
            cache_path = Path(cache_dir).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_dir = str(cache_path)

        sample_shuffle = int(self._webdataset_value("sample_shuffle", split, 1000))
        pipelines = []
        weights = []
        for source in data:
            weight = float(source.get("weight", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"WebDataset source {source.get('name', split)!r} has invalid "
                    f"weight {weight}"
                )
            if weight == 0:
                continue

            shards = source.get("shards", [])
            if not shards:
                raise ValueError(
                    f"WebDataset source {source.get('name', split)!r} has no shards"
                )
            paths = [str(shard["path"]) for shard in shards]
            metadata_by_shard = {}
            for shard in shards:
                metadata = {
                    key: value
                    for key, value in shard.items()
                    if key not in {"path", "public_url"}
                }
                metadata.setdefault("source_name", source.get("name", split))
                metadata.setdefault("source_weight", weight)
                metadata_by_shard[str(shard.get("public_url", shard["path"]))] = metadata

            pipeline = wds.WebDataset(
                paths,
                resampled=resampled,
                shardshuffle=shard_shuffle,
                detshuffle=bool(self.deterministic and shard_shuffle),
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                empty_check=bool(
                    self._webdataset_value("empty_check", split, True)
                ),
                cache_dir=cache_dir,
                cache_size=int(config.get("cache_size", -1)),
                seed=self.seed,
            )
            if shuffle and sample_shuffle > 0:
                initial = int(
                    self._webdataset_value(
                        "sample_shuffle_initial",
                        split,
                        min(100, sample_shuffle),
                    )
                )
                if self.deterministic:
                    pipeline = pipeline.compose(
                        wds.detshuffle(
                            sample_shuffle,
                            initial=initial,
                            seed=self.seed,
                        )
                    )
                else:
                    pipeline = pipeline.shuffle(sample_shuffle, initial=initial)
            pipelines.append(
                pipeline.map(
                    partial(
                        self._transform_webdataset_sample,
                        split=split,
                        metadata_by_shard=metadata_by_shard,
                    )
                )
            )
            weights.append(weight)

        if not pipelines:
            raise ValueError(f"No positive-weight WebDataset sources found for {split}")
        dataset = pipelines[0]
        if len(pipelines) > 1:
            dataset = wds.RandomMix(
                pipelines,
                probs=weights,
                longest=bool(
                    self._webdataset_value("mix_longest", split, not resampled)
                ),
            )
        dataset.is_distributed = True
        return dataset

    def _get_split(
        self,
        split: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        shuffle: bool | None = None,
        pin_memory: bool | None = None,
        drop_last: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> DataLoader | None:
        """Pass the resolved shuffle choice into the streaming pipeline."""

        self._webdataset_shuffle[split] = bool(
            self._resolve_split_override(shuffle, self.shuffle[split])
        )
        return super()._get_split(
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
        )


__all__ = ["TarShardWrapper"]
