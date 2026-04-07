"""Built-in label-balanced sampler."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dl_core.core import BaseSampler, register_sampler
from dl_core.core.config_metadata import config_field


@register_sampler("label")
class LabelSampler(BaseSampler):
    """Balance samples by a metadata key using over- or under-sampling."""

    CONFIG_FIELDS = BaseSampler.CONFIG_FIELDS + [
        config_field(
            "key",
            "str",
            "Metadata key used to group samples before balancing.",
            required=True,
        ),
        config_field(
            "mode",
            "str",
            "Balancing strategy: `undersample` or `oversample`.",
            default="undersample",
        ),
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the label-balanced sampler."""
        super().__init__(**kwargs)
        self.key = str(kwargs["key"])
        self.mode = str(kwargs.get("mode", "undersample")).casefold()
        if self.mode not in {"undersample", "oversample"}:
            raise ValueError(
                "LabelSampler mode must be either 'undersample' or 'oversample'."
            )

    def sample_data(
        self,
        files: list[dict[str, Any]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Return a label-balanced subset for the requested split."""
        if not files:
            return []

        groups = self._group_files(files, split)
        if len(groups) < 2:
            return files

        target_count = self._resolve_target_count(groups)
        rng = self._get_epoch_rng()
        sampled_files: list[dict[str, Any]] = []
        for group_name, group_files in groups.items():
            if len(group_files) >= target_count:
                sampled_group = rng.sample(group_files, k=target_count)
            else:
                sampled_group = rng.choices(group_files, k=target_count)
            sampled_files.extend(sampled_group)
            self.logger.debug(
                "[%s] Balanced group %s from %s to %s samples",
                split,
                group_name,
                len(group_files),
                len(sampled_group),
            )
        return sampled_files

    def _group_files(
        self,
        files: list[dict[str, Any]],
        split: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group files by the configured metadata key."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for file_dict in files:
            if self.key not in file_dict:
                self.logger.warning(
                    "[%s] Missing '%s' in file metadata; skipping label sampling",
                    split,
                    self.key,
                )
                return {}
            groups[str(file_dict[self.key])].append(file_dict)
        return dict(groups)

    def _resolve_target_count(
        self,
        groups: dict[str, list[dict[str, Any]]],
    ) -> int:
        """Pick the target class size from the configured balancing mode."""
        group_sizes = [len(group_files) for group_files in groups.values()]
        if self.mode == "undersample":
            return min(group_sizes)
        return max(group_sizes)
