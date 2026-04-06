"""Built-in attack-balanced sampler."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dl_core.core import BaseSampler, register_sampler
from dl_core.core.config_metadata import config_field


@register_sampler("attack")
class AttackSampler(BaseSampler):
    """Balance PAD samples using the per-item ``attack`` metadata key.

    The sampler groups incoming file dictionaries by ``attack`` value. Entries
    whose value matches one of the configured bonafide markers are treated as
    bonafide; every other value is treated as a separate attack group.

    If the ``attack`` key only contains coarse labels like ``real`` and
    ``attack``, this sampler behaves like a class balancer. If it contains more
    specific attack names, it balances each attack subtype evenly.
    """

    CONFIG_FIELDS = BaseSampler.CONFIG_FIELDS + [
        config_field(
            "key",
            "str",
            "Metadata key that stores the PAD attack label.",
            default="attack",
        ),
        config_field(
            "balance_method",
            "str",
            "Sampling strategy. Common values are 'undersample' and "
            "'oversample'.",
            default="undersample",
        ),
        config_field(
            "samples_per_class",
            "int | None",
            "Optional cap for the bonafide class sample count.",
            default=None,
        ),
        config_field(
            "samples_per_attack",
            "int | None",
            "Optional target sample count for each individual attack group.",
            default=None,
        ),
        config_field(
            "force_oversample_attacks",
            "bool",
            "Oversample rare attacks when they have fewer files than requested.",
            default=True,
        ),
        config_field(
            "enforce_equal_classes",
            "bool",
            "Keep bonafide and total attack counts aligned when class caps are "
            "set.",
            default=True,
        ),
        config_field(
            "bonafide_values",
            "list[str]",
            "Attack labels that should be treated as bonafide/live samples.",
            default=["real", "bonafide", "genuine", "live"],
        ),
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the attack-balanced sampler."""
        super().__init__(**kwargs)
        self.key = str(kwargs.get("key", "attack"))
        self.balance_method = str(kwargs.get("balance_method", "undersample"))
        self.samples_per_class = kwargs.get("samples_per_class")
        self.samples_per_attack = kwargs.get("samples_per_attack")
        self.force_oversample_attacks = bool(
            kwargs.get("force_oversample_attacks", True)
        )
        self.enforce_equal_classes = bool(kwargs.get("enforce_equal_classes", True))

        bonafide_values = kwargs.get(
            "bonafide_values",
            ["real", "bonafide", "genuine", "live"],
        )
        self.bonafide_values = {
            self._normalize_group_value(value)
            for value in bonafide_values
        }

    def sample_data(
        self,
        files: list[dict[str, Any]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Return an attack-balanced subset for the requested split."""
        if not files:
            return []

        files_by_group = self._group_files(files, split)
        if not files_by_group:
            return files

        sample_counts = self._calculate_sample_counts(files_by_group)
        rng = self._get_epoch_rng()
        return self._sample_from_groups(
            files_by_group,
            sample_counts,
            split=split,
            original_count=len(files),
            rng=rng,
        )

    def _group_files(
        self,
        files: list[dict[str, Any]],
        split: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group files by attack value, separating bonafide from attacks."""
        files_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for file_dict in files:
            if self.key not in file_dict:
                self.logger.warning(
                    f"[{split}] Missing '{self.key}' in file metadata; "
                    "skipping attack-balanced sampling"
                )
                return {}

            group_name = str(file_dict[self.key])
            if self._is_bonafide_group(group_name):
                files_by_group["real"].append(file_dict)
            else:
                files_by_group[group_name].append(file_dict)

        attack_groups = [name for name in files_by_group if name != "real"]
        if "real" not in files_by_group or not attack_groups:
            self.logger.warning(
                f"[{split}] Missing bonafide or attack groups for key "
                f"'{self.key}' (groups: {list(files_by_group.keys())}); "
                "returning the original sample list"
            )
            return {}

        self.logger.debug(
            f"[{split}] Found {len(attack_groups)} attack groups for '{self.key}': "
            f"{attack_groups}"
        )
        return dict(files_by_group)

    def _calculate_sample_counts(
        self,
        files_by_group: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        """Calculate target sample counts for bonafide and each attack group."""
        bonafide_count = len(files_by_group.get("real", []))
        attack_groups = [name for name in files_by_group if name != "real"]
        num_attack_groups = len(attack_groups)

        if self.samples_per_attack is not None:
            samples_per_attack = int(self.samples_per_attack)
        else:
            total_attacks = sum(len(files_by_group[name]) for name in attack_groups)
            samples_per_attack = total_attacks // num_attack_groups

        total_attack_samples = samples_per_attack * num_attack_groups

        if self.balance_method == "undersample":
            target_bonafide = min(bonafide_count, total_attack_samples)
        else:
            target_bonafide = bonafide_count

        if self.samples_per_class is not None:
            target_bonafide = min(int(self.samples_per_class), target_bonafide)
            if self.enforce_equal_classes and num_attack_groups > 0:
                samples_per_attack = min(
                    samples_per_attack,
                    target_bonafide // num_attack_groups,
                )

        sample_counts = {"real": target_bonafide}
        for group_name in attack_groups:
            sample_counts[group_name] = samples_per_attack
        return sample_counts

    def _sample_from_groups(
        self,
        files_by_group: dict[str, list[dict[str, Any]]],
        sample_counts: dict[str, int],
        *,
        split: str,
        original_count: int,
        rng: Any,
    ) -> list[dict[str, Any]]:
        """Sample from bonafide and attack groups according to target counts."""
        sampled_files: list[dict[str, Any]] = []
        attack_groups = [name for name in sample_counts if name != "real"]

        for group_name in attack_groups:
            available_files = files_by_group[group_name]
            target_count = sample_counts[group_name]

            if len(available_files) >= target_count:
                sampled_group = rng.sample(available_files, k=target_count)
            elif self.force_oversample_attacks:
                sampled_group = rng.choices(available_files, k=target_count)
                self.logger.debug(
                    f"[{split}] Oversampling {group_name}: "
                    f"{len(available_files)} -> {target_count}"
                )
            else:
                sampled_group = available_files
                self.logger.warning(
                    f"[{split}] Insufficient samples for {group_name}: "
                    f"need {target_count}, have {len(available_files)}"
                )

            sampled_files.extend(sampled_group)

        bonafide_files = files_by_group["real"]
        target_bonafide = sample_counts["real"]
        if target_bonafide < len(bonafide_files):
            sampled_bonafide = rng.sample(bonafide_files, k=target_bonafide)
        else:
            sampled_bonafide = bonafide_files

        sampled_files.extend(sampled_bonafide)
        rng.shuffle(sampled_files)

        samples_per_attack = sample_counts[attack_groups[0]] if attack_groups else 0
        num_attacks = len(sampled_files) - len(sampled_bonafide)
        self.logger.info(
            f"[{split}] Attack-balanced sampling (epoch {self.current_epoch}): "
            f"{num_attacks} attacks ({samples_per_attack} per group x "
            f"{len(attack_groups)} groups) + {len(sampled_bonafide)} real = "
            f"{len(sampled_files)} total "
            f"({len(sampled_files) / original_count * 100:.1f}% of original)"
        )
        return sampled_files

    def _is_bonafide_group(self, value: Any) -> bool:
        """Return whether the group value should be treated as bonafide."""
        return self._normalize_group_value(value) in self.bonafide_values

    @staticmethod
    def _normalize_group_value(value: Any) -> str:
        """Normalize group labels before comparing them."""
        return str(value).strip().lower()
