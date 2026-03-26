"""Sequence-oriented trainer foundation."""

from __future__ import annotations

from typing import Any

from .base_trainer import EpochTrainer


class SequenceTrainer(EpochTrainer):
    """
    Epoch-based trainer foundation for sequence and NLP workloads.

    This subclass intentionally reuses the same accelerator, callback,
    checkpoint, and multi-GPU behavior as `EpochTrainer`. Sequence-specific
    trainers can extend it with token-level losses, decoding logic, or custom
    batching hooks without needing a separate distributed execution path.
    """

    @property
    def sequence_config(self) -> dict[str, Any]:
        """Return sequence-specific trainer configuration."""

        return dict(self.trainer_config.get("sequence", {}))
