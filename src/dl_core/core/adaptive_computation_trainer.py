"""Adaptive computation trainer foundation."""

from __future__ import annotations

from typing import Any

from .base_trainer import EpochTrainer


class AdaptiveComputationTrainer(EpochTrainer):
    """
    Epoch-based trainer foundation for adaptive-time computation models.

    This keeps the same artifact, callback, accelerator, and multi-GPU flow as
    `EpochTrainer` while giving recursive or halting-based trainers a dedicated
    semantic base class for their extra control logic.
    """

    @property
    def adaptive_computation_config(self) -> dict[str, Any]:
        """Return adaptive-computation-specific trainer configuration."""

        return dict(self.trainer_config.get("adaptive_computation", {}))
