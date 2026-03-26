"""Tests for the specialized trainer base hierarchy."""

from __future__ import annotations

from typing import Any

from dl_core.core import (
    AdaptiveComputationTrainer,
    BaseTrainer,
    EpochTrainer,
    SequenceTrainer,
)
from dl_core.trainers import StandardTrainer


class _ConcreteSequenceTrainer(SequenceTrainer):
    """Concrete sequence trainer used for hierarchy tests."""

    def __init__(self) -> None:
        pass

    def setup_model(self) -> None:
        """No-op test implementation."""

    def setup_criterion(self) -> None:
        """No-op test implementation."""

    def setup_optimizer(self) -> None:
        """No-op test implementation."""

    def setup_scheduler(self) -> None:
        """No-op test implementation."""

    def train_step(self, batch_data: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """No-op test implementation."""

        return {}

    def test_step(self, batch_data: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """No-op test implementation."""

        return {}

    def validation_step(
        self,
        batch_data: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, Any]:
        """No-op test implementation."""

        return {}


class _ConcreteAdaptiveTrainer(AdaptiveComputationTrainer):
    """Concrete adaptive trainer used for hierarchy tests."""

    def __init__(self) -> None:
        pass

    def setup_model(self) -> None:
        """No-op test implementation."""

    def setup_criterion(self) -> None:
        """No-op test implementation."""

    def setup_optimizer(self) -> None:
        """No-op test implementation."""

    def setup_scheduler(self) -> None:
        """No-op test implementation."""

    def train_step(self, batch_data: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """No-op test implementation."""

        return {}

    def test_step(self, batch_data: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """No-op test implementation."""

        return {}

    def validation_step(
        self,
        batch_data: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, Any]:
        """No-op test implementation."""

        return {}


def test_base_trainer_is_epoch_trainer_alias() -> None:
    """`BaseTrainer` should remain a compatibility alias."""

    assert BaseTrainer is EpochTrainer


def test_standard_trainer_uses_epoch_trainer() -> None:
    """The built-in standard trainer should extend the epoch-based base."""

    assert issubclass(StandardTrainer, EpochTrainer)


def test_sequence_trainer_exposes_sequence_config() -> None:
    """Sequence trainers should expose their specialized config section."""

    trainer = _ConcreteSequenceTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "sequence": {"max_length": 128, "teacher_forcing": True},
    }

    assert trainer.sequence_config == {
        "max_length": 128,
        "teacher_forcing": True,
    }


def test_adaptive_trainer_exposes_adaptive_config() -> None:
    """Adaptive trainers should expose their specialized config section."""

    trainer = _ConcreteAdaptiveTrainer()
    trainer.trainer_config = {
        "epochs": 2,
        "adaptive_computation": {"max_halt_steps": 8, "ponder_cost": 0.01},
    }

    assert trainer.adaptive_computation_config == {
        "max_halt_steps": 8,
        "ponder_cost": 0.01,
    }
