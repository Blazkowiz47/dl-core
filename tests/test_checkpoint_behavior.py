"""Tests for checkpoint callback and trainer checkpoint policy."""

from __future__ import annotations

from typing import Any

from dl_core.callbacks.checkpoint import CheckpointCallback
from dl_core.callbacks.early_stopping import EarlyStoppingCallback
from dl_core.core.base_trainer import BaseTrainer


class _ConcreteTrainer(BaseTrainer):
    """Small concrete trainer used to exercise helper logic."""

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


class _CheckpointTrainerStub:
    """Minimal trainer stub for callback-focused tests."""

    def __init__(self) -> None:
        self.accelerator = None
        self.saved_epochs: list[tuple[int, str | None]] = []
        self.stop_training = False

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> None:
        """Record checkpoint save requests."""

        self.saved_epochs.append((epoch, filename))


def test_checkpoint_callback_resolves_monitor_aliases() -> None:
    """Checkpoint callback should accept underscore monitor aliases."""

    trainer = _CheckpointTrainerStub()
    callback = CheckpointCallback(
        monitor="validation_accuracy",
        mode="max",
        save_best_only=True,
    )
    callback.set_trainer(trainer)

    callback.on_epoch_end(1, {"validation/accuracy": 0.75})

    assert trainer.saved_epochs == [(1, None), (1, "best.pth")]


def test_early_stopping_resolves_monitor_aliases() -> None:
    """Early stopping should accept underscore monitor aliases."""

    trainer = _CheckpointTrainerStub()
    callback = EarlyStoppingCallback(
        monitor="validation_accuracy",
        mode="max",
        patience=2,
    )
    callback.set_trainer(trainer)

    callback.on_epoch_end(1, {"validation/accuracy": 0.65})

    assert callback.metric_states["validation_accuracy"]["best_value"] == 0.65


def test_trainer_reuses_current_checkpoint_for_same_epoch() -> None:
    """Trainer should cache the current checkpoint payload within one epoch."""

    trainer = _ConcreteTrainer()
    trainer.callbacks = type("CallbackContainer", (), {"callbacks": []})()
    trainer.current_checkpoint = None
    trainer.current_checkpoint_epoch = None
    build_calls: list[int] = []

    def _build_checkpoint_payload(epoch: int) -> dict[str, int]:
        build_calls.append(epoch)
        return {"epoch": epoch}

    trainer._build_checkpoint_payload = _build_checkpoint_payload

    first = trainer._get_current_checkpoint(2)
    second = trainer._get_current_checkpoint(2)

    assert first == {"epoch": 2}
    assert second == {"epoch": 2}
    assert build_calls == [2]


def test_trainer_rebuilds_current_checkpoint_for_new_epoch() -> None:
    """Trainer should rebuild the cached checkpoint payload on epoch changes."""

    trainer = _ConcreteTrainer()
    trainer.callbacks = type("CallbackContainer", (), {"callbacks": []})()
    trainer.current_checkpoint = None
    trainer.current_checkpoint_epoch = None
    build_calls: list[int] = []

    def _build_checkpoint_payload(epoch: int) -> dict[str, int]:
        build_calls.append(epoch)
        return {"epoch": epoch}

    trainer._build_checkpoint_payload = _build_checkpoint_payload

    trainer._get_current_checkpoint(1)
    trainer._get_current_checkpoint(2)

    assert build_calls == [1, 2]


def test_select_best_epoch_resolves_monitor_aliases() -> None:
    """Best-epoch selection should honor slash and underscore aliases."""

    trainer = _ConcreteTrainer()
    trainer.metrics_history = {
        "train": {},
        "validation": {1: {"accuracy": 0.8}, 2: {"accuracy": 0.7}},
        "test": {},
        "general": {},
    }
    trainer.accelerator = type(
        "AcceleratorStub",
        (),
        {"is_main_process": lambda self: True},
    )()

    best_epoch, selection_value = trainer._select_best_epoch(
        "validation_accuracy",
        "max",
        [1, 2],
    )

    assert best_epoch == 1
    assert selection_value == 0.8
