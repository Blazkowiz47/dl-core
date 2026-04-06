"""Tests for checkpoint callback and trainer checkpoint policy."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import logging
import torch
from dl_core.callbacks.checkpoint import CheckpointCallback
from dl_core.callbacks.early_stopping import EarlyStoppingCallback
from dl_core.core.base_callback import Callback, CallbackList
from dl_core.core.base_trainer import BaseTrainer
from dl_core.utils.artifact_manager import ArtifactManager


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


class _MainProcessAcceleratorStub:
    """Simple accelerator test double for trainer helper tests."""

    def is_main_process(self) -> bool:
        """Report main-process ownership."""

        return True

    def get_device(self) -> torch.device:
        """Return a CPU device for distributed sync tests."""

        return torch.device("cpu")

    def wait_for_everyone(self, context: str | None = None) -> None:
        """No-op barrier stub."""


class _TrainingStartCallback(Callback):
    """Callback that records how many times training-start fired."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_training_start(self, logs: dict[str, Any] | None = None) -> None:
        """Count the number of times the callback ran."""

        self.calls += 1


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


def test_checkpoint_dir_cleanup_only_removes_current_run() -> None:
    """Empty-checkpoint cleanup should only remove the active run directory."""

    with TemporaryDirectory() as temp_dir:
        active = ArtifactManager(
            run_name="run-a",
            output_dir=temp_dir,
            experiment_name="demo-exp",
        )
        sibling = ArtifactManager(
            run_name="run-b",
            output_dir=temp_dir,
            experiment_name="demo-exp",
        )
        trainer = _ConcreteTrainer()
        trainer.accelerator = _MainProcessAcceleratorStub()
        trainer.artifact_manager = active
        trainer.checkpoint_dir = str(active.get_checkpoints_dir())

        Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        trainer._checkpoint_dir_cleanup()

        assert not active.run_dir.exists()
        assert sibling.run_dir.exists()
        assert Path(temp_dir).exists()


def test_inject_seed_updates_named_trainer_config() -> None:
    """Seed injection should update the named trainer section only."""

    trainer = _ConcreteTrainer()
    trainer.seed = 123
    trainer.trainer_name = "standard"
    trainer.config = {
        "trainer": {"standard": {"epochs": 2}},
        "dataset": {"name": "dummy"},
        "models": {"main": {}},
        "accelerator": {},
    }

    trainer._inject_seed_into_configs()

    assert trainer.config["trainer"]["standard"]["seed"] == 123
    assert "seed" not in trainer.config["trainer"]
    assert trainer.config["dataset"]["seed"] == 123
    assert trainer.config["models"]["main"]["seed"] == 123
    assert trainer.config["accelerator"]["seed"] == 123


def test_run_raises_setup_error_instead_of_exiting() -> None:
    """Trainer setup failures should propagate instead of hard-exiting."""

    class _FailingTrainer(_ConcreteTrainer):
        def setup(self) -> None:
            raise RuntimeError("boom")

    trainer = _FailingTrainer()
    trainer.logger = logging.getLogger("test_setup_failure")

    try:
        trainer._run()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected RuntimeError from setup failure")


def test_callback_list_syncs_enabled_state_across_ranks(monkeypatch: Any) -> None:
    """Distributed callback sync should disable hooks when any rank disabled them."""

    callback = _TrainingStartCallback()
    trainer = _CheckpointTrainerStub()
    trainer.accelerator = _MainProcessAcceleratorStub()
    callback_list = CallbackList([callback])
    callback_list.set_trainer(trainer)

    monkeypatch.setattr("dl_core.core.base_callback.dist.is_available", lambda: True)
    monkeypatch.setattr("dl_core.core.base_callback.dist.is_initialized", lambda: True)

    def _fake_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        tensor.fill_(1)

    monkeypatch.setattr("dl_core.core.base_callback.dist.all_reduce", _fake_all_reduce)

    callback_list.on_training_start()

    assert callback.enabled is False
    assert callback.calls == 0
