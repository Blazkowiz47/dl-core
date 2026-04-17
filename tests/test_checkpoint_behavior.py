"""Tests for checkpoint callback and trainer checkpoint policy."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import logging
import pytest
import torch
from dl_core.callbacks.checkpoint import CheckpointCallback
from dl_core.callbacks.dataset_refresh import DatasetRefreshCallback
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


class _LifecycleAcceleratorStub(_MainProcessAcceleratorStub):
    """Accelerator stub that records teardown synchronization."""

    def __init__(self) -> None:
        self.wait_calls: list[str | None] = []

    def wait_for_everyone(self, context: str | None = None) -> None:
        """Record barrier requests instead of synchronizing."""

        self.wait_calls.append(context)


class _PrepareAcceleratorStub(_MainProcessAcceleratorStub):
    """Accelerator stub that records dataloader prepare calls."""

    def __init__(self) -> None:
        self.prepared_dataloaders: list[dict[str, Any]] = []

    def prepare(
        self,
        models: dict[str, Any] | None = None,
        optimizers: dict[str, Any] | None = None,
        criterions: dict[str, Any] | None = None,
        schedulers: dict[str, Any] | None = None,
        dataloaders: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Record and return prepared dataloaders only."""

        dataloaders = dataloaders or {}
        self.prepared_dataloaders.append(dataloaders.copy())
        prepared = {name: f"prepared:{value}" for name, value in dataloaders.items()}
        return {}, {}, {}, {}, prepared


class _LifecycleCallbacksStub:
    """Callback list stub used to observe trainer finalization behavior."""

    def __init__(self) -> None:
        self.training_end_calls: list[tuple[dict[str, Any], bool]] = []
        self.finalized_calls: list[dict[str, Any]] = []

    def on_training_end(
        self,
        logs: dict[str, Any] | None = None,
        synchronize: bool = True,
    ) -> None:
        """Record end-of-training callbacks."""

        self.training_end_calls.append((logs or {}, synchronize))

    def on_training_finalized(self, logs: dict[str, Any] | None = None) -> None:
        """Record post-cleanup callbacks."""

        self.finalized_calls.append(logs or {})


class _RefreshingDatasetStub:
    """Dataset stub that records refresh and split rebuild calls."""

    def __init__(self) -> None:
        self.refreshed_splits: list[str] = []
        self.requested_splits: list[str] = []

    def refresh_dataset(self, split: str | None = None) -> None:
        """Record refresh requests."""

        if split is not None:
            self.refreshed_splits.append(split)

    def get_split(self, split: str) -> str:
        """Return a synthetic loader token for the requested split."""

        self.requested_splits.append(split)
        return f"loader:{split}"


def _build_lifecycle_trainer(
    tmp_path: Path,
) -> tuple[
    _ConcreteTrainer,
    _LifecycleAcceleratorStub,
    _LifecycleCallbacksStub,
    list[tuple[str, str | None]],
    list[bool],
]:
    """Create a trainer stub configured for `_run` lifecycle tests."""

    trainer = _ConcreteTrainer()
    accelerator = _LifecycleAcceleratorStub()
    callbacks = _LifecycleCallbacksStub()
    persisted: list[tuple[str, str | None]] = []
    finalize_sync: list[bool] = []

    trainer.logger = logging.getLogger("test_lifecycle")
    trainer.accelerator = accelerator
    trainer.callbacks = callbacks
    trainer.artifact_manager = ArtifactManager(
        run_name="demo-run",
        output_dir=str(tmp_path),
        experiment_name="demo-exp",
    )
    trainer.checkpoint_dir = str(tmp_path / "missing-checkpoints")
    trainer.current_epoch = 0
    trainer.epochs = 3
    trainer.overfit_single_batch_enabled = False
    trainer.metric_managers = {}
    trainer.dataset_wrapper = type(
        "DatasetStub",
        (),
        {"set_epoch": lambda self, epoch: None},
    )()
    trainer.load_continue_model = lambda: None
    trainer.setup_current_epoch = lambda epoch: setattr(trainer, "current_epoch", epoch)
    trainer.persist_run_analysis = (
        lambda status, error_message: persisted.append((status, error_message))
    )
    trainer.finalize_training = (
        lambda synchronize=True: finalize_sync.append(synchronize)
    )

    return trainer, accelerator, callbacks, persisted, finalize_sync


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
    """Seed injection should update downstream configs used during setup."""

    trainer = _ConcreteTrainer()
    trainer.seed = 123
    trainer.deterministic = False
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
    assert trainer.config["dataset"]["deterministic"] is False
    assert trainer.config["models"]["main"]["seed"] == 123
    assert trainer.config["accelerator"]["seed"] == 123


def test_setup_passes_deterministic_to_seed_helper(monkeypatch: Any) -> None:
    """Trainer setup should forward the configured deterministic flag to seeding."""

    seed_calls: list[tuple[int, bool]] = []
    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_setup_deterministic")
    trainer.seed = 123
    trainer.deterministic = False
    trainer.trainer_name = "standard"
    trainer.config = {
        "trainer": {"standard": {"epochs": 1}},
        "dataset": {"name": "dummy"},
        "models": {},
        "accelerator": {},
        "metric_managers": {},
        "callbacks": {},
    }
    trainer.models = {}
    trainer.optimizers = {}
    trainer.criterions = {}
    trainer.schedulers = {}
    trainer.data_loader = {}
    trainer.metric_managers = {}
    trainer.accelerator = _PrepareAcceleratorStub()

    monkeypatch.setattr(
        "dl_core.core.base_trainer.set_seeds",
        lambda seed, deterministic=True: seed_calls.append((seed, deterministic)),
    )
    trainer.setup_accelerator = lambda: None
    trainer.setup_data = lambda: None
    trainer.setup_model = lambda: None
    trainer.setup_criterion = lambda: None
    trainer.setup_optimizer = lambda: None
    trainer.setup_scheduler = lambda: None
    trainer.setup_metrics = lambda: None
    trainer.setup_ema = lambda: None
    trainer.setup_callbacks = lambda: None

    trainer._setup()

    assert seed_calls == [(123, False)]


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


def test_dataset_refresh_callback_rebuilds_selected_split_loaders() -> None:
    """Dataset refresh should rebuild the requested split loaders."""

    trainer = _CheckpointTrainerStub()
    trainer.accelerator = _PrepareAcceleratorStub()
    trainer.dataset_wrapper = _RefreshingDatasetStub()
    trainer.data_loader = {
        "train": "stale-train",
        "validation": "stale-validation",
        "test": "stale-test",
    }

    callback = DatasetRefreshCallback(refresh_frequency=1, splits=["train", "test"])
    callback.set_trainer(trainer)
    callback.on_epoch_start(1)

    assert trainer.dataset_wrapper.refreshed_splits == ["train", "test"]
    assert trainer.dataset_wrapper.requested_splits == ["train", "test"]
    assert trainer.accelerator.prepared_dataloaders == [
        {"train": "loader:train", "test": "loader:test"}
    ]
    assert trainer.data_loader["train"] == "prepared:loader:train"
    assert trainer.data_loader["validation"] == "stale-validation"
    assert trainer.data_loader["test"] == "prepared:loader:test"


def test_dataset_refresh_callback_skips_non_matching_epochs() -> None:
    """Dataset refresh should respect the configured epoch frequency."""

    trainer = _CheckpointTrainerStub()
    trainer.accelerator = _PrepareAcceleratorStub()
    trainer.dataset_wrapper = _RefreshingDatasetStub()
    trainer.data_loader = {"train": "stale-train"}

    callback = DatasetRefreshCallback(refresh_frequency=2, splits=["train"])
    callback.set_trainer(trainer)
    callback.on_epoch_start(1)

    assert trainer.dataset_wrapper.refreshed_splits == []
    assert trainer.accelerator.prepared_dataloaders == []


def test_run_interrupt_skips_synchronized_teardown(tmp_path: Path) -> None:
    """Interrupted runs should finalize without teardown barriers."""

    trainer, accelerator, callbacks, persisted, finalize_sync = (
        _build_lifecycle_trainer(tmp_path)
    )
    trainer.setup = lambda: None

    def _interrupt() -> None:
        raise KeyboardInterrupt()

    trainer.perform_training = _interrupt

    with pytest.raises(KeyboardInterrupt):
        trainer._run()

    assert persisted == [("interrupted", "Training interrupted by user")]
    assert finalize_sync == [False]
    assert callbacks.training_end_calls[0][0]["status"] == "interrupted"
    assert callbacks.training_end_calls[0][1] is False
    assert callbacks.finalized_calls[0]["status"] == "interrupted"
    assert "before on_training_end callbacks" not in accelerator.wait_calls


def test_run_setup_failure_uses_best_effort_finalization(tmp_path: Path) -> None:
    """Setup failures should still finalize without synchronized teardown."""

    trainer, accelerator, callbacks, persisted, finalize_sync = (
        _build_lifecycle_trainer(tmp_path)
    )

    def _fail_setup() -> None:
        raise RuntimeError("boom")

    trainer.setup = _fail_setup
    trainer.perform_training = lambda: None

    with pytest.raises(RuntimeError, match="boom"):
        trainer._run()

    assert persisted == [("failed", "boom")]
    assert finalize_sync == [False]
    assert callbacks.training_end_calls[0][0]["status"] == "failed"
    assert callbacks.training_end_calls[0][1] is False
    assert callbacks.finalized_calls[0]["status"] == "failed"
    assert "before on_training_end callbacks" not in accelerator.wait_calls
