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
from dl_core.core.base_trainer import EpochTrainer
from dl_core.utils.artifact_manager import ArtifactManager


class _ConcreteTrainer(EpochTrainer):
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

    def set_sampler_epoch(self, epoch: int) -> None:
        """Accept restored epoch state."""

        del epoch

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return an unwrapped local model."""

        return model

    def load_accelerator_state(self, checkpoint: dict[str, Any]) -> None:
        """Accept an empty accelerator state."""

    def get_accelerator_state(self) -> dict[str, Any]:
        """Return no additional accelerator checkpoint state."""

        return {}


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
        self.training_start_calls = 0
        self.training_end_calls: list[tuple[dict[str, Any], bool]] = []
        self.finalized_calls: list[dict[str, Any]] = []

    def on_training_start(self) -> None:
        """Record overfit-mode callback startup."""
        self.training_start_calls += 1

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


def test_checkpoint_callback_supports_iteration_windows() -> None:
    """Checkpoint selection should run at iteration reporting boundaries."""

    trainer = _CheckpointTrainerStub()
    callback = CheckpointCallback(
        monitor="validation_accuracy",
        mode="max",
        save_best_only=True,
    )
    callback.set_trainer(trainer)

    callback.on_iteration_end(25, {"validation/accuracy": 0.8})

    assert trainer.saved_epochs == [(25, None), (25, "best.pth")]


def test_checkpoint_callback_restores_best_state() -> None:
    """Best checkpoint selection should continue across resumed runs."""

    callback = CheckpointCallback(monitor="loss", mode="min")
    callback.best_value = 0.25
    callback.best_epoch = 4

    restored = CheckpointCallback(monitor="loss", mode="min")
    restored.set_state(callback.get_state())

    assert restored.best_value == 0.25
    assert restored.best_epoch == 4


def test_checkpoint_callback_ignores_non_finite_metrics() -> None:
    """A NaN metric must not become the permanent best value."""

    trainer = _CheckpointTrainerStub()
    callback = CheckpointCallback(
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callback.set_trainer(trainer)

    callback.on_epoch_end(1, {"loss": float("nan")})
    callback.on_epoch_end(2, {"loss": 0.5})

    assert callback.best_value == 0.5
    assert callback.best_epoch == 2
    assert trainer.saved_epochs == [(2, None), (2, "best.pth")]


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


def test_early_stopping_counts_non_finite_metrics_toward_patience() -> None:
    """A persistently non-finite metric should stop instead of running forever."""

    trainer = _CheckpointTrainerStub()
    callback = EarlyStoppingCallback(monitor="loss", mode="min", patience=2)
    callback.set_trainer(trainer)

    callback.on_epoch_end(1, {"loss": float("nan")})
    assert trainer.stop_training is False
    callback.on_epoch_end(2, {"loss": float("inf")})

    assert callback.metric_states["loss"]["best_value"] is None
    assert callback.metric_states["loss"]["wait"] == 2
    assert trainer.stop_training is True


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


def test_checkpoint_save_is_atomic_on_write_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed save must preserve the previous checkpoint alias."""

    trainer = _ConcreteTrainer()
    trainer.accelerator = _MainProcessAcceleratorStub()
    trainer.artifact_manager = ArtifactManager(
        run_name="atomic-save",
        output_dir=str(tmp_path),
    )
    trainer._get_current_checkpoint = lambda epoch: {"epoch": epoch}
    checkpoint_path = trainer.artifact_manager.get_final_checkpoint_path(
        "latest.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1}, checkpoint_path)
    stale_path = checkpoint_path.parent / ".latest.pth.stale.tmp"
    stale_path.write_bytes(b"partial")

    def _fail_save(payload: dict[str, int], destination: Any) -> None:
        destination.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("dl_core.core.base_trainer.torch.save", _fail_save)

    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint(2, filename="latest.pth")

    assert torch.load(checkpoint_path, weights_only=False) == {"epoch": 1}
    assert not stale_path.exists()
    assert not list(checkpoint_path.parent.glob(".latest.pth.*.tmp"))


def test_checkpoint_load_restores_criterion_and_progress_state(tmp_path: Path) -> None:
    """Resume should restore criterion buffers and trainer selection counters."""

    class _StatefulCriterion(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.register_buffer("running_value", torch.tensor(value))

    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_checkpoint_restore")
    trainer.accelerator = _MainProcessAcceleratorStub()
    trainer.models = {"main": torch.nn.Linear(1, 1)}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.criterions = {"stateful": _StatefulCriterion(0.0)}
    trainer.ema = None
    trainer.callbacks = CallbackList([])
    trainer.dataset_wrapper = type(
        "DatasetStub",
        (),
        {"set_epoch": lambda self, epoch: None},
    )()
    trainer.metric_managers = {}
    trainer.best_metric = None
    trainer.epochs_no_improvement = 0
    trainer.global_step = 0
    trainer.current_epoch = 0
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(
        {
            "models_state_dict": {
                "main": trainer.models["main"].state_dict(),
            },
            "criterion_stateful_state_dict": _StatefulCriterion(3.5).state_dict(),
            "epoch": 4,
            "global_step": 19,
            "best_metric": 0.2,
            "epochs_no_improvement": 3,
        },
        checkpoint_path,
    )

    trainer.load_checkpoint(str(checkpoint_path))

    assert trainer.criterions["stateful"].running_value.item() == pytest.approx(3.5)
    assert trainer.current_epoch == 4
    assert trainer.global_step == 19
    assert trainer.best_metric == pytest.approx(0.2)
    assert trainer.epochs_no_improvement == 3


def test_checkpoint_payload_round_trips_callback_state(tmp_path: Path) -> None:
    """Callback state should survive the trainer's real save/load payload path."""

    source = _ConcreteTrainer()
    source.logger = logging.getLogger("test_callback_payload_source")
    source.accelerator = _MainProcessAcceleratorStub()
    source.models = {"main": torch.nn.Linear(1, 1)}
    source.optimizers = {}
    source.schedulers = {}
    source.criterions = {}
    source.ema = None
    source.config = {}
    source.metrics_history = {
        "train": {},
        "validation": {},
        "test": {},
        "general": {},
    }
    source.best_metric = None
    source.epochs_no_improvement = 0
    source.global_step = 5
    source_callback = CheckpointCallback(monitor="loss", mode="min")
    source_callback.best_value = 0.25
    source_callback.best_epoch = 3
    source.callbacks = CallbackList([source_callback])
    source.callbacks.set_trainer(source)
    checkpoint_path = tmp_path / "callback-state.pth"
    torch.save(source._build_checkpoint_payload(3), checkpoint_path)

    target = _ConcreteTrainer()
    target.logger = logging.getLogger("test_callback_payload_target")
    target.accelerator = _MainProcessAcceleratorStub()
    target.models = {"main": torch.nn.Linear(1, 1)}
    target.optimizers = {}
    target.schedulers = {}
    target.criterions = {}
    target.ema = None
    target.dataset_wrapper = type(
        "DatasetStub",
        (),
        {"set_epoch": lambda self, epoch: None},
    )()
    target.metric_managers = {}
    target.best_metric = None
    target.epochs_no_improvement = 0
    target.global_step = 0
    target.current_epoch = 0
    target_callback = CheckpointCallback(monitor="loss", mode="min")
    target.callbacks = CallbackList([target_callback])
    target.callbacks.set_trainer(target)

    target.load_checkpoint(str(checkpoint_path))

    assert target_callback.best_value == pytest.approx(0.25)
    assert target_callback.best_epoch == 3


@pytest.mark.parametrize("layout", ["current", "legacy", "old_single_run"])
def test_auto_resume_falls_back_through_real_epoch_layout(
    tmp_path: Path,
    layout: str,
) -> None:
    """Trainer auto-resume should recover from each local run layout."""

    if layout == "legacy":
        run_dir = tmp_path / "demo-exp" / "demo"
    elif layout == "old_single_run":
        run_dir = tmp_path / "sweeps" / "training" / "demo"
    else:
        run_dir = tmp_path / "runs" / "demo"
    checkpoint_dir = run_dir / "final" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.pth").write_bytes(b"truncated")

    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_auto_resume_fallback")
    trainer.accelerator = _MainProcessAcceleratorStub()
    trainer.models = {"main": torch.nn.Linear(1, 1)}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.criterions = {}
    trainer.ema = None
    trainer.callbacks = CallbackList([])
    trainer.dataset_wrapper = type(
        "DatasetStub",
        (),
        {"set_epoch": lambda self, epoch: None},
    )()
    trainer.metric_managers = {}
    trainer.best_metric = None
    trainer.epochs_no_improvement = 0
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.continue_model = None
    trainer.trainer_config = {}
    trainer.config = {
        "auto_resume_local": True,
        "_config_path": str(tmp_path / "training.yaml"),
    }
    trainer.artifact_manager = ArtifactManager(
        run_name="demo",
        output_dir=str(tmp_path),
        experiment_name="demo-exp",
    )
    trainer.checkpoint_dir = str(
        trainer.artifact_manager.get_checkpoints_dir()
    )
    epoch_checkpoint = run_dir / "epoch_6" / "checkpoint.pth"
    epoch_checkpoint.parent.mkdir()
    torch.save(
        {
            "models_state_dict": {
                "main": trainer.models["main"].state_dict(),
            },
            "epoch": 6,
            "global_step": 17,
        },
        epoch_checkpoint,
    )
    trainer._load_auto_resume_model()

    assert trainer.continue_model == str(epoch_checkpoint)
    assert trainer.current_epoch == 6
    assert trainer.global_step == 17


def test_checkpoint_load_broadcasts_main_rank_failure(monkeypatch: Any) -> None:
    """Every rank should fail when rank zero cannot deserialize a checkpoint."""

    class _WorkerAcceleratorStub(_MainProcessAcceleratorStub):
        def is_main_process(self) -> bool:
            return False

    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_distributed_checkpoint_failure")
    trainer.accelerator = _WorkerAcceleratorStub()
    trainer.models = {"main": torch.nn.Linear(1, 1)}
    broadcast_values: list[list[Any]] = []

    monkeypatch.setattr("dl_core.core.base_trainer.dist.is_initialized", lambda: True)

    def _broadcast(values: list[Any], src: int) -> None:
        assert src == 0
        broadcast_values.append(list(values))
        values[:] = [None, "UnpicklingError: invalid load key"]

    monkeypatch.setattr(
        "dl_core.core.base_trainer.dist.broadcast_object_list",
        _broadcast,
    )

    with pytest.raises(RuntimeError, match="UnpicklingError: invalid load key"):
        trainer.load_checkpoint("corrupt.pth")

    assert broadcast_values == [[None, None]]


def test_checkpoint_load_broadcasts_rank_zero_read_error(
    monkeypatch: Any,
) -> None:
    """Rank zero must send its deserialization failure before raising."""

    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_rank_zero_checkpoint_failure")
    trainer.accelerator = _MainProcessAcceleratorStub()
    trainer.models = {"main": torch.nn.Linear(1, 1)}
    broadcast_values: list[list[Any]] = []

    def _fail_load(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("unreadable checkpoint")

    def _broadcast(values: list[Any], src: int) -> None:
        assert src == 0
        broadcast_values.append(list(values))

    monkeypatch.setattr("dl_core.core.base_trainer.torch.load", _fail_load)
    monkeypatch.setattr("dl_core.core.base_trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "dl_core.core.base_trainer.dist.broadcast_object_list", _broadcast
    )

    with pytest.raises(RuntimeError, match="OSError: unreadable checkpoint"):
        trainer.load_checkpoint("corrupt.pth")

    assert broadcast_values == [[None, "OSError: unreadable checkpoint"]]


def test_auto_resume_broadcasts_rank_zero_discovery_error(
    monkeypatch: Any,
) -> None:
    """A discovery failure must not strand non-main ranks in a collective."""

    trainer = _ConcreteTrainer()
    trainer.logger = logging.getLogger("test_rank_zero_discovery_failure")
    trainer.accelerator = _MainProcessAcceleratorStub()
    trainer.config = {"auto_resume_local": True}
    trainer.continue_model = None
    trainer.checkpoint_dir = "unreadable"
    broadcast_values: list[list[Any]] = []

    def _fail_discovery(checkpoint_dir: str) -> list[str]:
        del checkpoint_dir
        raise OSError("permission denied")

    def _broadcast(values: list[Any], src: int) -> None:
        assert src == 0
        broadcast_values.append(list(values))

    monkeypatch.setattr(
        "dl_core.core.base_trainer.find_checkpoint_candidates_local",
        _fail_discovery,
    )
    monkeypatch.setattr("dl_core.core.base_trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "dl_core.core.base_trainer.dist.broadcast_object_list", _broadcast
    )

    with pytest.raises(RuntimeError, match="OSError: permission denied"):
        trainer._load_auto_resume_model()

    assert broadcast_values == [[[], "OSError: permission denied"]]


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


def test_select_checkpoint_prefers_best_then_latest(tmp_path: Path) -> None:
    """Default post-training checkpoint selection should use existing aliases."""

    trainer = _ConcreteTrainer()
    trainer.artifact_manager = ArtifactManager(
        run_name="demo-run",
        output_dir=str(tmp_path),
        experiment_name="demo-exp",
    )
    best_path = trainer.artifact_manager.get_final_checkpoint_path("best.pth")
    latest_path = trainer.artifact_manager.get_final_checkpoint_path("latest.pth")

    assert trainer.select_checkpoint() is None

    latest_path.write_text("latest", encoding="utf-8")
    assert trainer.select_checkpoint() == latest_path

    best_path.write_text("best", encoding="utf-8")
    assert trainer.select_checkpoint() == best_path


def test_run_calls_post_training_before_persisting_analysis(tmp_path: Path) -> None:
    """Completed runs should expose the selected checkpoint to post-training hooks."""

    trainer, _, callbacks, _, finalize_sync = _build_lifecycle_trainer(tmp_path)
    best_path = trainer.artifact_manager.get_final_checkpoint_path("best.pth")
    best_path.write_text("best", encoding="utf-8")
    events: list[tuple[str, str | None]] = []

    trainer.setup = lambda: None
    trainer.perform_training = lambda: events.append(("training", None))

    def _post_training(checkpoint_path: Path | None) -> None:
        events.append(
            (
                "post_training",
                str(checkpoint_path) if checkpoint_path is not None else None,
            )
        )

    def _persist_run_analysis(status: str, error_message: str | None) -> None:
        events.append(("persist", status))

    trainer.post_training = _post_training
    trainer.persist_run_analysis = _persist_run_analysis

    trainer._run()

    assert events == [
        ("training", None),
        ("post_training", str(best_path)),
        ("persist", "completed"),
    ]
    assert trainer.selected_checkpoint_path == best_path
    assert callbacks.training_end_calls[0][0]["selected_checkpoint_path"] == str(
        best_path
    )
    assert finalize_sync == [True]


def test_overfit_mode_emits_one_final_status(tmp_path: Path) -> None:
    """Overfit callbacks stay open until post-training has completed."""
    trainer, _, callbacks, _, _ = _build_lifecycle_trainer(tmp_path)
    trainer.overfit_single_batch_enabled = True
    trainer.overfit_iterations = 3
    trainer.setup = lambda: None
    trainer._overfit_single_batch = lambda: {"loss": 0.1}
    trainer.post_training = lambda checkpoint_path: None

    trainer._run()

    assert callbacks.training_start_calls == 1
    assert len(callbacks.training_end_calls) == 1
    final_logs = callbacks.training_end_calls[0][0]
    assert final_logs["status"] == "completed"
    assert final_logs["overfit_iterations"] == 3
    assert final_logs["test_type"] == "single_batch_overfit"
    assert callbacks.finalized_calls == [final_logs]


def test_checkpoint_cleanup_preserves_failure_artifacts() -> None:
    """Resume failures should retain config and logs while removing temp files."""

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

        config_path = active.run_dir / "config.yaml"
        config_path.write_text("trainer: demo\n", encoding="utf-8")
        temporary_path = active.get_checkpoints_dir() / ".latest.pth.failed.tmp"
        temporary_path.write_bytes(b"partial")
        trainer._checkpoint_dir_cleanup()

        assert active.run_dir.exists()
        assert config_path.exists()
        assert not temporary_path.exists()
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


def test_run_fails_when_requested_continue_checkpoint_cannot_load(
    tmp_path: Path,
) -> None:
    """An explicit resume failure must not silently start fresh training."""

    trainer, _, callbacks, persisted, finalize_sync = _build_lifecycle_trainer(
        tmp_path
    )
    training_calls: list[bool] = []
    corrupt_checkpoint = tmp_path / "corrupt.pth"
    corrupt_checkpoint.write_bytes(b"truncated")
    config_path = trainer.artifact_manager.run_dir / "config.yaml"
    config_path.write_text("trainer: demo\n", encoding="utf-8")
    trainer.setup = lambda: None
    trainer.models = {"main": torch.nn.Linear(1, 1)}
    trainer.continue_model = str(corrupt_checkpoint)
    trainer.load_continue_model = lambda: trainer._load_continue_model()
    trainer.perform_training = lambda: training_calls.append(True)

    with pytest.raises(RuntimeError, match="Could not load checkpoint"):
        trainer._run()

    assert training_calls == []
    assert persisted[0][0] == "failed"
    assert "Could not load checkpoint" in str(persisted[0][1])
    assert callbacks.training_end_calls[0][0]["status"] == "failed"
    assert finalize_sync == [False]
    assert config_path.exists()


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
