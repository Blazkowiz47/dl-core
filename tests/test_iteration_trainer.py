"""Tests for the fixed-iteration training lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from dl_core.core import IterationTrainer
from dl_core.utils import ArtifactManager


class _ConcreteIterationTrainer(IterationTrainer):
    """Minimal concrete implementation for lifecycle tests."""

    def setup_artifact_manager(self) -> None:
        """Avoid filesystem setup in unit tests."""

    def setup_model(self) -> None:
        """No-op test implementation."""

    def setup_criterion(self) -> None:
        """No-op test implementation."""

    def setup_optimizer(self) -> None:
        """No-op test implementation."""

    def setup_scheduler(self) -> None:
        """No-op test implementation."""

    def train_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        """Return the global batch index as a scalar metric."""

        return {"loss": float(batch_idx)}

    def test_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """No-op test implementation."""

        return {}

    def validation_step(
        self,
        batch_data: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """No-op test implementation."""

        return {}


class _AcceleratorStub:
    """Single-process accelerator with sampler-cycle recording."""

    def __init__(self) -> None:
        self.sampler_epochs: list[int] = []

    def is_main_process(self) -> bool:
        """Report main-process ownership."""

        return True

    def wait_for_everyone(self, context: str | None = None) -> None:
        """No-op synchronization point."""

    def set_sampler_epoch(self, epoch: int) -> None:
        """Record the deterministic loader cycle."""

        self.sampler_epochs.append(epoch)

    def to_device(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Keep test batches on CPU."""

        return batch_data


class _DatasetStub:
    """Dataset wrapper with cycle recording."""

    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        """Record deterministic cycle propagation."""

        self.epochs.append(epoch)


class _CallbacksStub:
    """Record the iteration callback sequence used by the trainer."""

    def __init__(self) -> None:
        self.batch_indices: list[int] = []
        self.iteration_ends: list[int] = []
        self.checkpoints: list[int] = []

    def on_training_start(self) -> None:
        """No-op lifecycle hook."""

    def on_train_start(self, iteration: int, logs: dict[str, Any]) -> None:
        """No-op reporting-window hook."""

    def on_train_end(self, iteration: int, logs: dict[str, Any]) -> None:
        """No-op reporting-window hook."""

    def on_batch_start(
        self,
        batch_idx: int,
        split: str,
        batch_data: dict[str, Any],
    ) -> None:
        """Record the global iteration passed to batch callbacks."""

        self.batch_indices.append(batch_idx)

    def on_batch_end(
        self,
        batch_idx: int,
        split: str,
        batch_data: dict[str, Any],
    ) -> None:
        """No-op batch hook."""

    def on_iteration_end(self, iteration: int, logs: dict[str, Any]) -> None:
        """Record reporting-window completion."""

        self.iteration_ends.append(iteration)

    def on_validation_start(self, iteration: int) -> None:
        """No-op evaluation hook."""

    def on_validation_end(
        self,
        iteration: int,
        logs: dict[str, Any],
    ) -> None:
        """No-op evaluation hook."""

    def on_test_start(self, iteration: int) -> None:
        """No-op evaluation hook."""

    def on_test_end(self, iteration: int, logs: dict[str, Any]) -> None:
        """No-op evaluation hook."""

    def on_checkpoint(
        self,
        iteration: int,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """Record trainer-managed latest checkpoints."""

        self.checkpoints.append(iteration)


class _CheckpointRequestingCallbacksStub(_CallbacksStub):
    """Request a callback-managed alias during one accumulation window."""

    def __init__(
        self,
        trainer: _ConcreteIterationTrainer,
        request_iteration: int,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.request_iteration = request_iteration

    def on_batch_end(
        self,
        batch_idx: int,
        split: str,
        batch_data: dict[str, Any],
    ) -> None:
        """Request a best alias while the accumulation window is pending."""

        super().on_batch_end(batch_idx, split, batch_data)
        iteration = batch_idx + 1
        if iteration == self.request_iteration:
            self.trainer.save_checkpoint(iteration, filename="best.pth")


class _EarlyStoppingCallbacksStub(_CallbacksStub):
    """Stop at one reporting window to exercise safe deferred termination."""

    def __init__(
        self,
        trainer: _ConcreteIterationTrainer,
        stop_iteration: int,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.stop_iteration = stop_iteration

    def on_iteration_end(self, iteration: int, logs: dict[str, Any]) -> None:
        """Set the trainer stop flag at the configured reporting window."""

        super().on_iteration_end(iteration, logs)
        if iteration == self.stop_iteration:
            self.trainer.stop_training = True


def _build_trainer(iterations: int = 5) -> _ConcreteIterationTrainer:
    """Build a configured iteration trainer without running full setup."""

    trainer = _ConcreteIterationTrainer(
        {
            "trainer": {
                "iteration": {
                    "iterations": iterations,
                    "log_frequency": 2,
                    "checkpoint_frequency": 0,
                    "validation_frequency": 0,
                    "test_frequency": 0,
                    "skip_baseline_eval": True,
                }
            }
        }
    )
    trainer.accelerator = _AcceleratorStub()
    trainer.dataset_wrapper = _DatasetStub()
    trainer.callbacks = _CallbacksStub()
    trainer.data_loader = {
        "train": [
            {"image": torch.ones(1, 1)},
            {"image": torch.ones(1, 1)},
        ],
        "validation": None,
        "test": None,
    }
    return trainer


def test_iteration_config_does_not_require_epochs() -> None:
    """IterationTrainer should use an iteration limit without an epoch shim."""

    trainer = _build_trainer(7)

    assert trainer.iterations == 7
    assert trainer.epochs == 0
    assert "epochs" not in {field["name"] for field in trainer.CONFIG_FIELDS}


def test_iteration_training_cycles_finite_loader_and_reports_final_window() -> None:
    """Finite loaders should cycle until the exact iteration count is reached."""

    trainer = _build_trainer(5)
    saved: list[tuple[int, str | None]] = []
    trainer._save_checkpoint = (
        lambda iteration, filename=None: saved.append((iteration, filename))
    )

    trainer.perform_training()

    assert trainer.current_iteration == 5
    assert trainer.global_step == 5
    assert trainer.data_cycle == 2
    assert trainer.iteration_in_cycle == 1
    assert trainer.accelerator.sampler_epochs == [0, 1, 2]
    assert trainer.dataset_wrapper.epochs == [0, 1, 2]
    assert trainer.callbacks.batch_indices == [0, 1, 2, 3, 4]
    assert trainer.callbacks.iteration_ends == [2, 4, 5]
    assert trainer.callbacks.checkpoints == [5]
    assert saved == [(5, "latest.pth")]
    assert sorted(trainer.train_metrics) == [2, 4, 5]


def test_iteration_training_restores_train_mode_after_baseline() -> None:
    """The first optimization batch must not inherit baseline evaluation mode."""

    trainer = _build_trainer(1)
    trainer.skip_baseline_eval = False
    trainer.models["main"] = torch.nn.Linear(1, 1)
    observed_modes: list[bool] = []
    trainer.perform_baseline_evaluation = lambda: trainer.set_models_mode("eval")
    trainer.train_step = lambda batch_data, batch_idx: (
        observed_modes.append(trainer.model.training) or {"loss": 0.0}
    )
    trainer._save_checkpoint = lambda iteration, filename=None: None

    trainer.perform_training()

    assert observed_modes == [True]


def test_iteration_checkpoint_waits_for_accumulation_boundary() -> None:
    """Periodic checkpoints should not capture unsynchronized partial gradients."""

    trainer = _build_trainer(8)
    trainer.checkpoint_frequency = 3
    trainer.log_frequency = 100
    trainer.accelerator.accumulation_counter = 0
    saved: list[tuple[int, str | None]] = []

    def _train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_data, batch_idx
        trainer.accelerator.accumulation_counter = (
            trainer.accelerator.accumulation_counter + 1
        ) % 4
        return {"loss": 0.0}

    trainer.train_step = _train_step
    trainer._save_checkpoint = (
        lambda iteration, filename=None: saved.append((iteration, filename))
    )

    trainer.perform_training()

    assert saved == [(4, "latest.pth"), (8, "latest.pth")]
    assert trainer.callbacks.checkpoints == [4, 8]
    assert trainer.callbacks.iteration_ends == [8]
    assert sorted(trainer.train_metrics) == [8]


def test_callback_checkpoint_waits_for_accumulation_boundary() -> None:
    """Callback aliases should be deferred until optimizer state is consistent."""

    trainer = _build_trainer(4)
    trainer.accelerator.accumulation_counter = 0
    trainer.callbacks = _CheckpointRequestingCallbacksStub(trainer, 2)
    saved: list[tuple[int, str | None]] = []

    def _train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_data, batch_idx
        trainer.accelerator.accumulation_counter = (
            trainer.accelerator.accumulation_counter + 1
        ) % 4
        return {"loss": 0.0}

    trainer.train_step = _train_step
    trainer._save_checkpoint = (
        lambda iteration, filename=None: saved.append((iteration, filename))
    )

    trainer.perform_training()

    assert saved == [(4, "best.pth"), (4, "latest.pth")]
    assert trainer.callbacks.checkpoints == [4]


def test_early_stop_finishes_pending_accumulation_before_checkpoint() -> None:
    """Early stopping should finish the active optimizer window before saving."""

    trainer = _build_trainer(8)
    trainer.log_frequency = 3
    trainer.accelerator.accumulation_counter = 0
    trainer.callbacks = _EarlyStoppingCallbacksStub(trainer, 4)
    saved: list[tuple[int, str | None]] = []

    def _train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_data, batch_idx
        trainer.accelerator.accumulation_counter = (
            trainer.accelerator.accumulation_counter + 1
        ) % 4
        return {"loss": 0.0}

    trainer.train_step = _train_step
    trainer._save_checkpoint = (
        lambda iteration, filename=None: saved.append((iteration, filename))
    )

    trainer.perform_training()

    assert trainer.current_iteration == 4
    assert trainer.callbacks.iteration_ends == [4]
    assert saved == [(4, "latest.pth")]


def test_final_partial_accumulation_is_flushed_before_checkpoint() -> None:
    """A trainer honoring the finalization flag should save the final state."""

    trainer = _build_trainer(6)
    trainer.log_frequency = 100
    trainer.accelerator.accumulation_counter = 0
    saved: list[tuple[int, str | None]] = []

    def _train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_data, batch_idx
        trainer.accelerator.accumulation_counter += 1
        if (
            trainer.accelerator.accumulation_counter == 4
            or trainer._finalize_accumulation
        ):
            trainer.accelerator.accumulation_counter = 0
        return {"loss": 0.0}

    trainer.train_step = _train_step
    trainer._save_checkpoint = (
        lambda iteration, filename=None: saved.append((iteration, filename))
    )

    trainer.perform_training()

    assert saved == [(6, "latest.pth")]
    assert trainer.callbacks.checkpoints == [6]


def test_final_pending_accumulation_fails_loudly() -> None:
    """Custom steps must not silently save a checkpoint with pending gradients."""

    trainer = _build_trainer(6)
    trainer.log_frequency = 100
    trainer.accelerator.accumulation_counter = 0

    def _train_step(
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, float]:
        del batch_data, batch_idx
        trainer.accelerator.accumulation_counter = (
            trainer.accelerator.accumulation_counter + 1
        ) % 4
        return {"loss": 0.0}

    trainer.train_step = _train_step

    with pytest.raises(RuntimeError, match="finalization flag"):
        trainer.perform_training()


def test_iteration_checkpoint_progress_restores_loader_cursor() -> None:
    """Resume state should retain iteration and finite-loader cycle position."""

    trainer = _build_trainer(10)

    trainer.restore_progress_state(
        {
            "global_step": 7,
            "iteration": 7,
            "data_cycle": 3,
            "iteration_in_cycle": 1,
        }
    )

    assert trainer.current_iteration == 7
    assert trainer.global_step == 7
    assert trainer.current_epoch == 7
    assert trainer.data_cycle == 3
    assert trainer.iteration_in_cycle == 1
    assert trainer.accelerator.sampler_epochs[-1] == 3
    assert trainer.dataset_wrapper.epochs[-1] == 3


def test_numbered_checkpoint_uses_iteration_directory(tmp_path: Path) -> None:
    """Numbered iteration checkpoints should not be labeled as epochs."""

    trainer = _build_trainer(5)
    trainer.artifact_manager = ArtifactManager(
        run_name="iteration-checkpoint",
        output_dir=str(tmp_path),
    )
    trainer._get_current_checkpoint = lambda iteration: {"iteration": iteration}

    trainer.save_checkpoint(3)

    checkpoint_path = (
        trainer.artifact_manager.get_iteration_checkpoint_path(3)
    )
    assert checkpoint_path.exists()
    assert torch.load(checkpoint_path, weights_only=False) == {"iteration": 3}
    assert trainer.artifact_manager.list_artifacts()["iterations"] == [
        "iteration_3"
    ]


def test_iteration_logs_use_iteration_key() -> None:
    """Flattened callback logs should expose iteration rather than epoch."""

    trainer = _build_trainer(5)
    trainer.current_iteration = 2
    trainer.current_epoch = 2
    trainer.metrics_history["train"][2] = {"loss": 0.5}

    assert trainer.compile_epoch_logs() == {
        "iteration": 2.0,
        "train/loss": 0.5,
    }
