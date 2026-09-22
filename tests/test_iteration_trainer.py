"""Tests for the fixed-iteration training lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    trainer.save_checkpoint = (
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
    trainer.save_checkpoint = lambda iteration, filename=None: None

    trainer.perform_training()

    assert observed_modes == [True]


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
