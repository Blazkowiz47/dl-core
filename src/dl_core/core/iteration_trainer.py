"""Iteration-based trainer lifecycle for streaming and cyclic datasets."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist
from tqdm import tqdm

from dl_core.core.config_metadata import config_field
from dl_core.utils import MeterTracker

from .base_trainer import EpochTrainer


class IterationTrainer(EpochTrainer):
    """Train for a fixed number of batches instead of dataset epochs.

    One iteration consumes one batch on every distributed rank. Finite training
    loaders share a data cycle across ranks. Shorter ranks replay their current
    selection until every rank completes a pass; infinite loaders continue
    without a restart. Validation, testing, logging, and checkpoints run on
    iteration-based frequencies, with a final reporting window always emitted.
    """

    REQUIRES_EPOCHS = False
    PROGRESS_UNIT = "iteration"
    PROGRESS_UNITS = "iterations"
    CONFIG_FIELDS = [
        field
        for field in EpochTrainer.CONFIG_FIELDS
        if field["name"]
        not in {
            "epochs",
            "show_progress",
            "test_frequency",
            "validation_frequency",
        }
    ] + [
        config_field(
            "iterations",
            "int",
            "Total number of training batches to consume on each rank.",
            required=True,
        ),
        config_field(
            "show_progress",
            "bool",
            "Enable the iteration progress bar.",
            default=False,
        ),
        config_field(
            "log_frequency",
            "int",
            "Close and log a training-metric window every N iterations.",
            default=1000,
        ),
        config_field(
            "checkpoint_frequency",
            "int",
            "Save latest.pth every N iterations; zero means final only.",
            default=0,
        ),
        config_field(
            "validation_frequency",
            "int",
            "Run validation every N iterations; zero means final only.",
            default=0,
        ),
        config_field(
            "test_frequency",
            "int",
            "Run testing every N iterations; zero means final only.",
            default=0,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        """Initialize iteration limits, reporting frequencies, and resume state."""

        super().__init__(config)
        self.iterations = int(self.trainer_config["iterations"])
        self.log_frequency = int(self.trainer_config.get("log_frequency", 1000))
        self.checkpoint_frequency = int(
            self.trainer_config.get("checkpoint_frequency", 0)
        )
        self.validation_frequency = int(
            self.trainer_config.get("validation_frequency", 0)
        )
        self.test_frequency = int(self.trainer_config.get("test_frequency", 0))

        if self.iterations <= 0:
            raise ValueError("IterationTrainer iterations must be greater than zero")
        if self.log_frequency <= 0:
            raise ValueError("IterationTrainer log_frequency must be greater than zero")
        if self.print_freq <= 0:
            raise ValueError("IterationTrainer print_freq must be greater than zero")
        for name in (
            "checkpoint_frequency",
            "validation_frequency",
            "test_frequency",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"IterationTrainer {name} cannot be negative")

        self.current_iteration = 0
        self.data_cycle = 0
        self.iteration_in_cycle = 0
        self._pending_checkpoint_filenames: list[str | None] = []
        self._data_iterator: Iterator[Any] | None = None
        self._cycle_pass_complete = False
        self._cycle_has_batch = False
        self._resuming_data_cycle = False
        self._restored_dataset_cycle_state: dict[str, Any] | None = None
        self._restored_cycle_world_size: int | None = None

    def _set_data_cycle(self, data_cycle: int) -> None:
        """Advance deterministic sampler and dataset state without an epoch hook."""

        self.data_cycle = data_cycle
        self.accelerator.set_sampler_epoch(data_cycle)
        self.dataset_wrapper.set_epoch(data_cycle)

    @contextmanager
    def _synchronize_cycle_errors(self, stage: str) -> Iterator[None]:
        """Stop all ranks when cycle setup, replay, or resume fails."""
        local_error: Exception | None = None
        try:
            yield
        except Exception as error:
            local_error = error
        if dist.is_available() and dist.is_initialized():
            failed = torch.tensor(
                int(local_error is not None),
                dtype=torch.int32,
                device=self.accelerator.get_device(),
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item() and local_error is None:
                raise RuntimeError(f"Data cycle {stage} failed on another rank")
        if local_error is not None:
            raise local_error

    def _close_data_iterator(self) -> None:
        """Finish workers before a refresh can release their cached files."""
        iterator = self._data_iterator
        self._data_iterator = None
        if iterator is None:
            return
        try:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            else:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        finally:
            loader = self.train_loader
            if loader is not None and getattr(loader, "_iterator", None) is iterator:
                loader._iterator = None

    def _start_data_cycle(self, cycle: int, *, initial: bool = False) -> None:
        """Set selection state, run callbacks, then create the prepared iterator."""
        with self._synchronize_cycle_errors("selection setup"):
            if initial and self._resuming_data_cycle:
                world_size = (
                    dist.get_world_size()
                    if dist.is_available() and dist.is_initialized()
                    else 1
                )
                if self._restored_cycle_world_size not in {None, world_size}:
                    raise RuntimeError(
                        "Checkpoint data cycle requires the same world size"
                    )
                if self._restored_dataset_cycle_state is not None:
                    self.dataset_wrapper.restore_data_cycle_state(
                        self._restored_dataset_cycle_state
                    )
            self._set_data_cycle(cycle)

        self.callbacks.on_data_cycle_start(
            cycle,
            {
                "iteration": self.current_iteration,
                "iteration_in_cycle": self.iteration_in_cycle,
                "initial": initial,
                "resuming": initial and self._resuming_data_cycle,
            },
        )
        try:
            with self._synchronize_cycle_errors("loader setup"):
                if self.train_loader is None:
                    raise RuntimeError("IterationTrainer requires a train data loader")
                if (
                    initial
                    and self._restored_dataset_cycle_state is not None
                    and self.dataset_wrapper.get_data_cycle_state()
                    != self._restored_dataset_cycle_state
                ):
                    raise RuntimeError(
                        "Checkpoint data-cycle selection changed; restore the "
                        "original shard inventory or candidate pool"
                    )
                self.accelerator.set_sampler_epoch(cycle)
                self._cycle_pass_complete = False
                self._cycle_has_batch = False
                self._data_iterator = iter(self.train_loader)
        except Exception:
            self._close_data_iterator()
            raise

    def _read_training_batch(self) -> Any:
        """Skip empty collated results without counting them as training batches."""
        while True:
            batch = next(self._data_iterator)
            if batch is None or (isinstance(batch, dict) and not batch):
                continue
            self._cycle_has_batch = True
            return batch

    def _next_training_batch(self) -> Any:
        """Coordinate exhaustion before any rank enters a model forward."""
        while True:
            load_error: Exception | None = None
            exhausted = False
            batch = None
            try:
                batch = self._read_training_batch()
            except StopIteration:
                exhausted = True
                self._cycle_pass_complete = True
                if not self._cycle_has_batch:
                    load_error = RuntimeError(
                        "IterationTrainer cannot cycle an empty train data loader"
                    )
            except Exception as error:
                load_error = error

            # MAX keeps the cycle open while any rank is still on its first pass.
            status = [
                not self._cycle_pass_complete, exhausted, load_error is not None,
            ]
            if dist.is_available() and dist.is_initialized():
                status_tensor = torch.tensor(
                    status,
                    dtype=torch.int32,
                    device=self.accelerator.get_device(),
                )
                dist.all_reduce(status_tensor, op=dist.ReduceOp.MAX)
                status = status_tensor.tolist()
            if status[2]:
                if load_error is not None:
                    raise load_error
                raise RuntimeError("Training data stream failed on another rank")
            if not status[0]:
                # A replay batch fetched on a shorter rank belongs to the old
                # selection. Discard it rather than training past this boundary.
                batch = None
                self.callbacks.on_data_cycle_end(
                    self.data_cycle,
                    {
                        "iteration": self.current_iteration,
                        "iteration_in_cycle": self.iteration_in_cycle,
                        "completed": True,
                        "reason": "exhausted",
                    },
                )
                with self._synchronize_cycle_errors("worker shutdown"):
                    self._close_data_iterator()
                self.iteration_in_cycle = 0
                self._start_data_cycle(self.data_cycle + 1)
                continue
            if status[1]:
                # Every rank participates in the replay error check, including
                # ranks that already fetched their next first-pass batch.
                with self._synchronize_cycle_errors("replay"):
                    if exhausted:
                        self._data_iterator = iter(self.train_loader)
                        try:
                            batch = self._read_training_batch()
                        except StopIteration as error:
                            raise RuntimeError(
                                "IterationTrainer cannot replay an empty train data loader"
                            ) from error
            return batch

    def _accumulation_pending(self) -> bool:
        """Return whether gradients are waiting for an optimizer boundary."""

        return getattr(self.accelerator, "accumulation_counter", 0) > 0

    def _queue_checkpoint(self, filename: str | None) -> None:
        """Queue one checkpoint filename without duplicating pending requests."""

        if filename not in self._pending_checkpoint_filenames:
            self._pending_checkpoint_filenames.append(filename)

    def _flush_pending_checkpoints(self) -> bool:
        """Write queued checkpoints at a completed accumulation boundary."""

        if self._accumulation_pending():
            return False

        filenames = self._pending_checkpoint_filenames
        if dist.is_available() and dist.is_initialized():
            pending = torch.tensor(
                bool(filenames),
                dtype=torch.bool,
                device=self.accelerator.get_device(),
            )
            dist.all_reduce(pending, op=dist.ReduceOp.MAX)
            if not pending.item():
                return False
            requests: list[list[str | None]] = [
                [] for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(requests, filenames)
            filenames = list(
                dict.fromkeys(
                    filename
                    for rank_requests in requests
                    for filename in rank_requests
                )
            )
        self._pending_checkpoint_filenames = []
        if not filenames:
            return False

        self.accelerator.wait_for_everyone("before deferred checkpoint save")
        for filename in filenames:
            super().save_checkpoint(self.current_iteration, filename=filename)
        self.callbacks.on_checkpoint(
            self.current_iteration,
            self.current_metrics,
        )
        self.accelerator.wait_for_everyone("after deferred checkpoint save")
        return True

    def save_checkpoint(
        self,
        epoch: int,
        filename: str | None = None,
    ) -> None:
        """Save now or defer callback-driven saves until gradients are applied."""

        if self._accumulation_pending():
            self._queue_checkpoint(filename)
            return
        super().save_checkpoint(epoch, filename=filename)

    def _perform_training(self) -> None:
        """Consume training batches until the configured iteration limit."""

        self.logger.info(f"Starting training for {self.iterations} iterations")
        self.accelerator.wait_for_everyone("before on_training_start callbacks")
        self.callbacks.on_training_start()
        self.accelerator.wait_for_everyone("after on_training_start callbacks")

        if self.current_iteration == 0:
            self.perform_baseline_evaluation()
            self.set_models_mode("train")

        self._start_data_cycle(self.data_cycle, initial=True)
        try:
            with self._synchronize_cycle_errors("resume"):
                for _ in range(self.iteration_in_cycle):
                    try:
                        self._read_training_batch()
                    except StopIteration:
                        self._cycle_pass_complete = True
                        self._data_iterator = iter(self.train_loader)
                        try:
                            self._read_training_batch()
                        except StopIteration as error:
                            raise RuntimeError(
                                "Checkpoint loader position is incompatible with "
                                "the current training loader"
                            ) from error
            if self.iteration_in_cycle:
                first_pass_pending = not self._cycle_pass_complete
                if dist.is_available() and dist.is_initialized():
                    pending = torch.tensor(
                        int(first_pass_pending),
                        dtype=torch.int32,
                        device=self.accelerator.get_device(),
                    )
                    dist.all_reduce(pending, op=dist.ReduceOp.MAX)
                    first_pass_pending = bool(pending.item())
                if not first_pass_pending:
                    raise RuntimeError(
                        "Checkpoint loader position is incompatible with the "
                        "current training loader: cursor is past the shared cycle"
                    )
        except Exception:
            self._close_data_iterator()
            raise
        self._resuming_data_cycle = False
        self._restored_dataset_cycle_state = None

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        for manager in self.metric_managers.values():
            manager.reset_metrics("train")

        meters = MeterTracker()
        self.callbacks.on_train_start(
            self.current_iteration,
            {"iteration": self.current_iteration, "data_cycle": self.data_cycle},
        )
        show_progress = self.show_progress and self.accelerator.is_main_process()
        pbar = tqdm(
            total=self.iterations,
            initial=self.current_iteration,
            desc="Iterations [Train]",
            leave=False,
            disable=not show_progress,
        )
        report_pending = False
        validation_pending = False
        test_pending = False

        try:
            while self.current_iteration < self.iterations:
                batch_data = self._next_training_batch()

                batch_idx = self.current_iteration
                start = time.time()
                batch_data = self.preprocess_batch(batch_data, "train")
                batch_data = self.accelerator.to_device(batch_data)
                if not isinstance(batch_data, dict):
                    raise TypeError("preprocess_batch must return a dict")

                self.callbacks.on_batch_start(batch_idx, "train", batch_data)
                self._finalize_accumulation = (
                    self.current_iteration + 1 == self.iterations
                )
                self.accelerator.finalize_accumulation = self._finalize_accumulation
                try:
                    step_metrics = self.train_step(batch_data, batch_idx)
                finally:
                    self.accelerator.finalize_accumulation = False
                step_metrics = self.compute_probability_diagnostics(
                    step_metrics,
                    batch_data,
                )
                step_metrics["batch_time"] = time.time() - start
                self.callbacks.on_batch_end(batch_idx, "train", batch_data)

                batch_size = self._get_batch_size(batch_data)
                meters.update(step_metrics, batch_size)
                self.global_step += 1
                self.current_iteration = self.global_step
                self.current_epoch = self.current_iteration
                self.iteration_in_cycle += 1
                pbar.update(1)
                pbar.set_postfix(meters.get_postfix(self.pbar_metrics["train"]))

                if self.current_iteration % self.print_freq == 0:
                    metric_text = "  ".join(
                        f"{key.capitalize()}: {value:.4f}"
                        for key, value in meters.get_averages().items()
                    )
                    self.logger.info(
                        f"Iteration {self.current_iteration}/{self.iterations} "
                        f"{metric_text}"
                    )

                is_final = self.current_iteration == self.iterations
                should_log = self.current_iteration % self.log_frequency == 0
                should_validate = (
                    self.validation_frequency > 0
                    and self.current_iteration % self.validation_frequency == 0
                )
                should_test = (
                    self.test_frequency > 0
                    and self.current_iteration % self.test_frequency == 0
                )
                report_pending = report_pending or should_log
                validation_pending = validation_pending or should_validate
                test_pending = test_pending or should_test
                checkpoint_requested = (
                    self.checkpoint_frequency > 0
                    and self.current_iteration % self.checkpoint_frequency == 0
                )
                if checkpoint_requested:
                    self._queue_checkpoint("latest.pth")
                accumulation_pending = self._accumulation_pending()
                if not accumulation_pending:
                    self.broadcast_stop_training()
                stop_at_boundary = self.stop_training and not accumulation_pending
                if stop_at_boundary or is_final:
                    self._queue_checkpoint("latest.pth")

                if is_final and accumulation_pending:
                    raise RuntimeError(
                        "The final iteration left accumulated gradients pending. "
                        "train_step() must pass the trainer's finalization flag to "
                        "the accelerator optimizer step."
                    )

                if stop_at_boundary or is_final:
                    report_pending = True
                should_report = not accumulation_pending and any(
                    (report_pending, validation_pending, test_pending)
                )
                if not should_report:
                    self._flush_pending_checkpoints()
                    if self.stop_training and not self._accumulation_pending():
                        self.logger.info(
                            f"Training stopped early at iteration "
                            f"{self.current_iteration}"
                        )
                        break
                    continue

                self.accelerator.wait_for_everyone(
                    f"after training iteration {self.current_iteration}"
                )
                train_metrics: dict[str, float] = meters.get_averages()
                for manager_name, manager in self.metric_managers.items():
                    manager.set_epoch(self.current_iteration)
                    train_metrics.update(manager.compute("train"))
                    self.accelerator.wait_for_everyone(
                        f"after computing train metrics for {manager_name}"
                    )
                    train_metrics.update(
                        manager.compute_epoch_diagnostics("train")
                    )
                    manager.generate_plots(self.current_iteration, "train")
                    self.accelerator.wait_for_everyone(
                        f"after generating train plots for {manager_name}"
                    )
                self.set_metrics("train", train_metrics)
                self.callbacks.on_train_end(
                    self.current_iteration,
                    train_metrics,
                )

                general_logs = self.generate_epoch_logs(self.current_iteration)
                general_logs["state/data_cycle"] = float(self.data_cycle)
                general_logs["state/iteration_in_cycle"] = float(
                    self.iteration_in_cycle
                )
                self.set_metrics("general", general_logs)

                if (validation_pending or is_final) and self._eval_loader_available(
                    "validation"
                ):
                    self.callbacks.on_validation_start(self.current_iteration)
                    validation_metrics = self.validation_epoch()
                    self.set_metrics("validation", validation_metrics)
                    self.callbacks.on_validation_end(
                        self.current_iteration,
                        validation_metrics,
                    )

                if (test_pending or is_final) and self._eval_loader_available(
                    "test"
                ):
                    self.callbacks.on_test_start(self.current_iteration)
                    test_metrics = self.test_epoch()
                    self.set_metrics("test", test_metrics)
                    self.callbacks.on_test_end(
                        self.current_iteration,
                        test_metrics,
                    )

                logs = self.compile_epoch_logs()
                self.callbacks.on_iteration_end(self.current_iteration, logs)
                self.log_metrics(self.current_iteration)
                report_pending = False
                validation_pending = False
                test_pending = False

                self.broadcast_stop_training()
                if self.stop_training:
                    self._queue_checkpoint("latest.pth")

                self._flush_pending_checkpoints()
                if self.stop_training:
                    self.logger.info(
                        f"Training stopped early at iteration "
                        f"{self.current_iteration}"
                    )
                    break

                if self.current_iteration < self.iterations:
                    meters = MeterTracker()
                    for manager in self.metric_managers.values():
                        manager.reset_metrics("train")
                    self.set_models_mode("train")
                    self.callbacks.on_train_start(
                        self.current_iteration,
                        {
                            "iteration": self.current_iteration,
                            "data_cycle": self.data_cycle,
                        },
                    )
            self.callbacks.on_data_cycle_end(
                self.data_cycle,
                {
                    "iteration": self.current_iteration,
                    "iteration_in_cycle": self.iteration_in_cycle,
                    "completed": False,
                    "reason": "training_end",
                },
            )
        finally:
            pbar.close()
            self._close_data_iterator()
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            self._finalize_accumulation = False
            self.accelerator.finalize_accumulation = False

        self.accelerator.wait_for_everyone("Iteration training complete")
        self.logger.info("Iteration training completed")

    def _build_checkpoint_payload(self, iteration: int) -> dict[str, Any]:
        """Build a checkpoint with exact iteration and loader-cycle progress."""

        payload = super()._build_checkpoint_payload(iteration)
        payload.update(
            {
                "epoch": self.data_cycle,
                "iteration": iteration,
                "data_cycle": self.data_cycle,
                "iteration_in_cycle": self.iteration_in_cycle,
                "data_cycle_world_size": (
                    dist.get_world_size()
                    if dist.is_available() and dist.is_initialized()
                    else 1
                ),
                "dataset_cycle_state": self.dataset_wrapper.get_data_cycle_state(),
            }
        )
        return payload

    def restore_progress_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore iteration count and the finite-loader resume cursor."""

        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]
        if "epochs_no_improvement" in checkpoint:
            self.epochs_no_improvement = int(checkpoint["epochs_no_improvement"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.current_iteration = int(
            checkpoint.get("iteration", self.global_step)
        )
        self.global_step = self.current_iteration
        self.current_epoch = self.current_iteration
        self.data_cycle = int(
            checkpoint.get("data_cycle", checkpoint.get("epoch", 0))
        )
        self.iteration_in_cycle = int(checkpoint.get("iteration_in_cycle", 0))
        if self.data_cycle < 0 or self.iteration_in_cycle < 0:
            raise ValueError("Checkpoint data cycle and cursor must be nonnegative")
        self._resuming_data_cycle = True
        self._restored_dataset_cycle_state = checkpoint.get("dataset_cycle_state")
        self._restored_cycle_world_size = checkpoint.get("data_cycle_world_size")
        self._set_data_cycle(self.data_cycle)
        for manager in self.metric_managers.values():
            manager.set_epoch(self.current_iteration)
        self.logger.info(
            f"Resuming from iteration {self.current_iteration}, "
            f"data cycle {self.data_cycle}, position {self.iteration_in_cycle}"
        )

__all__ = ["IterationTrainer"]
