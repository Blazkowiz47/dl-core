"""Iteration-based trainer lifecycle for streaming and cyclic datasets."""

from __future__ import annotations

import time
from typing import Any, Iterator

from tqdm import tqdm

from dl_core.core.config_metadata import config_field
from dl_core.utils import MeterTracker

from .base_trainer import EpochTrainer


class IterationTrainer(EpochTrainer):
    """Train for a fixed number of batches instead of dataset epochs.

    One iteration consumes one batch on every distributed rank. Finite training
    loaders are restarted locally when exhausted; infinite loaders continue
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

    def _set_data_cycle(self, data_cycle: int) -> None:
        """Advance deterministic sampler and dataset state without an epoch hook."""

        self.data_cycle = data_cycle
        self.accelerator.set_sampler_epoch(data_cycle)
        self.dataset_wrapper.set_epoch(data_cycle)

    @staticmethod
    def _loader_has_batches(loader: Any) -> bool:
        """Return whether an optional sized or streaming loader can be evaluated."""

        if loader is None:
            return False
        try:
            return len(loader) > 0
        except TypeError:
            return True

    def _perform_training(self) -> None:
        """Consume training batches until the configured iteration limit."""

        self.logger.info(f"Starting training for {self.iterations} iterations")
        self.accelerator.wait_for_everyone("before on_training_start callbacks")
        self.callbacks.on_training_start()
        self.accelerator.wait_for_everyone("after on_training_start callbacks")

        if self.current_iteration == 0:
            self.perform_baseline_evaluation()
            self.set_models_mode("train")

        if self.train_loader is None:
            raise RuntimeError("IterationTrainer requires a train data loader")

        self._set_data_cycle(self.data_cycle)
        data_loader = self.train_loader
        if data_loader is None:
            raise RuntimeError("IterationTrainer requires a train data loader")
        data_iterator: Iterator[Any] = iter(data_loader)

        for _ in range(self.iteration_in_cycle):
            try:
                next(data_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Checkpoint loader position is incompatible with the current "
                    "training loader"
                ) from exc

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

        try:
            while self.current_iteration < self.iterations:
                try:
                    batch_data = next(data_iterator)
                except StopIteration:
                    self._set_data_cycle(self.data_cycle + 1)
                    self.iteration_in_cycle = 0
                    data_loader = self.train_loader
                    if data_loader is None:
                        raise RuntimeError(
                            "IterationTrainer lost its train data loader"
                        )
                    data_iterator = iter(data_loader)
                    try:
                        batch_data = next(data_iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "IterationTrainer cannot cycle an empty train data loader"
                        ) from exc

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
                step_metrics = self.train_step(batch_data, batch_idx)
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
                should_checkpoint = (
                    self.checkpoint_frequency > 0
                    and self.current_iteration % self.checkpoint_frequency == 0
                )
                if not any(
                    (
                        should_log,
                        should_validate,
                        should_test,
                        should_checkpoint,
                        is_final,
                    )
                ):
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

                if (should_validate or is_final) and self._loader_has_batches(
                    self.validation_loader
                ):
                    self.callbacks.on_validation_start(self.current_iteration)
                    validation_metrics = self.validation_epoch()
                    self.set_metrics("validation", validation_metrics)
                    self.callbacks.on_validation_end(
                        self.current_iteration,
                        validation_metrics,
                    )

                if (should_test or is_final) and self._loader_has_batches(
                    self.test_loader
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

                saved_latest = should_checkpoint or is_final
                if saved_latest:
                    self.save_checkpoint(
                        self.current_iteration,
                        filename="latest.pth",
                    )
                    self.callbacks.on_checkpoint(
                        self.current_iteration,
                        self.current_metrics,
                    )

                self.broadcast_stop_training()
                if self.stop_training:
                    if not saved_latest:
                        self.save_checkpoint(
                            self.current_iteration,
                            filename="latest.pth",
                        )
                        self.callbacks.on_checkpoint(
                            self.current_iteration,
                            self.current_metrics,
                        )
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
        finally:
            pbar.close()
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            self._finalize_accumulation = False

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
            }
        )
        return payload

    def restore_progress_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore iteration count and the finite-loader resume cursor."""

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
        self._set_data_cycle(self.data_cycle)
        for manager in self.metric_managers.values():
            manager.set_epoch(self.current_iteration)
        self.logger.info(
            f"Resuming from iteration {self.current_iteration}, "
            f"data cycle {self.data_cycle}, position {self.iteration_in_cycle}"
        )

__all__ = ["IterationTrainer"]
