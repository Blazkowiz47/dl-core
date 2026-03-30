"""
Base callback interface for training hooks.

This module provides the abstract base class for implementing training callbacks
that can be plugged into the training loop for extensibility.
"""

import logging
from abc import ABC
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dl_core.core.base_trainer import BaseTrainer


def _normalize_log_key(key: str) -> str:
    """Normalize a log key so separator differences still match."""

    return "".join(char for char in key.casefold() if char.isalnum())


class Callback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks provide hooks into the training loop for extensible functionality
    without modifying the core trainer code.
    """

    def __init__(self, **kwargs):
        """
        Initialize callback.

        Args:
            **kwargs: Callback-specific parameters
        """
        self.params = kwargs
        self.enabled = kwargs.get("enabled", True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.trainer: BaseTrainer

    def set_trainer(self, trainer):
        """
        Set reference to the trainer instance.

        Args:
            trainer: The trainer instance using this callback
        """
        self.trainer = trainer

    def is_main_process(self) -> bool:
        """
        Check if current process is the main process (rank 0).

        For multi-GPU training, only the main process should perform certain operations
        like logging to an external tracker, saving artifacts, or printing
        detailed logs.

        Returns:
            True if main process or single GPU, False otherwise
        """
        if self.trainer.accelerator is None:
            return True
        return self.trainer.accelerator.is_main_process()

    def gather_metric(self, value: float) -> float:
        """
        Gather and average a metric value across all processes in multi-GPU training.

        **IMPORTANT**: This method is provided for convenience but should generally
        NOT be used in callbacks. Metric synchronization should be handled by
        MetricManagers during their compute() phase to ensure all metrics are
        synchronized consistently at the source.

        Only use this if you need to synchronize custom values that aren't
        part of the regular metric pipeline.

        Args:
            value: Metric value to average (e.g., loss, accuracy)

        Returns:
            Averaged value across all processes, or original value if single GPU

        Example:
            # Only if you have a custom metric not from MetricManager
            avg_custom_value = self.gather_metric(custom_value)
        """
        if not hasattr(self.trainer, "accelerator") or self.trainer.accelerator is None:
            return value

        accelerator = self.trainer.accelerator
        if not hasattr(accelerator, "gather_for_metrics"):
            return value

        tensor = torch.tensor(value, dtype=torch.float32)
        averaged = accelerator.gather_for_metrics(tensor)
        return averaged.item()

    @staticmethod
    def resolve_log_key(logs: Dict[str, Any], requested_key: str) -> str | None:
        """Resolve one requested metric key against available log keys."""

        if requested_key in logs:
            return requested_key

        normalized_requested = _normalize_log_key(requested_key)
        matches = [
            key for key in logs if _normalize_log_key(key) == normalized_requested
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_train_start() to use this behavior.

        Args:
            logs: Dictionary of logs/metrics (e.g., config, experiment_name)
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_train_end() to use this behavior.

        Args:
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_training_finalized(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called after trainer finalization and cleanup completes.

        This hook is intended for end-of-run artifact handling that should see the
        fully finalized run directory, such as complete log files. It runs after
        accelerator cleanup, so callbacks must not assume distributed
        synchronization is still available here.

        Args:
            logs: Dictionary of final run metadata
        """
        if not self.is_main_process():
            return

    def on_train_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each training epoch.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_train_start() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics (e.g., config, experiment_name)
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_train_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each training epoch.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_train_end() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics (e.g., train_loss, train_accuracy)
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_test_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of evaluation.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_eval_start() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_test_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of evaluation.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_eval_end() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_validation_start(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Called at the beginning of validation epoch.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_validation_epoch_start() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_validation_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Called at the end of validation epoch.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_validation_epoch_end() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each epoch.

        By default, runs on ALL processes (no rank filtering).
        This is because epoch start often involves data refresh which
        needs to happen on all processes independently.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Run on all processes (no return)
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each epoch.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_epoch_end() to use this behavior.

        Important: If you need to gather metrics across GPUs, call gather_metric()
        BEFORE calling super().on_epoch_end() since gathering must happen on all processes.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics (e.g., train_loss, val_loss, metrics)
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_batch_start(
        self,
        batch: int,
        mode: Literal["train", "validation", "test"],
        batch_data: dict,
        **kwargs,
    ) -> None:
        """
        Called at the beginning of each batch.

        Args:
            batch: Current batch number
            mode: 'train' or 'test'
            batch_data: The batch data dict
            **kwargs: Additional keyword arguments
        """
        pass

    def on_batch_end(
        self,
        batch: int,
        mode: Literal["train", "validation", "test"],
        batch_data: dict,
        **kwargs,
    ) -> None:
        """
        Called at the end of each batch.

        Args:
            batch: Current batch number
            mode: 'train' or 'test'
            **kwargs: Additional keyword arguments (e.g., loss, outputs)
        """
        pass

    def on_checkpoint(self, epoch: int, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Called when a checkpoint is saved.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_checkpoint() to use this behavior.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def on_early_stop(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when early stopping is triggered.

        By default, only runs on main process (rank 0) for multi-GPU training.
        Override with super().on_early_stop() to use this behavior.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        # Default: Only run on main process
        if not self.is_main_process():
            return

    def get_state(self) -> dict | None:
        """
        Get callback state for checkpoint saving.

        Override this method in callbacks that need to persist state across
        training sessions (e.g., tracker run IDs, best metric tracking).

        Returns:
            Dictionary of state to save, or None if no state to persist

        Example:
            def get_state(self) -> dict:
                return {
                    "run_id": self.run_id,
                    "best_metric": self.best_metric,
                }
        """
        return None

    def set_state(self, state: dict) -> None:
        """
        Restore callback state from checkpoint.

        Override this method in callbacks that need to persist state across
        training sessions.

        Args:
            state: Dictionary of state to restore (from get_state())

        Example:
            def set_state(self, state: dict) -> None:
                self.run_id = state.get("run_id")
                self.best_metric = state.get("best_metric")
        """
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    Provides a unified interface to call all registered callbacks
    at the appropriate hooks.
    """

    def __init__(self, callbacks: Optional[list] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks: list[Callback] = callbacks or []
        self.trainer: BaseTrainer
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def set_trainer(self, trainer):
        """
        Set trainer reference for all callbacks.

        Args:
            trainer: The trainer instance
        """
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def append(self, callback: Callback) -> None:
        """
        Add a callback to the list.

        Args:
            callback: Callback instance to add
        """
        self.callbacks.append(callback)
        if self.trainer:
            callback.set_trainer(self.trainer)

    def __len__(self) -> int:
        """
        Return the number of callbacks in the list.

        Returns:
            Number of callbacks
        """
        return len(self.callbacks)

    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_start for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_training_start(logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_training_start", e)

    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_training_end(logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_training_end", e)
            # Synchronize after each callback to prevent rank drift
            self.trainer.accelerator.wait_for_everyone()

    def on_training_finalized(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_training_finalized for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_training_finalized(logs)
                except Exception as e:
                    self._handle_callback_error(
                        callback, "on_training_finalized", e
                    )

    def on_train_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_start for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_train_start(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_train_start", e)

    def on_train_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_train_end(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_train_end", e)
            # Synchronize after each callback to prevent rank drift
            self.trainer.accelerator.wait_for_everyone()

    def on_test_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_eval_start for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_test_start(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_test_start", e)

    def on_test_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_eval_end for all callbacks."""
        for callback in self.callbacks:
            callback_name = callback.__class__.__name__

            # LOG: Track callback enabled status on this rank
            self.trainer.logger.debug(
                f"on_test_end: Callback {callback_name} enabled={callback.enabled} for epoch {epoch}"
            )
            self.trainer.accelerator.wait_for_everyone()
            self.trainer.logger.debug(
                f"on_test_end: All ranks ready, starting callback: {callback_name} for epoch {epoch}"
            )

            if callback.enabled:
                try:
                    callback.on_test_end(epoch, logs)
                    self.trainer.logger.debug(
                        f"on_test_end: Finished {callback_name} for epoch {epoch} [SUCCESS]"
                    )
                except Exception as e:
                    self.trainer.logger.debug(
                        f"on_test_end: Finished {callback_name} for epoch {epoch} [EXCEPTION: {e}]"
                    )
                    self._handle_callback_error(callback, "on_test_end", e)

            self.trainer.logger.debug(
                f"on_test_end: Callback {callback_name} enabled={callback.enabled} for epoch {epoch} ended"
            )
            self.trainer.accelerator.wait_for_everyone()
            self.trainer.logger.debug(
                f"on_test_end: Callback {callback_name} enabled={callback.enabled} for epoch {epoch} ended"
            )

    def on_validation_start(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_validation_start for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_validation_start(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_validation_start", e)

    def on_validation_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_validation_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    self.trainer.logger.debug(
                        f"on_validation_end: Waiting for all ranks BEFORE callback: {callback.__class__.__name__} for epoch {epoch}"
                    )
                    self.trainer.accelerator.wait_for_everyone()
                    self.trainer.logger.debug(
                        f"on_validation_end: All ranks ready, starting callback: {callback.__class__.__name__} for epoch {epoch}"
                    )
                    callback.on_validation_end(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_validation_end", e)

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_start for all callbacks."""

        # CRITICAL: Synchronize BEFORE any on_epoch_start callbacks run
        # This ensures all ranks complete previous epoch's work, including slow
        # tracker uploads on rank 0,
        # before DataRefresher's accelerator.prepare() creates hidden barriers
        self.trainer.logger.debug(
            f"on_epoch_start: Waiting for all ranks BEFORE callbacks for epoch {epoch}"
        )
        self.trainer.accelerator.wait_for_everyone()
        self.trainer.logger.debug(
            f"on_epoch_start: All ranks ready, starting callbacks for epoch {epoch}"
        )

        for callback in self.callbacks:
            callback_name = callback.__class__.__name__

            # LOG: Track callback enabled status on this rank
            self.trainer.logger.debug(
                f"on_epoch_start: Callback {callback_name} enabled={callback.enabled} for epoch {epoch}"
            )

            if callback.enabled:
                try:
                    self.trainer.logger.debug(
                        f"on_epoch_start: Calling {callback_name} for epoch {epoch}"
                    )
                    callback.on_epoch_start(epoch, logs)
                    self.trainer.logger.debug(
                        f"on_epoch_start: Finished {callback_name} for epoch {epoch} [SUCCESS]"
                    )
                except Exception as e:
                    self.trainer.logger.debug(
                        f"on_epoch_start: Finished {callback_name} for epoch {epoch} [EXCEPTION: {e}]"
                    )
                    self._handle_callback_error(callback, "on_epoch_start", e)

                # CRITICAL: Synchronize AFTER each callback completes - OUTSIDE try-catch!
                # This ensures barrier executes even if callback fails, preventing deadlock
                self.trainer.logger.debug(
                    f"on_epoch_start: [INSIDE IF enabled=True] Waiting for all ranks AFTER callback: {callback_name} for epoch {epoch}"
                )
                self.trainer.accelerator.wait_for_everyone()
                self.trainer.logger.debug(
                    f"on_epoch_start: [INSIDE IF enabled=True] Passed barrier AFTER callback: {callback_name} for epoch {epoch}"
                )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback_name = callback.__class__.__name__

            # LOG: Track callback enabled status on this rank
            self.trainer.logger.debug(
                f"on_epoch_end: Callback {callback_name} enabled={callback.enabled} for epoch {epoch}"
            )

            if callback.enabled:
                try:
                    self.trainer.logger.debug(
                        f"on_epoch_end: Calling {callback_name} for epoch {epoch}"
                    )
                    callback.on_epoch_end(epoch, logs)
                    self.trainer.logger.debug(
                        f"on_epoch_end: Finished {callback_name} for epoch {epoch} [SUCCESS]"
                    )
                except Exception as e:
                    self.trainer.logger.debug(
                        f"on_epoch_end: Finished {callback_name} for epoch {epoch} [EXCEPTION: {e}]"
                    )
                    self._handle_callback_error(callback, "on_epoch_end", e)

                # CRITICAL: Synchronize AFTER each callback completes - OUTSIDE try-catch!
                # This ensures barrier executes even if callback fails, preventing deadlock
                self.trainer.logger.debug(
                    f"on_epoch_end: [INSIDE IF enabled=True] Waiting for all ranks AFTER callback: {callback_name} for epoch {epoch}"
                )
                self.trainer.accelerator.wait_for_everyone()
                self.trainer.logger.debug(
                    f"on_epoch_end: [INSIDE IF enabled=True] Passed barrier AFTER callback: {callback_name} for epoch {epoch}"
                )

        # CRITICAL: Synchronize after all on_epoch_end callbacks complete.
        # This ensures slow tracker uploads on rank 0 finish before other ranks
        # proceed to the next epoch.
        self.trainer.logger.debug(
            f"on_epoch_end: Waiting for all ranks after all callbacks for epoch {epoch}"
        )
        self.trainer.accelerator.wait_for_everyone()
        self.trainer.logger.debug(
            f"on_epoch_end: All ranks synchronized after epoch {epoch}, proceeding"
        )

    def on_batch_start(
        self,
        batch: int,
        mode: Literal["train", "validation", "test"],
        batch_data: dict,
        **kwargs,
    ) -> None:
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_batch_start(batch, mode, batch_data, **kwargs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_batch_start", e)
            # Skip barrier for batch callbacks to avoid performance issues

    def on_batch_end(
        self,
        batch: int,
        mode: Literal["train", "validation", "test"],
        batch_data: dict,
        **kwargs,
    ) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_batch_end(batch, mode, batch_data, **kwargs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_batch_end", e)
            # Skip barrier for batch callbacks to avoid performance issues

    def on_checkpoint(self, epoch: int, metrics: Dict[str, Dict[str, float]]) -> None:
        """Call on_checkpoint for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_checkpoint(epoch, metrics)
                except Exception as e:
                    self._handle_callback_error(callback, "on_checkpoint", e)

    def on_early_stop(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_early_stop for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_early_stop(epoch, logs)
                except Exception as e:
                    self._handle_callback_error(callback, "on_early_stop", e)

    def _handle_callback_error(
        self, callback: Callback, hook_name: str, error: Exception
    ) -> None:
        """
        Handle callback errors gracefully.

        Args:
            callback: The callback that raised the error
            hook_name: Name of the hook that failed
            error: The exception that was raised
        """
        error_msg = (
            f"Callback {callback.__class__.__name__}.{hook_name} failed: {error}"
        )

        self.logger.warning(error_msg)
        self.logger.warning("Traceback: ", exc_info=True)

        # Optionally disable the problematic callback
        callback.enabled = False
