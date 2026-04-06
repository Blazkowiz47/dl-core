"""Checkpoint callback for saving model checkpoints based on monitored metrics."""

from typing import Any, Dict, Optional

from dl_core.core.base_callback import Callback
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_callback


@register_callback("checkpoint")
class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints based on monitored metrics.

    Saves checkpoints when the monitored metric improves and maintains
    a history of the best models.

    Config format (in callbacks list):
        callbacks:
          - name: checkpoint
            params:
              monitor: eer          # Metric to monitor
              mode: min             # 'min' or 'max'
            save_best_only: true  # Only save when metric improves
    """

    CONFIG_FIELDS = Callback.CONFIG_FIELDS + [
        config_field(
            "monitor",
            "str",
            "Metric key used to decide when checkpoints improve.",
            default="eer",
        ),
        config_field(
            "mode",
            "str",
            "Use 'min' when lower is better and 'max' when higher is better.",
            default="min",
        ),
        config_field(
            "save_best_only",
            "bool",
            "Only save checkpoints when the monitored metric improves.",
            default=False,
        ),
    ]

    def __init__(
        self,
        monitor: str = "eer",
        mode: str = "min",
        save_best_only: bool = False,
        **kwargs,
    ):
        """
        Initialize checkpoint callback.

        Args:
            monitor: Metric to monitor for checkpoint saving
            mode: 'min' if lower is better, 'max' if higher is better
            save_best_only: Only save when metric improves
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None
        self.best_epoch = None

        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if current metrics warrant saving a checkpoint."""
        super().on_epoch_end(epoch, logs)

        if not logs:
            # Save anyway if not monitoring specific metric
            if not self.save_best_only:
                self.trainer.save_checkpoint(epoch)
            return

        resolved_monitor = self.resolve_log_key(logs, self.monitor)
        if resolved_monitor is None:
            if not self.save_best_only:
                self.trainer.save_checkpoint(epoch)
            return

        current_value = logs[resolved_monitor]

        # Determine if this is a better value
        is_better = False
        if self.best_value is None:
            is_better = True
        elif self.mode == "min":
            is_better = current_value < self.best_value
        else:  # mode == "max"
            is_better = current_value > self.best_value

        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch

            direction = "lower" if self.mode == "min" else "higher"
            self.logger.info(
                f"New best {self.monitor}: {current_value:.4f} "
                f"({direction} is better) at epoch {epoch}"
            )

            # Trigger checkpoint save in trainer
            self.trainer.save_checkpoint(epoch)
            self.trainer.save_checkpoint(epoch, filename="best.pth")

        elif not self.save_best_only:
            # Save checkpoint even if not best
            self.trainer.save_checkpoint(epoch)
