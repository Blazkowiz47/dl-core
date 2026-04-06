"""Metric logger callback for enhanced metric logging and tracking."""

from typing import Any, Dict, Optional

from dl_core.core.base_callback import Callback
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_callback


@register_callback("metric_logger")
class MetricLoggerCallback(Callback):
    """
    Callback for enhanced metric logging and tracking.

    Logs metrics to console and maintains a history of all metrics
    for later analysis.

    Config format (in callbacks list):
        callbacks:
          - name: metric_logger
            params:
              log_frequency: 1  # Log every N epochs
    """

    CONFIG_FIELDS = Callback.CONFIG_FIELDS + [
        config_field(
            "log_frequency",
            "int",
            "Log metrics every N epochs.",
            default=1,
        )
    ]

    def __init__(self, log_frequency: int = 1, **kwargs):
        """
        Initialize metric logger callback.

        Args:
            log_frequency: How often to log metrics (every N epochs)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.log_frequency = log_frequency
        self.metric_history: Dict[int, Dict[str, Any]] = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Log metrics at end of epoch.

        Note: Assumes metrics in logs are already synchronized across GPUs.
        Metric synchronization should be handled by MetricManagers during
        their compute() phase, not in callbacks.
        """
        if not logs:
            return

        if not self.enabled:
            return

        # Call super() for rank filtering (only runs on main process after this point)
        super().on_epoch_end(epoch, logs)

        # Only rank 0 reaches here - store metrics in history
        self.metric_history[epoch] = logs.copy()

        # Log if frequency matches
        if epoch % self.log_frequency != 0:
            return

        # Enhanced logging
        metric_strs = []
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metric_strs.append(f"{key}: {value:.4f}")

        if metric_strs:
            self.logger.info(f"Epoch {epoch} metrics - {', '.join(metric_strs)}")

    def get_metric_history(self) -> Dict[int, Dict[str, Any]]:
        """Get the complete metric history."""
        return self.metric_history.copy()
