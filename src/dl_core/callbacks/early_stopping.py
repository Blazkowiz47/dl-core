"""Early stopping callback for stopping training based on monitored metrics."""

from typing import Any, Dict, List, Optional

from dl_core.core.base_callback import Callback
from dl_core.core.registry import register_callback


@register_callback("early_stopping")
class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on monitored metrics.

    Supports both single-metric and multi-metric monitoring with optional target values.
    Can stop when ANY metric reaches target or when ALL metrics reach targets.

    Single metric format (backward compatible):
        callbacks:
          early_stopping:
            monitor: validation/[global] eer
            mode: min
            patience: 10
            target_value: 0.0
            target_mode: less_equal

    Multi-metric format:
        callbacks:
          early_stopping:
            stop_condition: any  # or "all"
            patience: 10
            metrics:
              - monitor: validation/[global] eer
                mode: min
                target_value: 0.0
                target_mode: less_equal
              - monitor: validation/[global] apcer
                mode: min
                target_value: 0.01
    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        patience: int = 10,
        target_value: Optional[float] = None,
        target_mode: str = "less_equal",
        metrics: Optional[List[Dict[str, Any]]] = None,
        stop_condition: str = "any",
        enabled: bool = True,
        min_delta: float = 0.0,
        **kwargs,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Single metric to monitor (backward compatible)
            mode: 'min' or 'max' for single metric
            patience: Number of epochs with no improvement to wait
            target_value: Target value for single metric
            target_mode: How to compare target ('exact', 'less_than', 'greater_than', 'less_equal', 'greater_equal')
            metrics: List of metric configs for multi-metric monitoring
            stop_condition: 'any' or 'all' - when to trigger stopping for multi-metric
            enabled: Whether callback is enabled
            min_delta: Minimum change to qualify as improvement
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.patience = patience
        self.stop_condition = stop_condition
        self.enabled = enabled
        self.min_delta = min_delta
        self.stopped_epoch = None

        # Convert single metric to list format
        if metrics is None:
            if monitor is None:
                raise ValueError("Either 'monitor' or 'metrics' must be provided")
            metrics = [
                {
                    "monitor": monitor,
                    "mode": mode or "min",
                    "target_value": target_value,
                    "target_mode": target_mode,
                }
            ]

        self.metrics_config = metrics

        # Initialize tracking for each metric
        self.metric_states = {}
        for metric_cfg in self.metrics_config:
            monitor_key = metric_cfg["monitor"]
            mode = metric_cfg.get("mode", "min")

            if mode not in ["min", "max"]:
                raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

            self.metric_states[monitor_key] = {
                "mode": mode,
                "target_value": metric_cfg.get("target_value"),
                "target_mode": metric_cfg.get("target_mode", "less_equal"),
                "best_value": None,
                "wait": 0,
                "target_reached": False,
            }

        if enabled:
            if len(self.metrics_config) == 1:
                metric = self.metrics_config[0]
                self.logger.info(
                    f"EarlyStoppingCallback enabled: monitoring '{metric['monitor']}' "
                    f"with patience {self.patience}"
                )
                if metric.get("target_value") is not None:
                    self.logger.info(
                        f"  Target: {metric['target_mode']} {metric['target_value']}"
                    )
            else:
                self.logger.info(
                    f"EarlyStoppingCallback enabled: monitoring {len(self.metrics_config)} metrics "
                    f"with stop_condition='{self.stop_condition}', patience={self.patience}"
                )
                for metric in self.metrics_config:
                    msg = f"  - {metric['monitor']} ({metric.get('mode', 'min')})"
                    if metric.get("target_value") is not None:
                        msg += f" target: {metric.get('target_mode', 'less_equal')} {metric['target_value']}"
                    self.logger.info(msg)
        else:
            self.logger.info("EarlyStoppingCallback is disabled")

    def _check_target_reached(
        self, current_value: float, target_value: float, target_mode: str
    ) -> bool:
        """Check if current value meets the target condition."""
        if target_mode == "exact":
            return current_value == target_value
        elif target_mode == "less_than":
            return current_value < target_value
        elif target_mode == "greater_than":
            return current_value > target_value
        elif target_mode == "less_equal":
            return current_value <= target_value
        elif target_mode == "greater_equal":
            return current_value >= target_value
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Check if early stopping criteria is met for all monitored metrics.

        Note: This callback MUST run on ALL processes in multi-GPU training
        so that all processes stop training together. Do NOT call super()
        to avoid rank filtering.
        """
        if not self.enabled:
            return

        if not logs:
            return

        if not self.is_main_process():
            return

        # Check each monitored metric
        metrics_ready_to_stop = []

        for monitor_key, state in self.metric_states.items():
            if monitor_key not in logs:
                continue

            current_value = logs[monitor_key]
            mode = state["mode"]
            target_value = state["target_value"]
            target_mode = state["target_mode"]

            # Check if target is reached (highest priority)
            if target_value is not None and self._check_target_reached(
                current_value, target_value, target_mode
            ):
                if not state["target_reached"]:
                    state["target_reached"] = True
                    self.logger.info(
                        f"Target reached for '{monitor_key}': "
                        f"{current_value:.4f} {target_mode} {target_value}"
                    )
                metrics_ready_to_stop.append(monitor_key)
                continue

            # Check for improvement (with min_delta)
            is_better = False
            if state["best_value"] is None:
                is_better = True
            elif mode == "min":
                is_better = current_value < (state["best_value"] - self.min_delta)
            else:  # mode == "max"
                is_better = current_value > (state["best_value"] + self.min_delta)

            if is_better:
                state["best_value"] = current_value
                state["wait"] = 0
                self.logger.debug(
                    f"'{monitor_key}' improved to {current_value:.4f}, resetting patience"
                )
            else:
                state["wait"] += 1
                self.logger.debug(
                    f"'{monitor_key}' no improvement for {state['wait']}/{self.patience} epochs "
                    f"(current: {current_value:.4f}, best: {state['best_value']:.4f})"
                )

                # Check if patience exceeded
                if state["wait"] >= self.patience:
                    metrics_ready_to_stop.append(monitor_key)

        # Decide whether to stop based on stop_condition
        should_stop = False
        if self.stop_condition == "any" and len(metrics_ready_to_stop) > 0:
            should_stop = True
            self.logger.info(
                f"Early stopping triggered (condition='any'): {metrics_ready_to_stop}"
            )
        elif self.stop_condition == "all" and len(metrics_ready_to_stop) == len(
            self.metric_states
        ):
            should_stop = True
            self.logger.info(
                f"Early stopping triggered (condition='all'): all {len(self.metric_states)} metrics met criteria"
            )

        if should_stop:
            self.stopped_epoch = epoch
            # Set stop flag on trainer
            if hasattr(self.trainer, "stop_training"):
                self.trainer.stop_training = True

    def get_state(self) -> Dict[str, Any]:
        """
        Get early stopping state for checkpoint saving.

        Returns:
            Dictionary containing metric_states and stopped_epoch
        """
        return {
            "metric_states": self.metric_states,
            "stopped_epoch": self.stopped_epoch,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore early stopping state from checkpoint.

        Args:
            state: Dictionary containing metric_states and stopped_epoch
        """
        if "metric_states" in state:
            self.metric_states = state["metric_states"]
            self.logger.info(
                f"Restored early stopping state for {len(self.metric_states)} metrics"
            )
        if "stopped_epoch" in state:
            self.stopped_epoch = state["stopped_epoch"]
