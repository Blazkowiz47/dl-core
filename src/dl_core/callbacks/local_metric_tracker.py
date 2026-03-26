"""Callback that persists scalar metrics as per-metric JSONL streams."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch

from dl_core.core.base_callback import Callback
from dl_core.core.registry import register_callback


def _extract_scalars(logs: dict[str, Any] | None) -> dict[str, float]:
    """Extract scalar metrics from a callback log payload."""
    if not logs:
        return {}

    scalars: dict[str, float] = {}
    for key, value in logs.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
            continue
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalars[key] = float(value.item())
            continue
        if hasattr(value, "item") and callable(value.item):
            try:
                scalars[key] = float(value.item())
            except Exception:
                continue
    return scalars


def _sanitize_metric_filename(metric_name: str) -> str:
    """Convert a metric name into a stable JSONL filename."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", metric_name).strip("._-")
    return sanitized or "metric"


def _is_phase_metric(metric_name: str) -> bool:
    """Return whether a metric belongs to a train/validation/test phase."""
    return metric_name.startswith(("train/", "validation/", "test/"))


@register_callback("local_metric_tracker")
class LocalMetricTrackerCallback(Callback):
    """Append scalar metric values to per-metric JSONL files under run artifacts."""

    def __init__(self, log_frequency: int = 1, **kwargs: Any) -> None:
        """
        Initialize the local metric tracker callback.

        Args:
            log_frequency: Persist metrics every N epochs
            **kwargs: Additional callback parameters
        """
        super().__init__(log_frequency=log_frequency, **kwargs)
        self.log_frequency = log_frequency

    def _append_scalars(
        self,
        epoch: int,
        logs: dict[str, Any] | None,
        *,
        phase_only: bool,
    ) -> None:
        """Append scalar metrics to per-metric JSONL files."""
        if not self.enabled:
            return
        if not self.is_main_process():
            return
        if epoch % self.log_frequency != 0:
            return

        scalars = _extract_scalars(logs)
        if phase_only:
            scalars = {
                key: value for key, value in scalars.items() if _is_phase_metric(key)
            }
        else:
            scalars = {
                key: value for key, value in scalars.items() if not _is_phase_metric(key)
            }

        for metric_name, value in scalars.items():
            payload = {
                "metric": metric_name,
                "step": epoch,
                "epoch": epoch,
                "value": value,
            }
            filename = f"{_sanitize_metric_filename(metric_name)}.jsonl"
            self.trainer.artifact_manager.append_jsonl(
                Path("metrics") / "series" / filename,
                payload,
            )

    def on_train_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Append train metrics at the end of a train epoch."""
        super().on_train_end(epoch, logs)
        self._append_scalars(epoch, logs, phase_only=True)

    def on_validation_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Append validation metrics at the end of a validation epoch."""
        super().on_validation_end(epoch, logs)
        self._append_scalars(epoch, logs, phase_only=True)

    def on_test_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Append test metrics at the end of a test epoch."""
        super().on_test_end(epoch, logs)
        self._append_scalars(epoch, logs, phase_only=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Append non-phase epoch metrics after train/validation/test complete."""
        super().on_epoch_end(epoch, logs)
        self._append_scalars(epoch, logs, phase_only=False)
