"""ACT halt-step summary metric."""

from typing import Any, Dict

import numpy as np

from dl_core.core.base_metric import BaseMetric
from dl_core.core.registry import register_metric


@register_metric("halt_steps")
class HaltStepsMetric(BaseMetric):
    """Compute per-label halt-step mean and std."""

    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        steps = kwargs.get("steps")
        if not isinstance(labels, np.ndarray) or labels.size == 0:
            return {}
        if not isinstance(steps, np.ndarray) or steps.size == 0:
            return {}

        flat_labels = labels.reshape(-1)
        flat_steps = steps.reshape(-1)
        if flat_labels.shape[0] != flat_steps.shape[0]:
            return {}

        results: Dict[str, float] = {}
        for label in np.unique(flat_labels):
            label_steps = flat_steps[flat_labels == label]
            label_key = int(label)
            results[f"halt_steps_label_{label_key}_mean"] = float(label_steps.mean())
            results[f"halt_steps_label_{label_key}_std"] = float(label_steps.std())

        return results
