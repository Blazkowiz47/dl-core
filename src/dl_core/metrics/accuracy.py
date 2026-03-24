"""Accuracy metric implementation - pure function."""

from typing import Any, Dict

import numpy as np

from dl_core.core.registry import register_metric
from dl_core.core.base_metric import BaseMetric


@register_metric("accuracy")
class AccuracyMetric(BaseMetric):
    """
    Accuracy metric for classification tasks.

    Pure function implementation - no state, no distributed logic.
    Supports both binary and multiclass classification.

    COMPUTATION MODE: COMPUTE_THEN_AVERAGE
    This metric can be computed locally and averaged across ranks.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize accuracy metric.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.num_classes = self.config.get("num_classes", 2)

    def compute(
        self, predictions: np.ndarray, labels: np.ndarray, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Compute accuracy from predictions and labels.

        Args:
            predictions: Model predictions
                        - Shape: (N,) for class indices
                        - Shape: (N, num_classes) for probabilities/logits
            labels: Ground truth labels, shape (N,)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'accuracy' key

        Example:
            >>> metric = AccuracyMetric()
            >>> result = metric.compute(
            ...     predictions=np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
            ...     labels=np.array([1, 0, 1])
            ... )
            >>> print(result)
            {"accuracy": 100.0}
        """
        if len(predictions) == 0 or len(labels) == 0:
            return {"accuracy": 0.0}

        # Get predicted classes
        if predictions.ndim > 1:
            # Probabilities/logits - take argmax
            pred_classes = np.argmax(predictions, axis=1)
        else:
            # Already class indices
            pred_classes = predictions.astype(int)

        # Ensure labels are integers
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        labels = labels.astype(int)

        # Compute accuracy
        correct = np.sum(pred_classes == labels)
        total = len(labels)

        accuracy = (correct / total) * 100.0 if total > 0 else 0.0

        return {"accuracy": float(accuracy)}
