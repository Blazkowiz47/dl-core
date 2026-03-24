"""F1 Score metric implementation - pure function."""

from typing import Any, Dict, Literal

import numpy as np
from sklearn.metrics import f1_score

from dl_core.core.registry import register_metric
from dl_core.core.base_metric import BaseMetric


@register_metric("f1")
class F1Metric(BaseMetric):
    """
    F1 Score metric for classification tasks.

    Pure function implementation - no state, no distributed logic.
    Supports both binary and multiclass classification.

    COMPUTATION MODE: GATHER_THEN_COMPUTE
    F1 score computation is sensitive to class distribution, so we
    gather all data before computing.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize F1 metric.

        Args:
            config: Optional configuration dictionary
                   - num_classes: Number of classes (default: 2)
                   - average: Averaging strategy (default: 'binary')
        """
        super().__init__(config)
        self.num_classes = self.config.get("num_classes", 2)
        default_avg = "binary" if self.num_classes == 2 else "macro"
        self.average: Literal["micro", "macro", "samples", "weighted", "binary"] = (
            self.config.get("average", default_avg)
        )

    def compute(
        self, predictions: np.ndarray, labels: np.ndarray, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Compute F1 score from predictions and labels.

        Args:
            predictions: Model predictions
                        - Shape: (N,) for class indices
                        - Shape: (N, num_classes) for probabilities/logits
            labels: Ground truth labels, shape (N,)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'f1' key

        Example:
            >>> metric = F1Metric({"num_classes": 2})
            >>> result = metric.compute(
            ...     predictions=np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
            ...     labels=np.array([1, 0, 1])
            ... )
            >>> print(result)  # doctest: +SKIP
            {"f1": 100.0}
        """
        if len(predictions) == 0 or len(labels) == 0:
            return {"f1": 0.0}

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

        # Compute F1 score
        try:
            f1 = f1_score(labels, pred_classes, average=self.average, zero_division=0.0)  # type: ignore
            f1 *= 100.0
            return {"f1": float(f1)}
        except ValueError as e:
            self.log.warning(f"Could not compute F1 score: {e}")
            return {"f1": 0.0}
