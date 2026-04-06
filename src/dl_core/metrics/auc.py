"""AUC (Area Under ROC Curve) metric implementation - pure function."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_metric
from dl_core.core.base_metric import BaseMetric


@register_metric("auc")
class AUCMetric(BaseMetric):
    """
    Area Under the ROC Curve (AUC) metric.

    Pure function implementation - no state, no distributed logic.
    Supports both binary and multiclass classification.

    COMPUTATION MODE: GATHER_THEN_COMPUTE
    This metric requires global data to compute ROC curve accurately.
    """

    CONFIG_FIELDS = [
        config_field(
            "num_classes",
            "int",
            "Expected number of classes for AUC computation.",
            default=2,
        )
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize AUC metric.

        Args:
            config: Optional configuration dictionary
                   - num_classes: Number of classes (default: 2)
        """
        super().__init__(config)
        self.num_classes = self.config.get("num_classes", 2)

    def compute(
        self, predictions: np.ndarray, labels: np.ndarray, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Compute AUC from predictions and labels.

        Args:
            predictions: Model probability predictions
                        - Binary: shape (N,) or (N, 2)
                        - Multi-class: shape (N, num_classes)
            labels: Ground truth labels, shape (N,)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'auc' key

        Example:
            >>> metric = AUCMetric({"num_classes": 2})
            >>> result = metric.compute(
            ...     predictions=np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
            ...     labels=np.array([1, 0, 1])
            ... )
            >>> print(result)  # doctest: +SKIP
            {"auc": 100.0}
        """
        if len(predictions) == 0 or len(labels) == 0:
            return {"auc": 0.0}

        try:
            if self.num_classes == 2:
                # Binary classification
                if predictions.ndim > 1 and predictions.shape[1] >= 2:
                    # Use probability of positive class
                    scores = predictions[:, 1]
                elif predictions.ndim > 1 and predictions.shape[1] == 1:
                    # Single column - use as scores
                    scores = predictions[:, 0]
                else:
                    # Already 1D scores
                    scores = predictions

                auc = roc_auc_score(labels, scores)
            else:
                # Multiclass classification
                auc = roc_auc_score(
                    labels, predictions, multi_class="ovr", average="macro"
                )

            return {"auc": float(auc) * 100.0}

        except ValueError as e:
            # Handle case where only one class is present or other issues
            self.log.warning(f"Could not compute AUC: {e}")
            return {"auc": 0.0}
